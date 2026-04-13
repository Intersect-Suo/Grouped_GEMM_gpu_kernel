# Grouped GEMM 任务书答复（基于仓库现有资料）

## 1. 任务内容回答

### 1.1 定义 problem set 数据结构

实现回答：

1. Host 侧单问题描述使用 `ProblemDesc`：
   - 维度：`m, n, k`
   - 指针：`a, b, c, d`
   - 步长：`lda, ldb, ldc, ldd`
   - 系数：`alpha, beta`
2. 一次 grouped dispatch 的预处理结果使用 `PreparedProblemSet`：
   - `sorted_problems`
   - `sorted_to_original`
   - `tile_offsets`（长度 `problem_count + 1`）
   - `total_tiles`
3. Device 侧运行时描述使用 `DeviceProblemDesc`（在 host 描述基础上增加）：
   - `tiles_m = ceil_div(m, CTA_M)`
   - `tiles_n = ceil_div(n, CTA_N)`

对任务书要求的逐条对应：

1. 支持每个问题尺寸不同：已支持。`ProblemDesc` 按问题存储独立 `m/n/k`。
2. 支持问题集合一次性传入 kernel：已支持。`PreparedProblemSet` 聚合后由 `GroupedGemmExecutor::build` 一次上传到 device。
3. 支持 kernel 在 device 侧根据问题编号读取参数：已支持。kernel 通过 `tile_offsets` 二分定位问题，再读取 `problems[problem_index]`。



### 1.2 调研并选择调度策略

实现回答：

1. 当前采用 **device-side persistent scheduler**。
2. 所有 CTA 共享一个全局原子计数器 `global_tile_counter`。
3. CTA 领取方式：`tile_index = atomicAdd(counter, 1)`，直到 `tile_index >= total_tiles` 退出。
4. 全局 tile 到问题映射方式：
   - `problem_index = binary_search(tile_offsets, tile_index)`
   - `local_tile = tile_index - tile_offsets[problem_index]`
   - `tile_m = local_tile / tiles_n`
   - `tile_n = local_tile % tiles_n`

host 与 device 分工：

1. host：排序 + 前缀和构建（`prepare_problem_set`）。
2. device：原子领取 + 二分定位 + tile 计算（`grouped_gemm_kernel`）。

对任务书“需要分析并实现”的逐条对应：

1. threadblock 如何领取 tile：通过 `atomicAdd` 动态领取，已实现。
2. tile 与 threadblock 映射关系：一个 CTA 处理一个 tile，完成后继续领取，已实现。
3. device-side 还是 host-side：采用“host 预处理 + device 持续调度”的混合方案，核心调度在 device 侧。
4. 异构问题下负载不均衡缓解：persistent 领取 + 排序策略组合，已实现并有实验数据。
5. 是否需要排序：已实现 4 种模式（none/by-K/by-tiles/by-tiles-then-K）并完成对比。

### 1.3 设计 CTA 级分块策略

实现回答：

已编译并支持 3 套 CTA 配置：

1. `64x64x16`
2. `128x64x32`
3. `64x128x32`

线程组织与计算分解：

1. 每个 CTA 固定 `256` 线程。
2. 逻辑线程组为 `16 x 16`。
3. 每线程寄存器块：
   - `RowsPerThread = CTA_M / 16`
   - `ColsPerThread = CTA_N / 16`
4. 以 warp 视角可近似理解为：每 warp 覆盖 `CTA_N` 全宽和 `CTA_M/8` 的行高子块。
5. micro-kernel 采用 SIMT FMA 累加 / WMMA 计算。

**mma-level tile 与分块参数**

- WMMA 指令形状：16x16x16
- mma-level 参数：
  - M_mma = 16
  - N_mma = 16
  - K_mma = 16
- 每个 CTA 256 线程（8 warp）
- 每个 warp 以 stride 方式领取 CTA 内多个 MMA tile


**寄存器压力估算与实测**

**估算：**

SIMT 路径估算：

$$
R_{est}^{simt} = \left(\frac{CTA_M}{16}\cdot\frac{CTA_N}{16}\right) + \left(\frac{CTA_N}{16}\right) + R_{misc},\;R_{misc}=20
$$

Tensor Core 路径估算：

设
$$
T_{mma} = (CTA_M/16)\cdot(CTA_N/16),\quad T_{warp}=\left\lceil\frac{T_{mma}}{8}\right\rceil
$$
则
$$
R_{est}^{tc} = 8\cdot T_{warp} + 8 + 8 + R_{misc},\;R_{misc}=20
$$

**实测：**

代码使用 cudaFuncGetAttributes 获取：

$$
R_{meas/thread} = attr.numRegs
$$

并计算：

$$
R_{meas/block}=R_{meas/thread}\cdot threads\_per\_block
$$

寄存器限制的 block 上界：

$$
B_{reg}=\left\lfloor\frac{Regs_{SM}}{R_{meas/block}}\right\rfloor
$$

理论 occupancy：

$$
Occ=\frac{active\_warps\_per\_sm}{max\_warps\_per\_sm}
$$


流水与搬运：

1. stage 数固定为 `2`（double buffering）。
2. `SM80+` 路径使用 `cp.async.cg.shared.global` 进行 16B 向量异步搬运。
3. 对越界或未对齐访问，自动回退为标量加载并补零。

对任务书要求逐条对应：

1. 分块与硬件匹配：已匹配。实测在该设备上 `64x64x16` 可达更高活跃块数（`active_blocks_per_sm=6`）。
2. 可解释当前分块：已可解释。64x64 配置提供更高并发，128x64/64x128 提供更大 tile 吞吐。
3. 小问题与异构问题稳定性：已验证。正确性测试通过，benchmark 覆盖 small/mixed/tail。

### 1.4 设计资源分配策略

#### 1.4.1 每 CTA shared memory

采用公式：

$$
S_{smem} = S \cdot (CTA_M \cdot CTA_K + CTA_K \cdot CTA_N) \cdot sizeof(half),\quad S=2
$$

代入后：

1. `64x64x16`：`8192 B`
2. `128x64x32`：`24576 B`
3. `64x128x32`：`24576 B`

#### 1.4.2 寄存器压力（代码可见部分）

虽然仓库未给出 `ptxas -v` 的精确寄存器计数，但从代码可见：

1. 每线程核心累加器 `accum[RowsPerThread][ColsPerThread]`。
2. `64x64x16`：累加器 `16` 个 float。
3. `128x64x32` 与 `64x128x32`：累加器 `32` 个 float。
4. 更大的 per-thread 累加块通常意味着更高寄存器压力，这与 occupancy 从 `6` 降到 `3` 的现象一致。

#### 1.4.3 occupancy 与 SM 驻留

通过 `query_occupancy` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 获取：

1. `64x64x16`：`active_blocks_per_sm=6`
2. `128x64x32`：`active_blocks_per_sm=3`
3. `64x128x32`：`active_blocks_per_sm=3`
4. 当前设备 `sm_count=24`

因此理论最大常驻 CTA：

1. `64x64x16`：`24*6=144`
2. 另外两种：`24*3=72`

$$\text{Occupancy} = \frac{\text{Active Blocks Per SM} \times \text{Warps Per Block}}{\text{Max Warps Per SM}} \times 100\%$$

我本机设备型号是NVIDIA GeForce RTX 4060，Max Warps Per SM=48。此外，由于一个block设定为256个线程，所以Warps Per Block=256/32=8。

- 64x64x16配置：$\text{Occupancy} = \frac{\text{6} \times \text{8}}{\text{48}} \times 100\% = 100\%$
- 128x64x32配置：$\text{Occupancy} = \frac{\text{3} \times \text{8}}{\text{48}} \times 100\% = 50\%$

可见，64x64x16配置occupency高，并行度高。

#### 1.4.4 persistent grid 大小选择

自动模式（`persistent_ctas<=0`）使用：

$$
grid\_ctas = \max\left(1,\; \min\left(sm\_count\cdot active\_blocks\_per\_sm,\; \max(total\_tiles, sm\_count)\right)\right)
$$

手动模式（`persistent_ctas>0`）直接采用用户指定值。

对任务书要求逐条对应：

1. occupancy 分析：已提供接口与实测值。
2. grid 大小与 total tiles 关系：已通过上述公式明确。
3. 资源配置影响吞吐与负载均衡：已在 benchmark 中通过 config 与 persistent 扫描体现。

### 1.5 实现 grouped GEMM kernel

实现回答：

1. 一次 launch 处理多个问题：`grouped_gemm_kernel` + `PreparedProblemSet` 已实现。
2. CTA 完成当前 tile 后继续领取：`while(true)` + `atomicAdd` 已实现。
3. 优化手段：
   - shared memory tiled GEMM
   - `cp.async` 异步搬运（SM80+）
   - 2-stage pipeline
   - 边界回退路径
4. 数据组织：A/B 分 stage 预取到 shared memory，计算与下一 stage 预取重叠。

正确性与边界验证：

1. 多尺寸异构问题：通过。
2. 4 种排序模式：通过。
3. baseline 对比：通过。
4. 空问题集：通过。
5. 当前实测输出：两套配置均 `All correctness checks passed.`，最大误差 `4.76837e-07`。

说明：

1. 当前未使用 Tensor Core MMA 指令，但已满足“Tensor Core 或现代 GEMM 优化方式”中的“现代优化方式”分支。

### 1.6 分析问题排序与负载均衡

实现了并对比的策略：

1. `none`
2. `sort_by_k`
3. `sort_by_tiles`
4. `sort_by_tiles_then_k`

结果层面（来自 `bench/results_20260331` 和报告）：

1. best mode 出现频次（48 run）：
   - `none=21`
   - `sort_by_k=20`
   - `sort_by_tiles=5`
   - `sort_by_tiles_then_k=2`
2. speedup（best grouped 对 baseline）区间：`10.16x ~ 95.39x`。
3. 场景级平均 speedup：
   - `grid_workload_config`: `39.60x`
   - `sweep_problems_mixed_64x64x16`: `27.84x`
   - `sweep_problems_tail_128x64x32`: `15.32x`
   - `sweep_persistent_mixed_128x64x32`: `26.71x`
   - `sweep_persistent_tail_128x64x32`: `17.80x`
4. 尾部负载指标（全量结果均值）：
   - `none`: `tail_share=0.100`
   - `sort_by_k`: `0.018`
   - `sort_by_tiles`: `0.015`
   - `sort_by_tiles_then_k`: `0.015`

为什么有时排序更有效：

1. 当问题数量较小或分布长尾明显时，排序能把“重任务”前移，降低尾部拖尾。
2. 当问题数量很大时，persistent 原子领取本身已较均衡，排序收益下降，甚至不排序更快（报告中的 512/1024 问题规模可观察到该现象）。
3. `by_k` 更针对每 tile 计算代价差异（K 维影响循环步数）。
4. `by_tiles` 更针对任务数量差异（问题 tile 数不均）。

当前实现边界：

1. `sort_by_tiles_then_k` 在代码中当前与 `sort_by_tiles` 复用比较器（保留枚举是为未来扩展）。

### 1.7 与 baseline 对比

baseline 定义：

1. `run_baseline_per_problem` 按问题逐个 launch single-problem kernel。
2. launch 次数约等于非空问题数。

对比结果：

1. grouped launch 数恒为 1（results.csv 全量验证）。
2. baseline launch 数在本批实验中范围 `64~1024`。
3. 当单个problem大小较小时, 例如(m,n,k)=(200,100,80)，grouped 优于 baseline; 当单个problem大小较大时，例如(m,n,k)=(2000,1000,1500)，baseline优于grouped。（原因分析在"benchmark_full_report_20260331.md"中）
4. best speedup 区间 `10.16x~95.39x`。

优势来源分析：

1. 合并 launch 显著降低 host 启动开销。
2. persistent 调度降低异构问题静态分配不均。
3. `cp.async + double buffering` 改善搬运与计算重叠。

适用边界：

1. 排序并非恒定最优，需要按 workload/规模实测选择。
2. 当前实现以原型可分析性为主，若追求更高上限可进一步引入 Tensor Core 专用内核。
