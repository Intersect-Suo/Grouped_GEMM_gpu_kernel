# Grouped GEMM 设计文档（Tensor Core 版本）

## 1. 目标

实现一个可分析、可扩展的 grouped GEMM：

1. 一次 launch 处理多问题。
2. 支持异构 M/N/K。
3. 采用 persistent scheduler 做 device-side 动态调度。
4. 支持排序策略实验。
5. 支持 SIMT 与 Tensor Core 双路径。
6. 提供 shared memory / 寄存器压力的公式与实测统计。

## 2. 数据结构

公开结构见 include/grouped_gemm/grouped_gemm.cuh：

1. ProblemDesc: 单问题参数（M/N/K、A/B/C/D、ld、alpha/beta）。
2. PreparedProblemSet: 排序后的问题列表 + tile_offsets 前缀和。
3. KernelConfig:
   - cta_m, cta_n, cta_k, threads
   - persistent_ctas
   - use_tensor_cores
4. OccupancyInfo:
   - occupancy 指标
   - shared memory 指标
   - 寄存器估算值与编译器实测值

## 3. 调度策略

采用 device-side persistent scheduler：

1. 所有 CTA 共享 global_tile_counter。
2. block 内 thread0 用 atomicAdd 领取下一个 tile。
3. 通过 tile_offsets 二分定位 tile 属于哪个 problem。
4. 计算 local_tile -> (tile_m, tile_n) 映射。
5. 处理完继续领取，直到 total_tiles 耗尽。

## 4. 分块参数（CTA / Warp / MMA）

### 4.1 CTA 级

当前编译 3 套配置：

1. 64x64x16
2. 128x64x32
3. 64x128x32

每个 CTA 固定 256 线程。

### 4.2 Warp 级

- warp 数: $W = 256 / 32 = 8$
- persistent 模式下，每个 CTA 处理一个输出 tile。

### 4.3 MMA 级（Tensor Core）

- WMMA 指令形状: $16\times16\times16$
- MMA tile 参数:
  - $M_{mma}=16$
  - $N_{mma}=16$
  - $K_{mma}=16$
- 一个 CTA 内的 MMA tile 总数:
$$
T_{mma} = (CTA_M / 16)\cdot(CTA_N / 16)
$$
- 每个 warp 以 stride 方式领取 MMA tile:
$$
T_{warp} = \left\lceil \frac{T_{mma}}{8} \right\rceil
$$

## 5. 数据搬运与流水

两条计算路径都复用 cp.async 双缓冲预取：

1. stage 数: $S=2$
2. 先预取第 0 个 K-tile。
3. 每轮等待当前 stage，计算当前 stage，同时预取下一 stage。
4. 对越界和未对齐地址回退为标量加载并补零。

## 6. 资源公式

### 6.1 shared memory

设 Tensor Core 路径指示变量为 $I_{tc}$（开启为 1，否则 0），则每 CTA 动态 shared memory：

$$
S_{dyn}=S\cdot(CTA_M\cdot CTA_K + CTA_K\cdot CTA_N)\cdot sizeof(half)
+ I_{tc}\cdot CTA_M\cdot CTA_N\cdot sizeof(float)
$$

其中 $S=2$。

示例：

1. 64x64x16 + Tensor Core: $2\cdot(1024+1024)\cdot2 + 64\cdot64\cdot4 = 24576$ B
2. 128x64x32 + Tensor Core: $2\cdot(4096+2048)\cdot2 + 128\cdot64\cdot4 = 57344$ B

### 6.2 寄存器压力估算（代码内实现）

SIMT 路径估算：

$$
R_{est}^{simt} = \left(\frac{CTA_M}{16}\cdot\frac{CTA_N}{16}\right)
+ \left(\frac{CTA_N}{16}\right) + R_{misc}
$$

其中 $R_{misc}=20$。

Tensor Core 路径估算：

$$
R_{est}^{tc} = 8\cdot T_{warp} + 8 + 8 + R_{misc}
$$

解释：

1. $8\cdot T_{warp}$: 每个 MMA accumulator fragment 每 lane 持有 8 个 FP32 元素。
2. 额外 $8+8$: A/B operand fragment 开销。
3. $R_{misc}=20$: 指针、循环变量、临时寄存器预算。

### 6.3 寄存器实测（编译器）

代码使用 cudaFuncGetAttributes 获取每线程寄存器数：

$$
R_{meas/thread} = attr.numRegs
$$

并计算每 block 寄存器：

$$
R_{meas/block} = R_{meas/thread}\cdot threads\_per\_block
$$

寄存器限制下的 SM 最大 block 数：

$$
B_{reg} = \left\lfloor \frac{Regs_{SM}}{R_{meas/block}} \right\rfloor
$$

最终占用由多约束共同决定，理论 occupancy 输出为：

$$
Occ = \frac{active\_warps\_per\_sm}{max\_warps\_per\_sm}
$$

## 7. persistent grid 选择

自动模式（persistent_ctas <= 0）：

$$
grid\_ctas = \max\left(1,\min\left(sm\_count\cdot active\_blocks\_per\_sm,
\max(total\_tiles, sm\_count)\right)\right)
$$

手动模式直接使用用户给定 persistent_ctas。

## 8. 执行路径

### 8.1 grouped 路径

- run_grouped_gemm_once -> prepare_problem_set -> GroupedGemmExecutor::build -> run

### 8.2 baseline 路径

- run_baseline_per_problem: 逐问题 launch single_problem kernel

## 9. 当前实现状态

1. 已接入 WMMA Tensor Core。
2. 仍保留 SIMT 路径，支持 --tensor-cores on/off 对照。
3. 已补充寄存器压力代码（估算 + 实测）。
4. 正确性测试在 64x64x16 和 128x64x32 上通过。
