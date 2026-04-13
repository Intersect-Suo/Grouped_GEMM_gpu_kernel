# Grouped GEMM GPU Kernel Prototype

本仓库实现了面向异构小规模问题集合的 Grouped GEMM 原型，并在原有 persistent scheduler 基础上新增了 Tensor Core 路径与寄存器压力定量分析。

## 1. 核心能力

- 一次 kernel launch 处理多问题
- 支持每个问题独立的 M/N/K、alpha/beta 和矩阵指针
- device-side persistent tile scheduler（CTA 动态领取 tile）
- host 侧排序策略对比（none / by-K / by-tiles / by-tiles-then-K）
- baseline（每问题一次 launch）对照
- 双路径内核：
  - SIMT 路径：cp.async + 2-stage double buffering
  - Tensor Core 路径：WMMA 16x16x16 MMA
- 资源分析接口：occupancy + shared memory + 寄存器压力（估算与实测）

## 2. 目录结构

- include/grouped_gemm/grouped_gemm.cuh: 公开 API、配置和资源统计结构
- src/grouped_gemm.cu: kernel 实现与 host runtime
- tests/test_correctness.cu: 正确性测试
- bench/benchmark_grouped_gemm.cu: 基准测试
- docs/design.md: 设计与公式说明
- docs/grouped_gemm_summary.md: 代码级实现解读
- README_TESTING.md: 测试手册

## 3. 构建

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

说明：

1. 建议按你的 GPU 修改 CMAKE_CUDA_ARCHITECTURES。
2. 128x64x32 和 64x128x32 的 Tensor Core 变体需要较大动态 shared memory，代码已自动设置 kernel attribute。

## 4. 正确性测试

```bash
./build/test_grouped_gemm
```

测试会输出：

- Tensor Core 请求/实际启用状态
- occupancy 与寄存器压力信息
- 各排序模式与 baseline 的误差检查

成功标志：

```text
All correctness checks passed.
```

## 5. Benchmark

默认运行：

```bash
./build/bench_grouped_gemm
```

常用示例：

```bash
./build/bench_grouped_gemm \
  --workload mixed \
  --problems 512 \
  --warmup 20 \
  --iters 200 \
  --config 128x64x32 \
  --tensor-cores on
```

关闭 Tensor Core（用于对照）：

```bash
./build/bench_grouped_gemm --config 128x64x32 --tensor-cores off
```

固定 persistent grid：

```bash
./build/bench_grouped_gemm --persistent-ctas 128
```

固定输出缓存：

```bash
./build/bench_grouped_gemm \
  --workload mixed \
  --config 128x64x32 \
  --problems 512 \
  --warmup 10 \
  --iters 50 \
  --seed 42 \
  --persistent-ctas 0 \
  --tensor-cores on \
  --fixed-output bench_cache/mixed_128x64x32_p512_seed42.txt
```

## 6. 资源分析接口

query_occupancy 会返回：

- active_blocks_per_sm / theoretical_occupancy
- static_shared_bytes / dynamic_shared_bytes
- estimated_registers_per_thread（公式估算）
- registers_per_thread（cudaFuncGetAttributes 实测）
- registers_per_block / register_limited_blocks_per_sm
- using_tensor_cores（运行时实际模式）

## 7. 与任务书对应关系

1. 数据结构：ProblemDesc / PreparedProblemSet / DeviceProblemDesc
2. 调度策略：atomic 领取 + tile_offsets 前缀和 + device 二分定位
3. CTA / warp / mma 分块参数：64x64x16、128x64x32、64x128x32 + WMMA 16x16x16
4. 资源分析：shared memory 公式 + 寄存器压力公式 + 编译器实测
5. kernel 实现：SIMT 与 Tensor Core 双路径
6. 排序与负载均衡：4 种排序 + tail_share 指标
7. baseline 对比：run_baseline_per_problem

## 8. 备注

- 输入 A/B 为 FP16，累加与输出 D 为 FP32。
- Tensor Core 路径使用 WMMA，边界由预加载补零和写回边界判断处理。
- 若要进行严格性能对比，请固定 seed/warmup/iters，并多次重复取均值。
