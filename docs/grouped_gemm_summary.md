# Grouped GEMM 实现概述

本文聚焦代码当前状态：已接入 Tensor Core（WMMA）路径，并补充寄存器压力代码化分析。

## 1. 对外接口

头文件：include/grouped_gemm/grouped_gemm.cuh

1. ProblemDesc: 单问题描述。
2. KernelConfig:
   - cta_m, cta_n, cta_k, threads
   - persistent_ctas
   - use_tensor_cores
3. PreparedProblemSet: 排序结果与 tile_offsets。
4. OccupancyInfo:
   - occupancy 相关字段
   - shared memory 字段
   - 寄存器估算值与实测值
5. API:
   - prepare_problem_set
   - query_occupancy
   - run_grouped_gemm_once
   - run_baseline_per_problem

## 2. 核心调度

实现文件：src/grouped_gemm.cu

1. grouped_gemm_kernel 使用 persistent 任务池：
   - block 内 thread0: atomicAdd(global_counter, 1)
   - tile_offsets 二分定位 problem
   - local_tile -> tile_m/tile_n
2. CTA 完成一个 tile 后继续领取，直到 total_tiles 耗尽。

## 3. 计算路径

### 3.1 SIMT 路径

- compute_gemm_tile_simt
- cp.async + 2-stage 双缓冲
- shared memory tiled GEMM
- FP16 输入，FP32 累加与输出

### 3.2 Tensor Core 路径

- compute_gemm_tile_tensor_core
- WMMA shape: 16x16x16
- CTA 内 MMA tile 分配给 8 个 warp 做 stride 处理
- 计算结果先写 shared_c，再统一做 epilogue:
  - D = alpha * AB + beta * C

### 3.3 路径选择

- KernelConfig.use_tensor_cores 为请求开关。
- 运行时根据设备能力与配置检查决定 occupancy.using_tensor_cores。
- benchmark 支持 --tensor-cores on/off 做对照。

## 4. 资源分析（代码化）

query_occupancy 会返回以下两类寄存器压力信息：

1. 公式估算：estimated_registers_per_thread / estimated_registers_per_block
2. 编译器实测：registers_per_thread / registers_per_block（cudaFuncGetAttributes）

并返回：

- register_limited_blocks_per_sm
- active_blocks_per_sm
- theoretical_occupancy
- static_shared_bytes / dynamic_shared_bytes

## 5. 大 shared memory 处理

Tensor Core 的 128x64x32 与 64x128x32 变体动态 shared memory 可超过 48KB。

当前实现在 launch/query 前自动调用：

1. cudaFuncAttributeMaxDynamicSharedMemorySize
2. cudaFuncAttributePreferredSharedMemoryCarveout

避免 invalid argument 与 occupancy 误报。

## 6. 测试与基准

1. tests/test_correctness.cu:
   - 两套配置回归
   - grouped 各排序模式 + baseline + 空问题集
   - 输出 tensor core 实际启用与寄存器压力
2. bench/benchmark_grouped_gemm.cu:
   - 参数新增 --tensor-cores on|off
   - 启动日志输出 occupancy/寄存器/shared memory 指标

## 7. 其他

1. sort_by_tiles_then_k 与 sort_by_tiles 共享比较器。

