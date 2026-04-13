# 详细测试 README

本文档给出 Tensor Core 版本 grouped GEMM 的完整验证流程：环境检查、构建、正确性、benchmark 与资源分析。

## 1. 环境检查

```bash
nvcc --version
cmake --version
nvidia-smi
```

预期：

1. nvcc 可用。
2. cmake >= 3.22。
3. GPU 驱动正常。

## 2. 全新构建

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

## 3. 正确性测试

```bash
./build/test_grouped_gemm
```

关键输出检查：

1. 每套配置都显示 tensor_cores_requested 与 tensor_cores_enabled。
2. 打印 occupancy、寄存器压力（estimated/measured）信息。
3. 四种 grouped mode 全部 PASS。
4. baseline PASS。
5. empty problem set PASS。
6. 最后出现 All correctness checks passed.

## 4. Benchmark 参数

先看帮助：

```bash
./build/bench_grouped_gemm --help
```

主要参数：

1. --problems N
2. --warmup N
3. --iters N
4. --seed N
5. --workload small|mixed|tail
6. --config 64x64x16|128x64x32|64x128x32
7. --persistent-ctas N
8. --tensor-cores on|off
9. --fixed-output PATH

## 5. 常用测试命令

默认（Tensor Core on）：

```bash
./build/bench_grouped_gemm --workload mixed --config 128x64x32 --problems 512 --warmup 10 --iters 50 --tensor-cores on
```

Tensor Core 对照（off）：

```bash
./build/bench_grouped_gemm --workload mixed --config 128x64x32 --problems 512 --warmup 10 --iters 50 --tensor-cores off
```

长尾场景：

```bash
./build/bench_grouped_gemm --workload tail --config 128x64x32 --problems 512 --warmup 10 --iters 50 --tensor-cores on
```

固定输出缓存：

```bash
./build/bench_grouped_gemm --workload mixed --config 128x64x32 --problems 512 --warmup 10 --iters 50 --seed 42 --persistent-ctas 0 --tensor-cores on --fixed-output bench_cache/mixed_128x64x32_p512_seed42.txt
```

## 6. 结果解读建议

启动阶段日志重点关注：

1. Runtime mode: tensor_cores_enabled
2. Occupancy: active_blocks_per_sm / theoretical
3. Register pressure: estimated_regs_per_thread / measured_regs_per_thread
4. Shared memory: static / dynamic

表格阶段重点关注：

1. kernel_ms
2. tflops
3. launches
4. tail_share

## 7. 常见问题

1. 128x64x32 Tensor Core 路径报 invalid argument：
当前代码已在 launch 前配置 cudaFuncAttributeMaxDynamicSharedMemorySize，请确认已重新编译。

2. 结果波动大：
提高 iters，固定 seed，并在系统空闲状态重复测试。

3. 架构不匹配：
修改 CMAKE_CUDA_ARCHITECTURES 后重新构建。
