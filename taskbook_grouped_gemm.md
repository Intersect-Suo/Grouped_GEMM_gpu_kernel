# 任务书：Grouped GEMM 设计与实现

## 一、任务名称

面向小规模、异构问题集合的 Grouped GEMM 设计与实现

## 二、任务背景

在解码阶段或稀疏/异构工作负载中，经常会同时出现许多彼此独立、但规模不同的小型 GEMM 问题。若采用“一个问题对应一次 kernel launch”的方式，会带来较高的 launch 开销，并且难以在 GPU 上实现良好的负载均衡。为此，需要实现一个 Grouped GEMM kernel：一次 kernel 启动处理一组 GEMM 问题，并在 GPU 侧持续调度不同问题的 tile 计算。

本任务的目标是实现一个高性能 Grouped GEMM 原型，重点调研并实现：

- 调度策略
- 分块策略
- 资源分配方式
- 面向异构 problem set 的负载均衡机制

实现中可参考 CUTLASS 的 grouped GEMM 与 grouped kernel scheduler 设计。

## 三、任务目标

实现一个能够处理多组、不同尺寸 GEMM 问题的 Grouped GEMM kernel，满足以下目标：

1. 一次 kernel launch 处理多个独立 GEMM 问题。
2. 不同问题可以有不同的 M、N、K 尺寸。
3. GPU 上的 threadblock 能持续从问题集合中领取新的 tile 进行计算。
4. 调度方式能够在异构问题规模下保持较好的负载均衡。
5. 实现中需要明确并验证：
   - tile 调度策略
   - CTA / warp / Tensor Core 的分块参数
   - occupancy 与资源占用
   - 问题排序或预处理对负载均衡的影响

## 四、任务范围

本任务只关注 Grouped GEMM 本身，包括：

1. 问题集合的数据结构表示；
2. grouped GEMM 的 host-side 参数准备；
3. kernel-side tile 调度器；
4. GEMM 的 CTA 级分块与流水；
5. 不同问题规模下的负载均衡；
6. profiling 与性能分析。

## 五、任务内容

### 1. 定义 problem set 数据结构

需要设计一组能够描述 grouped GEMM 输入的问题描述结构。至少包括：

- 每个问题的 M、N、K
- A / B / C / D 指针
- leading dimensions
- problem count
- problem 到 tile 数的映射信息

要求：

1. 支持每个问题尺寸不同；
2. 支持问题集合一次性传入 kernel；
3. 支持 kernel 在 device 侧根据问题编号读取对应参数。

### 2. 调研并选择调度策略

需要调研 CUTLASS grouped scheduler 的基本机制，并在实现中明确采用一种可运行的策略。

至少需要分析并实现以下内容：

1. threadblock 如何从 grouped problem set 中领取下一个 tile；
2. tile 与 threadblock 的映射关系；
3. 调度是完全 device-side 计算，还是部分 host-side 预处理；
4. 当问题尺寸不均匀时，如何减少负载不均衡；
5. 是否需要按 K 或 tile 数对问题排序。

要求：

- 给出调度策略说明；
- 给出对应实现；
- 用实验说明该策略在异构问题集合下确实有效。

### 3. 设计 CTA 级分块策略

需要针对目标硬件确定 grouped GEMM 使用的 CTA tile 形状与分块方式。

至少需要明确：

- CTA_M
- CTA_N
- CTA_K
- warp-level tile
- mma-level tile
- shared memory staging 级数

要求：

1. 分块策略与目标硬件匹配；
2. 能清楚解释为什么选取当前分块；
3. 在小问题和异构问题集合下仍能运行稳定。

### 4. 设计资源分配策略

需要分析 grouped GEMM 在一个 kernel 中的资源分配方式，至少包括：

- 每个 CTA 占用多少 shared memory
- 每个线程或 warp 的寄存器压力
- 每个 SM 可同时驻留多少 CTA
- persistent 风格 tile 调度下，grid 大小如何选择

要求：

1. 给出 occupancy 分析；
2. 说明 grid 大小与 problem 总 tile 数的关系；
3. 解释资源配置如何影响吞吐与负载均衡。

### 5. 实现 grouped GEMM kernel

需要实现一个 grouped GEMM kernel，支持：

1. 一次 launch 处理多个问题；
2. threadblock 在完成当前 tile 后继续领取新的 tile；
3. 使用 Tensor Core 或现代 GEMM 优化手段；
4. 使用 shared memory / pipeline 组织数据搬运与计算。

要求：

1. kernel 可以对多种问题尺寸正确计算；
2. kernel 能在问题集合为空时正确结束；
3. 计算结果与 reference GEMM 一致。

### 6. 分析问题排序与负载均衡

需要专门研究异构问题集合的负载均衡问题。

至少比较：

1. 不排序问题集合；
2. 按 K 维度排序；
3. 按 tile 数排序；
4. 其他你认为合理的排序或分桶方式。

要求：

- 给出实验比较不同策略的影响；
- 解释为什么某些排序方式更有效；
- 分析问题规模差异如何影响 grouped GEMM 调度效率。

### 7. 与 baseline 对比

至少选择一个 baseline 做对比：

1. 一个问题一次 launch 的朴素多次 GEMM 方案；
2. 或按尺寸分桶后调用传统 batched GEMM 的方案。

要求：

- 比较总运行时间；
- 比较 kernel launch 数量；
- 比较吞吐和负载均衡；
- 分析 grouped GEMM 的优势来源。

## 六、交付要求

### 1. 设计文档

内容包括：

- problem set 数据结构
- 调度策略说明
- CTA / warp / mma 分块参数
- 资源分配分析
- kernel 主循环伪代码
- 性能分析计划

### 2. 代码实现

内容包括：

- host-side problem set 准备代码
- grouped GEMM kernel
- scheduler / visitor 或等价调度实现
- correctness test
- benchmark 代码

### 3. 实验与报告

至少包含：

- 正确性验证
- 不同问题规模分布下的性能测试
- 不同调度策略比较
- 不同分块参数比较
- 问题排序策略比较
- 与 baseline 的对比结果

## 七、验收标准

本任务验收以“是否形成一个正确且具有可分析性能表现的 Grouped GEMM 实现”为准。

### 必须满足的功能性要求

1. 支持多问题一次性 grouped 执行；
2. 支持不同问题有不同 M、N、K；
3. 输出结果与 reference GEMM 一致；
4. kernel 能持续领取新 tile，直到整组问题完成；
5. 对空闲 tile、问题结束和队列耗尽等情况处理正确。

### 必须满足的性能与实现要求

1. 有明确的调度策略实现；
2. 有明确的 CTA 分块与资源配置分析；
3. 使用现代 GEMM 优化方式组织计算；
4. 在异构问题集合下能展示 grouped GEMM 相对 baseline 的优势或适用边界。

### 建议达到的效果

1. 在小规模、异构 problem set 下优于逐个 launch 的朴素方案；
2. 能通过排序或调度优化改善负载均衡；
3. 能清楚解释 grouped GEMM 的性能瓶颈来自哪里。
