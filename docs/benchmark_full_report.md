# Grouped GEMM 全面 Benchmark 报告


## 1. 目标与方法

本报告依据 README_TESTING.md 的 4. Benchmark 基线流程，执行了控制变量的广泛测试。

覆盖维度：workload/config、problems、persistent-ctas、seed、warmup/iters。

## 2. 测试矩阵

- 总运行数：48 runs。
- 每个 run 输出 5 行 mode 结果。
- mode 级总结果：240 行。

## 3. 场景级汇总（run 级）

| 场景 | runs | avg_speedup | min_speedup | max_speedup |
|---|---:|---:|---:|---:|
| grid_workload_config | 9 | 39.60 | 17.17 | 95.39 |
| sweep_persistent_mixed_128x64x32 | 8 | 26.71 | 17.51 | 29.44 |
| sweep_persistent_tail_128x64x32 | 8 | 17.80 | 13.03 | 19.94 |
| sweep_problems_mixed_64x64x16 | 5 | 27.84 | 15.74 | 34.56 |
| sweep_problems_tail_128x64x32 | 5 | 15.32 | 10.16 | 19.18 |
| sweep_seed_mixed_64x64x16 | 5 | 30.78 | 25.48 | 36.34 |
| sweep_seed_tail_128x64x32 | 5 | 18.29 | 14.58 | 20.65 |
| sweep_warmup_iters | 3 | 29.78 | 27.61 | 32.67 |

## 4. 全组合基线（workload x config）

固定参数：problems=256, warmup=10, iters=50, seed=42, persistent-ctas=0。

| workload | config | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---|---|---|---:|---:|---:|---:|---:|
| small | 64x64x16 | sort_by_tiles | 0.036 | 3.434 | 95.39 | 0.959 | 0.035 |
| small | 128x64x32 | sort_by_k | 0.067 | 3.502 | 52.27 | 0.514 | 0.061 |
| small | 64x128x32 | sort_by_tiles | 0.058 | 3.829 | 66.02 | 0.593 | 0.071 |
| mixed | 64x64x16 | sort_by_k | 0.104 | 3.303 | 31.76 | 2.741 | 0.012 |
| mixed | 128x64x32 | sort_by_k | 0.126 | 3.692 | 29.30 | 2.257 | 0.027 |
| mixed | 64x128x32 | sort_by_k | 0.142 | 3.943 | 27.77 | 2.002 | 0.028 |
| tail | 64x64x16 | sort_by_k | 0.186 | 3.193 | 17.17 | 4.531 | 0.003 |
| tail | 128x64x32 | sort_by_k | 0.198 | 3.755 | 18.96 | 4.276 | 0.011 |
| tail | 64x128x32 | sort_by_k | 0.210 | 3.721 | 17.72 | 4.031 | 0.011 |

---

**结论：**

- 最佳的排序方式大多是sorted_by_k，因为这样可以让计算量大的problem一开始就计算，不会留到最后再开始算而形成长尾

**说明：**

- 这里的grouped_ms采用的是命令行输出的kernal_ms，未加上prep_ms
- grouped_gemm结果比普通的gemm好（grouped_ms < baseline_ms），且都是64x64x16配置占优，因为目前选的(m,n,k)都较小（<200）
- 运行命令行，可以发现workload经过相关排序后，tail_share显著降低，意味着长尾效应降低明显
- 之所以经过排序后，workload为tail的problems相比于workload为small的tail_share少，是因为tail类型的问题集中problem相差甚远，所以有更多的大的problem都被排到前面了
- $speed\_up=\frac{baseline\_ms}{grouped\_ms}$
- 由于每次命令行运行时都是随机生成problems的，所以每一次的best_mode未必一致

## 5. Problem 大小（M/N/K）

固定参数：problems=64, warmup=5, iters=20, seed=42, persistent-ctas=0。

| problem_scale | M | N | K | config | grouped_ms | baseline_ms | speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| small | 100 | 80 | 60 | 64x64x16 | 0.046 | 0.754 | 16.39 |
| small | 100 | 80 | 60 | 128x64x32 | 0.051 | 1.022 | 20.04 |
| large | 2000 | 1000 | 800 | 64x64x16 | 49.094 | 47.888 | 0.98 |
| large | 2000 | 1000 | 800 | 128x64x32 | 44.933 | 42.337 | 0.94 |

**结论：**

- 如果problem小，那么grouped_gemm好于普通gemm; grouped_gemm中，64x64x16配置又好于128x64x32配置。这是因为64x64x16的active_blocks_per_sm要高于128x64x32（约为2倍）。这样64x64x16配置就可以完成更快的计算和更灵活的任务领取，减少block空转的可能，提高occupency。

- 如果problem大，那么普通gemm好于grouped_gemm; grouped_gemm中，128x64x32配置又好于64x64x16配置。这是因为128x64x32容许一次从Global Memory加载较大的Tile，因而可以在Block内部做更多的乘加运算，计算与访存比大，减少Block调度开销

- 综上，如果problem小，就选择grouped_gemmm，用64x64x16配置；如果problem大，就选择普通gemm；如果problem适中，就选择grouped_gemmm，用128x64x32/64x128x32配置


## 6. Problems 数量（规模扩展）

### 6.1 mixed + 64x64x16（warmup=10, iters=50, seed=42, persistent=0）

| problems | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---:|---|---:|---:|---:|---:|---:|
| 64 | none | 0.039 | 0.614 | 15.74 | 2.039 | 0.110 |
| 128 | sort_by_tiles | 0.052 | 1.495 | 28.75 | 2.414 | 0.012 |
| 256 | sort_by_k | 0.095 | 3.273 | 34.45 | 2.992 | 0.012 |
| 512 | none | 0.187 | 4.807 | 25.71 | 3.094 | 0.093 |
| 1024 | none | 0.383 | 13.236 | 34.56 | 2.986 | 0.095 |

### 6.2 tail + 128x64x32（warmup=10, iters=50, seed=42, persistent=0）

| problems | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---:|---|---:|---:|---:|---:|---:|
| 64 | sort_by_tiles_then_k | 0.080 | 0.813 | 10.16 | 2.687 | 0.010 |
| 128 | sort_by_k | 0.122 | 1.937 | 15.88 | 3.932 | 0.009 |
| 256 | sort_by_k | 0.200 | 3.836 | 19.18 | 4.231 | 0.011 |
| 512 | none | 0.452 | 7.642 | 16.91 | 3.507 | 0.101 |
| 1024 | none | 1.055 | 15.253 | 14.46 | 2.922 | 0.096 |

**结论：**

- 两份数据都表明，在problems少的情况下，提前sort problems会提升吞吐量，这是因为problems较少，SM之间会有严重的负载不均衡，进行排序可以缓解负载不均，减少长尾
- 而在problems多的情况下，SM一直在跑，负载不均衡没有那么明显，此时sort收益不大，反而会增加内存重排开销


## 7. Persistent-CTAs 扫描

### 7.1 mixed + 128x64x32 + problems=512（warmup=10, iters=50, seed=42）

| persistent-ctas | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---:|---|---:|---:|---:|---:|---:|
| 0 | sort_by_tiles | 0.274 | 7.142 | 26.07 | 2.106 | 0.018 |
| 24 | none | 0.459 | 8.038 | 17.51 | 1.258 | 0.094 |
| 48 | none | 0.319 | 8.077 | 25.32 | 1.811 | 0.094 |
| 72 | none | 0.276 | 8.061 | 29.21 | 2.090 | 0.094 |
| 96 | sort_by_tiles | 0.274 | 7.993 | 29.17 | 2.106 | 0.018 |
| 120 | none | 0.279 | 8.116 | 29.09 | 2.069 | 0.094 |
| 144 | none | 0.277 | 8.154 | 29.44 | 2.083 | 0.094 |
| 192 | none | 0.276 | 7.697 | 27.89 | 2.090 | 0.094 |

### 7.2 tail + 128x64x32 + problems=512（warmup=10, iters=50, seed=42）

| persistent-ctas | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---:|---|---:|---:|---:|---:|---:|
| 0 | none | 0.444 | 7.959 | 17.93 | 3.891 | 0.101 |
| 24 | none | 0.605 | 7.885 | 13.03 | 2.620 | 0.101 |
| 48 | none | 0.449 | 7.860 | 17.51 | 3.529 | 0.101 |
| 72 | none | 0.438 | 8.015 | 18.30 | 3.619 | 0.101 |
| 96 | none | 0.421 | 7.819 | 18.57 | 3.763 | 0.101 |
| 120 | none | 0.435 | 7.586 | 17.44 | 3.640 | 0.101 |
| 144 | sort_by_tiles | 0.396 | 7.897 | 19.94 | 4.005 | 0.011 |
| 192 | sort_by_k | 0.399 | 7.847 | 19.67 | 3.977 | 0.011 |


**结论：**

- persistent_ctas的设定和具体任务有关，对于不同的任务类型会有不同的最优选择

**说明：**

- persistent_ctas=0是默认的自动分配模式，采用的公式是：
$$max(1,min(max\_resident\_ctas, max(total\_tiles, sm\_count)))$$
- 这在最大可驻留cta数量和总tile数间进行了权衡。例如：如果命令行显示active_blocks_per_sm=3, sm_count=24,那么max_resident_ctas=3*24=72。与此同时，total_tiles=867，那么自动分配的persistent_ctas数为72
- persistent_ctas=0分配的未必是最优解，只是一个较为合理的值，可能存在更好的分配方式
- persistent_ctas=0自动推导出的72和手动设定persistent_ctas=72的结果有所差异，是因为problem集是随机的


## 8. Tensor core / SIMT

固定参数：config=64x64x16, problems=1024, warmup=5, iters=20, seed=42, persistent-ctas=0。

| m/n/k 取值集合 | tensor on grouped_ms | tensor off grouped_ms |
|---|---:|---:|
| {64, 96, 128, 256, 512} | 4.844 | 6.479 |
| {66, 98, 148, 300, 550} | 12.456 | 12.182 |
| {63, 94, 122, 240, 500} | 6.667 | 6.905 |

**结论：**

- 当m,n等于或小于且接近64的整数倍时，tensor on表现较好
- 当m,n略大于64的整数倍时，tensor off表现较好，这是因为配置是64x64x16，必定有一个block接近空载，此时tensor发挥不出计算复用的优势，却展现出额外数据搬运、数据对齐等开销

**说明：**

- 这里谈论的是problem小而多、负载不均的情况，即grouped_gemm优于普通gemm，所以不会考虑更大的64的整数倍

## 9. Seed 扫描（稳定性）

### 9.1 mixed + 64x64x16 + problems=256 + persistent=0

| seed | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | sort_by_k | 0.092 | 2.545 | 27.66 | 2.920 | 0.010 |
| 42 | sort_by_k | 0.095 | 3.083 | 32.45 | 2.986 | 0.012 |
| 20260328 | sort_by_k | 0.095 | 3.037 | 31.97 | 3.074 | 0.016 |
| 777 | sort_by_k | 0.096 | 3.489 | 36.34 | 3.115 | 0.015 |
| 20260331 | sort_by_k | 0.095 | 2.421 | 25.48 | 3.185 | 0.013 |

统计量：grouped mean=0.095 std=0.001 cv=1.43%; baseline mean=2.915 std=0.388 cv=13.32%

### 9.2 tail + 128x64x32 + problems=512 + persistent=0

| seed | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops | tail_share |
|---:|---|---:|---:|---:|---:|---:|
| 1 | none | 0.397 | 7.808 | 19.67 | 3.954 | 0.058 |
| 42 | none | 0.438 | 7.845 | 17.91 | 3.620 | 0.101 |
| 20260328 | sort_by_k | 0.369 | 7.620 | 20.65 | 3.982 | 0.012 |
| 777 | none | 0.394 | 5.746 | 14.58 | 3.710 | 0.108 |
| 20260331 | none | 0.414 | 7.708 | 18.62 | 3.920 | 0.058 |

统计量：grouped mean=0.402 std=0.023 cv=5.69%; baseline mean=7.345 std=0.804 cv=10.94%

**结论：**

- 切换不同的随机种子生成不同problem后，对于grouped_gemm而言，grouped_ms、best_mode和吞吐量都较为稳定
- 相比之下，baselin_ms的波动反映出当problem改变时，普通的gemm的性能变化大

## 10. Warmup / Iters 口径敏感性（mixed + 64x64x16 + problems=256 + seed=42 + persistent=0）

| warmup | iters | best_mode | grouped_ms | baseline_ms | speedup | grouped_tflops |
|---:|---:|---|---:|---:|---:|---:|
| 5 | 20 | sort_by_k | 0.095 | 2.760 | 29.05 | 2.998 |
| 10 | 50 | sort_by_k | 0.095 | 2.623 | 27.61 | 2.987 |
| 20 | 100 | sort_by_k | 0.095 | 3.104 | 32.67 | 2.986 |

**结论：**

- 同9

## 11. 补充结论

1. 48 组测试均实现 grouped 相对 baseline 的显著收益，speedup 区间为 10.16x 到 95.39x。
2. 全组合对比下：small 场景由 64x64x16 领先；mixed 场景 64x64x16 + sort_by_k 更稳；tail 场景 64x64x16 的绝对 grouped kernel 最短，但 speedup 在 128x64x32 上也具竞争力。
3. problems 扫描中，grouped 与 baseline 均随规模增长而变慢，但 baseline 增速更快，speedup 在 mixed/tail 上分别维持在 15.74-34.56x 与 10.16-19.18x。
4. persistent-ctas 扫描：mixed 场景 auto(0) 与 96 最优（~0.274ms）；tail 场景 144 最优（0.396ms），相比 auto(0) 的 0.444ms 提升约 10.8%。
5. seed 稳定性显示 grouped 波动显著小于 baseline，尤其在 mixed 场景更明显。
6. warmup/iters 变化时 grouped 最优 kernel 基本稳定在 0.095ms，而 baseline 波动更大。
7. best_mode 频次（run 级）：none=21, sort_by_k=20, sort_by_tiles=5, sort_by_tiles_then_k=2。排序策略并非在所有分布下恒定最优，应按 workload + 规模实测。

## 12. 参数建议

1. 通用默认：workload=mixed, config=64x64x16, persistent-ctas=0, warmup=10, iters=50。
2. tail 长尾优先尝试：config=128x64x32, persistent-ctas=144。
3. 报告口径建议固定 seed（如 42），并至少重复 3 次取均值。
4. 公平对比时，除目标变量外保持 problems/workload/config/seed/persistent 不变。
5. 若需要“同命令输出完全一致”，可使用 `--fixed-output <cache_file>`：首次生成缓存，后续复用缓存输出。

## 13. 全量结果附录

### 13.1 run 级全量结果（48 行）

```csv
run_id,scenario,workload,config,problems,warmup,iters,seed,persistent_ctas,best_grouped_mode,best_grouped_kernel_ms,baseline_kernel_ms,speedup_vs_baseline,best_grouped_tflops,baseline_tflops,best_grouped_tail_share,baseline_launches
1,grid_workload_config,small,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.036,3.434,95.39,0.959,0.010,0.035,256
2,grid_workload_config,small,128x64x32,256,10,50,42,0,sort_by_k,0.067,3.502,52.27,0.514,0.010,0.061,256
3,grid_workload_config,small,64x128x32,256,10,50,42,0,sort_by_tiles,0.058,3.829,66.02,0.593,0.009,0.071,256
4,grid_workload_config,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.104,3.303,31.76,2.741,0.086,0.012,256
5,grid_workload_config,mixed,128x64x32,256,10,50,42,0,sort_by_k,0.126,3.692,29.30,2.257,0.077,0.027,256
6,grid_workload_config,mixed,64x128x32,256,10,50,42,0,sort_by_k,0.142,3.943,27.77,2.002,0.072,0.028,256
7,grid_workload_config,tail,64x64x16,256,10,50,42,0,sort_by_k,0.186,3.193,17.17,4.531,0.265,0.003,256
8,grid_workload_config,tail,128x64x32,256,10,50,42,0,sort_by_k,0.198,3.755,18.96,4.276,0.225,0.011,256
9,grid_workload_config,tail,64x128x32,256,10,50,42,0,sort_by_k,0.210,3.721,17.72,4.031,0.227,0.011,256
10,sweep_problems_mixed_64x64x16,mixed,64x64x16,64,10,50,42,0,none,0.039,0.614,15.74,2.039,0.129,0.110,64
11,sweep_problems_mixed_64x64x16,mixed,64x64x16,128,10,50,42,0,sort_by_tiles,0.052,1.495,28.75,2.414,0.084,0.012,128
12,sweep_problems_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.095,3.273,34.45,2.992,0.087,0.012,256
13,sweep_problems_mixed_64x64x16,mixed,64x64x16,512,10,50,42,0,none,0.187,4.807,25.71,3.094,0.120,0.093,512
14,sweep_problems_mixed_64x64x16,mixed,64x64x16,1024,10,50,42,0,none,0.383,13.236,34.56,2.986,0.086,0.095,1024
15,sweep_problems_tail_128x64x32,tail,128x64x32,64,10,50,42,0,sort_by_tiles_then_k,0.080,0.813,10.16,2.687,0.263,0.010,64
16,sweep_problems_tail_128x64x32,tail,128x64x32,128,10,50,42,0,sort_by_k,0.122,1.937,15.88,3.932,0.248,0.009,128
17,sweep_problems_tail_128x64x32,tail,128x64x32,256,10,50,42,0,sort_by_k,0.200,3.836,19.18,4.231,0.220,0.011,256
18,sweep_problems_tail_128x64x32,tail,128x64x32,512,10,50,42,0,none,0.452,7.642,16.91,3.507,0.207,0.101,512
19,sweep_problems_tail_128x64x32,tail,128x64x32,1024,10,50,42,0,none,1.055,15.253,14.46,2.922,0.202,0.096,1024
20,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,0,sort_by_tiles,0.274,7.142,26.07,2.106,0.081,0.018,512
21,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,24,none,0.459,8.038,17.51,1.258,0.072,0.094,512
22,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,48,none,0.319,8.077,25.32,1.811,0.071,0.094,512
23,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,72,none,0.276,8.061,29.21,2.090,0.072,0.094,512
24,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,96,sort_by_tiles,0.274,7.993,29.17,2.106,0.072,0.018,512
25,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,120,none,0.279,8.116,29.09,2.069,0.071,0.094,512
26,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,144,none,0.277,8.154,29.44,2.083,0.071,0.094,512
27,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,192,none,0.276,7.697,27.89,2.090,0.075,0.094,512
28,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,0,none,0.444,7.959,17.93,3.574,0.199,0.101,512
29,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,24,none,0.605,7.885,13.03,2.620,0.201,0.101,512
30,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,48,none,0.449,7.860,17.51,3.529,0.202,0.101,512
31,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,72,none,0.438,8.015,18.30,3.619,0.198,0.101,512
32,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,96,none,0.421,7.819,18.57,3.763,0.203,0.101,512
33,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,120,none,0.435,7.586,17.44,3.640,0.209,0.101,512
34,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,144,sort_by_tiles,0.396,7.897,19.94,4.005,0.201,0.011,512
35,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,192,sort_by_k,0.399,7.847,19.67,3.977,0.202,0.011,512
36,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,1,0,sort_by_k,0.092,2.545,27.66,2.920,0.105,0.010,256
37,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.095,3.083,32.45,2.986,0.092,0.012,256
38,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260328,0,sort_by_k,0.095,3.037,31.97,3.074,0.096,0.016,256
39,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,777,0,sort_by_k,0.096,3.489,36.34,3.115,0.086,0.015,256
40,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260331,0,sort_by_k,0.095,2.421,25.48,3.185,0.125,0.013,256
41,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,1,0,none,0.397,7.808,19.67,3.954,0.201,0.058,512
42,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,42,0,none,0.438,7.845,17.91,3.620,0.202,0.101,512
43,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260328,0,sort_by_k,0.369,7.620,20.65,3.982,0.193,0.012,512
44,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,777,0,none,0.394,5.746,14.58,3.710,0.254,0.108,512
45,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260331,0,none,0.414,7.708,18.62,3.920,0.211,0.058,512
46,sweep_warmup_iters,mixed,64x64x16,256,5,20,42,0,sort_by_k,0.095,2.760,29.05,2.998,0.103,0.012,256
47,sweep_warmup_iters,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.095,2.623,27.61,2.987,0.108,0.012,256
48,sweep_warmup_iters,mixed,64x64x16,256,20,100,42,0,sort_by_k,0.095,3.104,32.67,2.986,0.092,0.012,256
```

### 13.2 mode 级全量结果（240 行）

```csv
run_id,scenario,workload,config,problems,warmup,iters,seed,persistent_ctas,mode,prep_ms,kernel_ms,tflops,tiles,grid_ctas,launches,tail_share
1,grid_workload_config,small,64x64x16,256,10,50,42,0,none,0.041,0.041,0.830,300,144,1,0.103
1,grid_workload_config,small,64x64x16,256,10,50,42,0,sort_by_k,0.056,0.039,0.876,300,144,1,0.035
1,grid_workload_config,small,64x64x16,256,10,50,42,0,sort_by_tiles,0.064,0.037,0.930,300,144,1,0.035
1,grid_workload_config,small,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.140,0.036,0.959,300,144,1,0.035
1,grid_workload_config,small,64x64x16,256,10,50,42,0,baseline_per_problem,0.000,3.434,0.010,300,0,256,0.000
2,grid_workload_config,small,128x64x32,256,10,50,42,0,none,0.040,0.070,0.490,300,72,1,0.100
2,grid_workload_config,small,128x64x32,256,10,50,42,0,sort_by_k,0.048,0.067,0.514,300,72,1,0.061
2,grid_workload_config,small,128x64x32,256,10,50,42,0,sort_by_tiles,0.047,0.068,0.509,300,72,1,0.061
2,grid_workload_config,small,128x64x32,256,10,50,42,0,sort_by_tiles_then_k,0.059,0.067,0.509,300,72,1,0.061
2,grid_workload_config,small,128x64x32,256,10,50,42,0,baseline_per_problem,0.000,3.502,0.010,300,0,256,0.000
3,grid_workload_config,small,64x128x32,256,10,50,42,0,none,0.053,0.062,0.558,256,72,1,0.105
3,grid_workload_config,small,64x128x32,256,10,50,42,0,sort_by_k,0.052,0.059,0.586,256,72,1,0.071
3,grid_workload_config,small,64x128x32,256,10,50,42,0,sort_by_tiles,0.065,0.058,0.593,256,72,1,0.071
3,grid_workload_config,small,64x128x32,256,10,50,42,0,sort_by_tiles_then_k,0.058,0.058,0.589,256,72,1,0.071
3,grid_workload_config,small,64x128x32,256,10,50,42,0,baseline_per_problem,0.000,3.829,0.009,256,0,256,0.000
4,grid_workload_config,mixed,64x64x16,256,10,50,42,0,none,0.048,0.111,2.561,620,144,1,0.123
4,grid_workload_config,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.084,0.104,2.741,620,144,1,0.012
4,grid_workload_config,mixed,64x64x16,256,10,50,42,0,sort_by_tiles,0.094,0.108,2.626,620,144,1,0.010
4,grid_workload_config,mixed,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.093,0.113,2.519,620,144,1,0.010
4,grid_workload_config,mixed,64x64x16,256,10,50,42,0,baseline_per_problem,0.000,3.303,0.086,620,0,256,0.000
5,grid_workload_config,mixed,128x64x32,256,10,50,42,0,none,0.064,0.140,2.033,457,72,1,0.117
5,grid_workload_config,mixed,128x64x32,256,10,50,42,0,sort_by_k,0.113,0.126,2.257,457,72,1,0.027
5,grid_workload_config,mixed,128x64x32,256,10,50,42,0,sort_by_tiles,0.052,0.136,2.094,457,72,1,0.018
5,grid_workload_config,mixed,128x64x32,256,10,50,42,0,sort_by_tiles_then_k,0.088,0.136,2.084,457,72,1,0.018
5,grid_workload_config,mixed,128x64x32,256,10,50,42,0,baseline_per_problem,0.000,3.692,0.077,457,0,256,0.000
6,grid_workload_config,mixed,64x128x32,256,10,50,42,0,none,0.059,0.152,1.877,456,72,1,0.120
6,grid_workload_config,mixed,64x128x32,256,10,50,42,0,sort_by_k,0.096,0.142,2.002,456,72,1,0.028
6,grid_workload_config,mixed,64x128x32,256,10,50,42,0,sort_by_tiles,0.110,0.144,1.977,456,72,1,0.018
6,grid_workload_config,mixed,64x128x32,256,10,50,42,0,sort_by_tiles_then_k,0.099,0.157,1.815,456,72,1,0.018
6,grid_workload_config,mixed,64x128x32,256,10,50,42,0,baseline_per_problem,0.000,3.943,0.072,456,0,256,0.000
7,grid_workload_config,tail,64x64x16,256,10,50,42,0,none,0.127,0.188,4.504,630,144,1,0.065
7,grid_workload_config,tail,64x64x16,256,10,50,42,0,sort_by_k,0.128,0.186,4.531,630,144,1,0.003
7,grid_workload_config,tail,64x64x16,256,10,50,42,0,sort_by_tiles,0.069,0.211,4.005,630,144,1,0.003
7,grid_workload_config,tail,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.062,0.209,4.051,630,144,1,0.003
7,grid_workload_config,tail,64x64x16,256,10,50,42,0,baseline_per_problem,0.000,3.193,0.265,630,0,256,0.000
8,grid_workload_config,tail,128x64x32,256,10,50,42,0,none,0.142,0.212,3.988,447,72,1,0.069
8,grid_workload_config,tail,128x64x32,256,10,50,42,0,sort_by_k,0.194,0.198,4.276,447,72,1,0.011
8,grid_workload_config,tail,128x64x32,256,10,50,42,0,sort_by_tiles,0.132,0.259,3.261,447,72,1,0.011
8,grid_workload_config,tail,128x64x32,256,10,50,42,0,sort_by_tiles_then_k,0.117,0.290,2.917,447,72,1,0.011
8,grid_workload_config,tail,128x64x32,256,10,50,42,0,baseline_per_problem,0.000,3.755,0.225,447,0,256,0.000
9,grid_workload_config,tail,64x128x32,256,10,50,42,0,none,0.052,0.223,3.791,444,72,1,0.064
9,grid_workload_config,tail,64x128x32,256,10,50,42,0,sort_by_k,0.060,0.210,4.031,444,72,1,0.011
9,grid_workload_config,tail,64x128x32,256,10,50,42,0,sort_by_tiles,0.062,0.228,3.700,444,72,1,0.011
9,grid_workload_config,tail,64x128x32,256,10,50,42,0,sort_by_tiles_then_k,0.076,0.308,2.744,444,72,1,0.011
9,grid_workload_config,tail,64x128x32,256,10,50,42,0,baseline_per_problem,0.000,3.721,0.227,444,0,256,0.000
10,sweep_problems_mixed_64x64x16,mixed,64x64x16,64,10,50,42,0,none,0.061,0.039,2.039,162,144,1,0.110
10,sweep_problems_mixed_64x64x16,mixed,64x64x16,64,10,50,42,0,sort_by_k,0.048,0.048,1.632,162,144,1,0.019
10,sweep_problems_mixed_64x64x16,mixed,64x64x16,64,10,50,42,0,sort_by_tiles,0.054,0.042,1.881,162,144,1,0.012
10,sweep_problems_mixed_64x64x16,mixed,64x64x16,64,10,50,42,0,sort_by_tiles_then_k,0.058,0.042,1.882,162,144,1,0.012
10,sweep_problems_mixed_64x64x16,mixed,64x64x16,64,10,50,42,0,baseline_per_problem,0.000,0.614,0.129,162,0,64,0.000
11,sweep_problems_mixed_64x64x16,mixed,64x64x16,128,10,50,42,0,none,0.058,0.056,2.235,300,144,1,0.128
11,sweep_problems_mixed_64x64x16,mixed,64x64x16,128,10,50,42,0,sort_by_k,0.055,0.056,2.238,300,144,1,0.016
11,sweep_problems_mixed_64x64x16,mixed,64x64x16,128,10,50,42,0,sort_by_tiles,0.081,0.052,2.414,300,144,1,0.012
11,sweep_problems_mixed_64x64x16,mixed,64x64x16,128,10,50,42,0,sort_by_tiles_then_k,0.055,0.053,2.388,300,144,1,0.012
11,sweep_problems_mixed_64x64x16,mixed,64x64x16,128,10,50,42,0,baseline_per_problem,0.000,1.495,0.084,300,0,128,0.000
12,sweep_problems_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,none,0.038,0.102,2.785,620,144,1,0.123
12,sweep_problems_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.086,0.095,2.992,620,144,1,0.012
12,sweep_problems_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_tiles,0.064,0.099,2.876,620,144,1,0.010
12,sweep_problems_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.055,0.100,2.849,620,144,1,0.010
12,sweep_problems_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,baseline_per_problem,0.000,3.273,0.087,620,0,256,0.000
13,sweep_problems_mixed_64x64x16,mixed,64x64x16,512,10,50,42,0,none,0.058,0.187,3.094,1220,144,1,0.093
13,sweep_problems_mixed_64x64x16,mixed,64x64x16,512,10,50,42,0,sort_by_k,0.187,0.210,2.743,1220,144,1,0.016
13,sweep_problems_mixed_64x64x16,mixed,64x64x16,512,10,50,42,0,sort_by_tiles,0.092,0.279,2.071,1220,144,1,0.012
13,sweep_problems_mixed_64x64x16,mixed,64x64x16,512,10,50,42,0,sort_by_tiles_then_k,0.160,0.638,0.905,1220,144,1,0.012
13,sweep_problems_mixed_64x64x16,mixed,64x64x16,512,10,50,42,0,baseline_per_problem,0.000,4.807,0.120,1220,0,512,0.000
14,sweep_problems_mixed_64x64x16,mixed,64x64x16,1024,10,50,42,0,none,0.115,0.383,2.986,2435,144,1,0.095
14,sweep_problems_mixed_64x64x16,mixed,64x64x16,1024,10,50,42,0,sort_by_k,0.212,0.398,2.873,2435,144,1,0.017
14,sweep_problems_mixed_64x64x16,mixed,64x64x16,1024,10,50,42,0,sort_by_tiles,0.231,0.395,2.897,2435,144,1,0.012
14,sweep_problems_mixed_64x64x16,mixed,64x64x16,1024,10,50,42,0,sort_by_tiles_then_k,0.257,0.811,1.410,2435,144,1,0.012
14,sweep_problems_mixed_64x64x16,mixed,64x64x16,1024,10,50,42,0,baseline_per_problem,0.000,13.236,0.086,2435,0,1024,0.000
15,sweep_problems_tail_128x64x32,tail,128x64x32,64,10,50,42,0,none,0.052,0.086,2.492,107,72,1,0.180
15,sweep_problems_tail_128x64x32,tail,128x64x32,64,10,50,42,0,sort_by_k,0.040,0.096,2.242,107,72,1,0.010
15,sweep_problems_tail_128x64x32,tail,128x64x32,64,10,50,42,0,sort_by_tiles,0.037,0.081,2.657,107,72,1,0.010
15,sweep_problems_tail_128x64x32,tail,128x64x32,64,10,50,42,0,sort_by_tiles_then_k,0.030,0.080,2.687,107,72,1,0.010
15,sweep_problems_tail_128x64x32,tail,128x64x32,64,10,50,42,0,baseline_per_problem,0.000,0.813,0.263,107,0,64,0.000
16,sweep_problems_tail_128x64x32,tail,128x64x32,128,10,50,42,0,none,0.034,0.145,3.310,223,72,1,0.130
16,sweep_problems_tail_128x64x32,tail,128x64x32,128,10,50,42,0,sort_by_k,0.047,0.122,3.932,223,72,1,0.009
16,sweep_problems_tail_128x64x32,tail,128x64x32,128,10,50,42,0,sort_by_tiles,0.037,0.130,3.697,223,72,1,0.009
16,sweep_problems_tail_128x64x32,tail,128x64x32,128,10,50,42,0,sort_by_tiles_then_k,0.051,0.163,2.949,223,72,1,0.009
16,sweep_problems_tail_128x64x32,tail,128x64x32,128,10,50,42,0,baseline_per_problem,0.000,1.937,0.248,223,0,128,0.000
17,sweep_problems_tail_128x64x32,tail,128x64x32,256,10,50,42,0,none,0.051,0.212,3.984,447,72,1,0.069
17,sweep_problems_tail_128x64x32,tail,128x64x32,256,10,50,42,0,sort_by_k,0.073,0.200,4.231,447,72,1,0.011
17,sweep_problems_tail_128x64x32,tail,128x64x32,256,10,50,42,0,sort_by_tiles,0.139,0.269,3.139,447,72,1,0.011
17,sweep_problems_tail_128x64x32,tail,128x64x32,256,10,50,42,0,sort_by_tiles_then_k,0.069,0.268,3.151,447,72,1,0.011
17,sweep_problems_tail_128x64x32,tail,128x64x32,256,10,50,42,0,baseline_per_problem,0.000,3.836,0.220,447,0,256,0.000
18,sweep_problems_tail_128x64x32,tail,128x64x32,512,10,50,42,0,none,0.047,0.452,3.507,867,72,1,0.101
18,sweep_problems_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_k,0.088,0.702,2.257,867,72,1,0.011
18,sweep_problems_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_tiles,0.181,0.741,2.139,867,72,1,0.011
18,sweep_problems_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_tiles_then_k,0.109,0.731,2.168,867,72,1,0.011
18,sweep_problems_tail_128x64x32,tail,128x64x32,512,10,50,42,0,baseline_per_problem,0.000,7.642,0.207,867,0,512,0.000
19,sweep_problems_tail_128x64x32,tail,128x64x32,1024,10,50,42,0,none,0.186,1.055,2.922,1741,72,1,0.096
19,sweep_problems_tail_128x64x32,tail,128x64x32,1024,10,50,42,0,sort_by_k,0.268,1.319,2.337,1741,72,1,0.011
19,sweep_problems_tail_128x64x32,tail,128x64x32,1024,10,50,42,0,sort_by_tiles,0.271,1.363,2.262,1741,72,1,0.011
19,sweep_problems_tail_128x64x32,tail,128x64x32,1024,10,50,42,0,sort_by_tiles_then_k,0.281,1.442,2.138,1741,72,1,0.011
19,sweep_problems_tail_128x64x32,tail,128x64x32,1024,10,50,42,0,baseline_per_problem,0.000,15.253,0.202,1741,0,1024,0.000
20,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,0,none,0.180,0.283,2.041,915,72,1,0.094
20,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,0,sort_by_k,0.158,0.278,2.076,915,72,1,0.031
20,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,0,sort_by_tiles,0.142,0.274,2.106,915,72,1,0.018
20,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,0,sort_by_tiles_then_k,0.081,0.299,1.928,915,72,1,0.018
20,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,0,baseline_per_problem,0.000,7.142,0.081,915,0,512,0.000
21,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,24,none,0.049,0.459,1.258,915,24,1,0.094
21,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,24,sort_by_k,0.136,0.574,1.005,915,24,1,0.031
21,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,24,sort_by_tiles,0.122,0.616,0.937,915,24,1,0.018
21,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,24,sort_by_tiles_then_k,0.124,0.612,0.943,915,24,1,0.018
21,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,24,baseline_per_problem,0.000,8.038,0.072,915,0,512,0.000
22,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,48,none,0.094,0.319,1.811,915,48,1,0.094
22,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,48,sort_by_k,0.143,0.348,1.660,915,48,1,0.031
22,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,48,sort_by_tiles,0.101,0.483,1.195,915,48,1,0.018
22,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,48,sort_by_tiles_then_k,0.112,0.479,1.205,915,48,1,0.018
22,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,48,baseline_per_problem,0.000,8.077,0.071,915,0,512,0.000
23,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,72,none,0.060,0.276,2.090,915,72,1,0.094
23,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,72,sort_by_k,0.138,0.351,1.646,915,72,1,0.031
23,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,72,sort_by_tiles,0.165,0.548,1.053,915,72,1,0.018
23,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,72,sort_by_tiles_then_k,0.362,0.548,1.053,915,72,1,0.018
23,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,72,baseline_per_problem,0.000,8.061,0.072,915,0,512,0.000
24,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,96,none,0.086,0.283,2.039,915,96,1,0.094
24,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,96,sort_by_k,0.117,0.278,2.077,915,96,1,0.031
24,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,96,sort_by_tiles,0.096,0.274,2.106,915,96,1,0.018
24,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,96,sort_by_tiles_then_k,0.137,0.340,1.698,915,96,1,0.018
24,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,96,baseline_per_problem,0.000,7.993,0.072,915,0,512,0.000
25,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,120,none,0.063,0.279,2.069,915,120,1,0.094
25,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,120,sort_by_k,0.105,0.312,1.848,915,120,1,0.031
25,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,120,sort_by_tiles,0.119,0.447,1.292,915,120,1,0.018
25,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,120,sort_by_tiles_then_k,0.139,0.452,1.275,915,120,1,0.018
25,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,120,baseline_per_problem,0.000,8.116,0.071,915,0,512,0.000
26,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,144,none,0.061,0.277,2.083,915,144,1,0.094
26,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,144,sort_by_k,0.132,0.291,1.981,915,144,1,0.031
26,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,144,sort_by_tiles,0.144,0.542,1.064,915,144,1,0.018
26,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,144,sort_by_tiles_then_k,0.133,0.506,1.139,915,144,1,0.018
26,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,144,baseline_per_problem,0.000,8.154,0.071,915,0,512,0.000
27,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,192,none,0.049,0.276,2.090,915,192,1,0.094
27,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,192,sort_by_k,0.089,0.299,1.928,915,192,1,0.031
27,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,192,sort_by_tiles,0.095,0.298,1.937,915,192,1,0.018
27,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,192,sort_by_tiles_then_k,0.094,0.299,1.928,915,192,1,0.018
27,sweep_persistent_mixed_128x64x32,mixed,128x64x32,512,10,50,42,192,baseline_per_problem,0.000,7.697,0.075,915,0,512,0.000
28,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,0,none,0.170,0.444,3.574,867,72,1,0.101
28,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_k,0.144,0.594,2.670,867,72,1,0.011
28,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_tiles,0.126,0.672,2.359,867,72,1,0.011
28,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_tiles_then_k,0.097,0.671,2.361,867,72,1,0.011
28,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,0,baseline_per_problem,0.000,7.959,0.199,867,0,512,0.000
29,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,24,none,0.050,0.605,2.620,867,24,1,0.101
29,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,24,sort_by_k,0.125,0.882,1.796,867,24,1,0.011
29,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,24,sort_by_tiles,0.116,0.859,1.845,867,24,1,0.011
29,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,24,sort_by_tiles_then_k,0.108,0.859,1.846,867,24,1,0.011
29,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,24,baseline_per_problem,0.000,7.885,0.201,867,0,512,0.000
30,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,48,none,0.064,0.449,3.529,867,48,1,0.101
30,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,48,sort_by_k,0.118,0.837,1.893,867,48,1,0.011
30,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,48,sort_by_tiles,0.164,0.776,2.042,867,48,1,0.011
30,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,48,sort_by_tiles_then_k,0.124,0.759,2.088,867,48,1,0.011
30,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,48,baseline_per_problem,0.000,7.860,0.202,867,0,512,0.000
31,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,72,none,0.059,0.438,3.619,867,72,1,0.101
31,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,72,sort_by_k,0.118,0.612,2.590,867,72,1,0.011
31,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,72,sort_by_tiles,0.122,0.682,2.326,867,72,1,0.011
31,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,72,sort_by_tiles_then_k,0.099,0.668,2.372,867,72,1,0.011
31,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,72,baseline_per_problem,0.000,8.015,0.198,867,0,512,0.000
32,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,96,none,0.111,0.421,3.763,867,96,1,0.101
32,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,96,sort_by_k,0.159,0.683,2.321,867,96,1,0.011
32,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,96,sort_by_tiles,0.101,0.680,2.333,867,96,1,0.011
32,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,96,sort_by_tiles_then_k,0.097,0.665,2.383,867,96,1,0.011
32,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,96,baseline_per_problem,0.000,7.819,0.203,867,0,512,0.000
33,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,120,none,0.050,0.435,3.640,867,120,1,0.101
33,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,120,sort_by_k,0.093,0.711,2.230,867,120,1,0.011
33,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,120,sort_by_tiles,0.133,0.767,2.067,867,120,1,0.011
33,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,120,sort_by_tiles_then_k,0.109,0.735,2.156,867,120,1,0.011
33,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,120,baseline_per_problem,0.000,7.586,0.209,867,0,512,0.000
34,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,144,none,0.063,0.420,3.776,867,144,1,0.101
34,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,144,sort_by_k,0.179,0.401,3.952,867,144,1,0.011
34,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,144,sort_by_tiles,0.095,0.396,4.005,867,144,1,0.011
34,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,144,sort_by_tiles_then_k,0.318,0.710,2.233,867,144,1,0.011
34,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,144,baseline_per_problem,0.000,7.897,0.201,867,0,512,0.000
35,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,192,none,0.060,0.420,3.773,867,192,1,0.101
35,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,192,sort_by_k,0.127,0.399,3.977,867,192,1,0.011
35,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,192,sort_by_tiles,0.182,0.598,2.651,867,192,1,0.011
35,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,192,sort_by_tiles_then_k,0.242,0.792,2.001,867,192,1,0.011
35,sweep_persistent_tail_128x64x32,tail,128x64x32,512,10,50,42,192,baseline_per_problem,0.000,7.847,0.202,867,0,512,0.000
36,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,1,0,none,0.038,0.097,2.768,617,144,1,0.111
36,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,1,0,sort_by_k,0.058,0.092,2.920,617,144,1,0.010
36,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,1,0,sort_by_tiles,0.053,0.096,2.783,617,144,1,0.010
36,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,1,0,sort_by_tiles_then_k,0.058,0.096,2.776,617,144,1,0.010
36,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,1,0,baseline_per_problem,0.000,2.545,0.105,617,0,256,0.000
37,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,none,0.076,0.103,2.771,620,144,1,0.123
37,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.051,0.095,2.986,620,144,1,0.012
37,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_tiles,0.078,0.099,2.868,620,144,1,0.010
37,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.063,0.100,2.852,620,144,1,0.010
37,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,42,0,baseline_per_problem,0.000,3.083,0.092,620,0,256,0.000
38,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260328,0,none,0.067,0.099,2.925,612,144,1,0.083
38,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260328,0,sort_by_k,0.084,0.095,3.074,612,144,1,0.016
38,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260328,0,sort_by_tiles,0.084,0.100,2.904,612,144,1,0.011
38,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260328,0,sort_by_tiles_then_k,0.082,0.099,2.933,612,144,1,0.011
38,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260328,0,baseline_per_problem,0.000,3.037,0.096,612,0,256,0.000
39,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,777,0,none,0.043,0.101,2.965,593,144,1,0.078
39,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,777,0,sort_by_k,0.053,0.096,3.115,593,144,1,0.015
39,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,777,0,sort_by_tiles,0.053,0.103,2.903,593,144,1,0.011
39,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,777,0,sort_by_tiles_then_k,0.104,0.102,2.927,593,144,1,0.011
39,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,777,0,baseline_per_problem,0.000,3.489,0.086,593,0,256,0.000
40,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260331,0,none,0.053,0.103,2.963,582,144,1,0.072
40,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260331,0,sort_by_k,0.053,0.095,3.185,582,144,1,0.013
40,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260331,0,sort_by_tiles,0.056,0.100,3.025,582,144,1,0.010
40,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260331,0,sort_by_tiles_then_k,0.087,0.100,3.032,582,144,1,0.010
40,sweep_seed_mixed_64x64x16,mixed,64x64x16,256,10,50,20260331,0,baseline_per_problem,0.000,2.421,0.125,582,0,256,0.000
41,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,1,0,none,0.062,0.397,3.954,876,72,1,0.058
41,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,1,0,sort_by_k,0.147,0.628,2.500,876,72,1,0.011
41,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,1,0,sort_by_tiles,0.274,0.639,2.456,876,72,1,0.011
41,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,1,0,sort_by_tiles_then_k,0.157,0.665,2.361,876,72,1,0.011
41,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,1,0,baseline_per_problem,0.000,7.808,0.201,876,0,512,0.000
42,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,42,0,none,0.132,0.438,3.620,867,72,1,0.101
42,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_k,0.125,0.641,2.474,867,72,1,0.011
42,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_tiles,0.124,0.764,2.074,867,72,1,0.011
42,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,42,0,sort_by_tiles_then_k,0.114,0.739,2.146,867,72,1,0.011
42,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,42,0,baseline_per_problem,0.000,7.845,0.202,867,0,512,0.000
43,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260328,0,none,0.064,0.383,3.829,851,72,1,0.084
43,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260328,0,sort_by_k,0.112,0.369,3.982,851,72,1,0.012
43,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260328,0,sort_by_tiles,0.121,0.399,3.679,851,72,1,0.012
43,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260328,0,sort_by_tiles_then_k,0.103,0.658,2.231,851,72,1,0.012
43,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260328,0,baseline_per_problem,0.000,7.620,0.193,851,0,512,0.000
44,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,777,0,none,0.395,0.394,3.710,841,72,1,0.108
44,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,777,0,sort_by_k,2.236,0.743,1.967,841,72,1,0.012
44,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,777,0,sort_by_tiles,0.397,0.700,2.090,841,72,1,0.012
44,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,777,0,sort_by_tiles_then_k,0.446,0.691,2.117,841,72,1,0.012
44,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,777,0,baseline_per_problem,0.000,5.746,0.254,841,0,512,0.000
45,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260331,0,none,0.092,0.414,3.920,883,72,1,0.058
45,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260331,0,sort_by_k,0.128,0.776,2.093,883,72,1,0.011
45,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260331,0,sort_by_tiles,0.101,0.781,2.078,883,72,1,0.011
45,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260331,0,sort_by_tiles_then_k,0.127,0.744,2.183,883,72,1,0.011
45,sweep_seed_tail_128x64x32,tail,128x64x32,512,10,50,20260331,0,baseline_per_problem,0.000,7.708,0.211,883,0,512,0.000
46,sweep_warmup_iters,mixed,64x64x16,256,5,20,42,0,none,0.033,0.103,2.766,620,144,1,0.123
46,sweep_warmup_iters,mixed,64x64x16,256,5,20,42,0,sort_by_k,0.092,0.095,2.998,620,144,1,0.012
46,sweep_warmup_iters,mixed,64x64x16,256,5,20,42,0,sort_by_tiles,0.047,0.100,2.831,620,144,1,0.010
46,sweep_warmup_iters,mixed,64x64x16,256,5,20,42,0,sort_by_tiles_then_k,0.042,0.099,2.880,620,144,1,0.010
46,sweep_warmup_iters,mixed,64x64x16,256,5,20,42,0,baseline_per_problem,0.000,2.760,0.103,620,0,256,0.000
47,sweep_warmup_iters,mixed,64x64x16,256,10,50,42,0,none,0.092,0.102,2.797,620,144,1,0.123
47,sweep_warmup_iters,mixed,64x64x16,256,10,50,42,0,sort_by_k,0.084,0.095,2.987,620,144,1,0.012
47,sweep_warmup_iters,mixed,64x64x16,256,10,50,42,0,sort_by_tiles,0.082,0.099,2.870,620,144,1,0.010
47,sweep_warmup_iters,mixed,64x64x16,256,10,50,42,0,sort_by_tiles_then_k,0.068,0.100,2.851,620,144,1,0.010
47,sweep_warmup_iters,mixed,64x64x16,256,10,50,42,0,baseline_per_problem,0.000,2.623,0.108,620,0,256,0.000
48,sweep_warmup_iters,mixed,64x64x16,256,20,100,42,0,none,0.039,0.102,2.784,620,144,1,0.123
48,sweep_warmup_iters,mixed,64x64x16,256,20,100,42,0,sort_by_k,0.061,0.095,2.986,620,144,1,0.012
48,sweep_warmup_iters,mixed,64x64x16,256,20,100,42,0,sort_by_tiles,0.056,0.113,2.528,620,144,1,0.010
48,sweep_warmup_iters,mixed,64x64x16,256,20,100,42,0,sort_by_tiles_then_k,0.091,0.145,1.961,620,144,1,0.010
48,sweep_warmup_iters,mixed,64x64x16,256,20,100,42,0,baseline_per_problem,0.000,3.104,0.092,620,0,256,0.000
```

### 13.3 原始 stdout 汇总

- 原始逐条命令与完整输出：bench/results_20260331/raw_runs.log
- 进度记录：bench/results_20260331/progress.log
