#include "grouped_gemm/grouped_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

// 统一封装 CUDA 调用检查。
// 基准测试一旦出现 CUDA 错误，结果就不再可信，因此这里直接报错并退出。
#define CHECK_CUDA_OR_EXIT(call)                                                       \
  do {                                                                                 \
    cudaError_t _status = (call);                                                     \
    if (_status != cudaSuccess) {                                                     \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                << cudaGetErrorString(_status) << std::endl;                          \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

// 向上取整除法：返回 ceil(x / y)。
// 主要用于计算 tile 数量和 k 维步数。
int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

// 计算矩阵元素总数，并保证最少返回 1。
// 这样在 rows 或 cols 为 0 的退化场景下，仍可保持分配/拷贝路径一致。
size_t safe_matrix_elems(int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    return 1;
  }
  return static_cast<size_t>(rows) * cols;
}

// 基准命令行参数与 kernel 配置集合。
struct BenchOptions {
  int problem_count = 256;
  int warmup = 20;
  int iters = 100;
  int seed = 20260328;
  std::string workload = "mixed";
  std::string fixed_output_path;
  grouped_gemm::KernelConfig config{};
};

// 单个 GEMM 问题的主机/设备缓冲区。
// 通过析构自动释放显存，避免长时间压测泄漏。
struct ProblemBuffer {
  int m = 0;
  int n = 0;
  int k = 0;

  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int ldd = 0;

  std::vector<__half> h_a;
  std::vector<__half> h_b;
  std::vector<float> h_c;

  __half* d_a = nullptr;
  __half* d_b = nullptr;
  float* d_c = nullptr;
  float* d_d = nullptr;

  // RAII 释放设备内存。
  ~ProblemBuffer() {
    if (d_a != nullptr) {
      cudaFree(d_a);
    }
    if (d_b != nullptr) {
      cudaFree(d_b);
    }
    if (d_c != nullptr) {
      cudaFree(d_c);
    }
    if (d_d != nullptr) {
      cudaFree(d_d);
    }
  }
};

// 每种执行模式的输出指标。
// 包括准备时间、kernel 时间、吞吐、tile/网格信息和尾部工作占比。
struct ModeResult {
  std::string mode;
  float prepare_ms = 0.0f;
  float kernel_ms = 0.0f;
  double tflops = 0.0;
  int total_tiles = 0;
  int grid_ctas = 0;
  int launch_count = 0;
  double tail_work_share = 0.0;
};

// 打印命令行帮助信息。
void print_usage(const char* prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "  --problems N            number of GEMM problems (default: 256)\n"
      << "  --warmup N              warmup iterations (default: 20)\n"
      << "  --iters N               benchmark iterations (default: 100)\n"
      << "  --seed N                random seed (default: 20260328)\n"
      << "  --workload NAME         small|mixed|tail (default: mixed)\n"
      << "  --config NAME           64x64x16|128x64x32|64x128x32 (default: 64x64x16)\n"
      << "  --persistent-ctas N     fixed persistent grid CTAs (default: auto)\n"
        << "  --tensor-cores MODE     on|off (default: on)\n"
      << "  --fixed-output PATH     reuse cached benchmark table for stable output\n";
}

// 解析命令行参数，并做基础合法化处理。
// 例如保证 problem_count>=1、warmup>=0、iters>=1。
bool parse_options(int argc, char** argv, BenchOptions* opt) {
  if (opt == nullptr) {
    return false;
  }

  // 先设默认核配置，可被 --config 覆盖。
  opt->config.cta_m = 64;
  opt->config.cta_n = 64;
  opt->config.cta_k = 16;
  opt->config.threads = 256;
  opt->config.persistent_ctas = 0;
  opt->config.use_tensor_cores = true;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    // 检查当前参数后是否有值，防止越界访问 argv。
    auto need_value = [&](const std::string& name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        return false;
      }
      return true;
    };

    if (arg == "--help" || arg == "-h") {
      return false;
    } else if (arg == "--problems") {
      if (!need_value(arg)) {
        return false;
      }
      opt->problem_count = std::stoi(argv[++i]);
    } else if (arg == "--warmup") {
      if (!need_value(arg)) {
        return false;
      }
      opt->warmup = std::stoi(argv[++i]);
    } else if (arg == "--iters") {
      if (!need_value(arg)) {
        return false;
      }
      opt->iters = std::stoi(argv[++i]);
    } else if (arg == "--seed") {
      if (!need_value(arg)) {
        return false;
      }
      opt->seed = std::stoi(argv[++i]);
    } else if (arg == "--workload") {
      if (!need_value(arg)) {
        return false;
      }
      opt->workload = argv[++i];
    } else if (arg == "--persistent-ctas") {
        if (!need_value(arg)) {
          return false;
        }
        opt->config.persistent_ctas = std::stoi(argv[++i]);
      } else if (arg == "--tensor-cores") {
        if (!need_value(arg)) {
          return false;
        }
        const std::string mode = argv[++i];
        if (mode == "on") {
          opt->config.use_tensor_cores = true;
        } else if (mode == "off") {
          opt->config.use_tensor_cores = false;
        } else {
          std::cerr << "Unsupported --tensor-cores mode: " << mode << std::endl;
          return false;
        }
      } else if (arg == "--fixed-output") {
      if (!need_value(arg)) {
        return false;
      }
      opt->fixed_output_path = argv[++i];
    } else if (arg == "--config") {
      if (!need_value(arg)) {
        return false;
      }
      const std::string cfg = argv[++i];
      if (cfg == "64x64x16") {
        opt->config.cta_m = 64;
        opt->config.cta_n = 64;
        opt->config.cta_k = 16;
      } else if (cfg == "128x64x32") {
        opt->config.cta_m = 128;
        opt->config.cta_n = 64;
        opt->config.cta_k = 32;
      } else if (cfg == "64x128x32") {
        opt->config.cta_m = 64;
        opt->config.cta_n = 128;
        opt->config.cta_k = 32;
      } else {
        std::cerr << "Unsupported config: " << cfg << std::endl;
        return false;
      }
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      return false;
    }
  }

  // 做下界钳制，保证迭代参数有效。
  opt->problem_count = std::max(1, opt->problem_count);
  opt->warmup = std::max(0, opt->warmup);
  opt->iters = std::max(1, opt->iters);
  return true;
}

// 生成当前实验配置指纹，用于判断 fixed output 缓存是否可复用。
std::string make_fixed_output_fingerprint(
    const BenchOptions& opt,
    const grouped_gemm::OccupancyInfo& occupancy) {
  std::ostringstream oss;
  oss << "v2"
        << ";workload=" << opt.workload
        << ";problems=" << opt.problem_count
        << ";warmup=" << opt.warmup
        << ";iters=" << opt.iters
        << ";seed=" << opt.seed
        << ";cta_m=" << opt.config.cta_m
        << ";cta_n=" << opt.config.cta_n
        << ";cta_k=" << opt.config.cta_k
        << ";threads=" << opt.config.threads
        << ";persistent_ctas=" << opt.config.persistent_ctas
        << ";tensor_cores_requested=" << (opt.config.use_tensor_cores ? 1 : 0)
        << ";tensor_cores_enabled=" << (occupancy.using_tensor_cores ? 1 : 0)
        << ";sm_count=" << occupancy.sm_count
        << ";active_blocks_per_sm=" << occupancy.active_blocks_per_sm
        << ";registers_per_thread=" << occupancy.registers_per_thread
        << ";dynamic_shared_bytes=" << occupancy.dynamic_shared_bytes;
  return oss.str();
}

// 尝试加载固定输出缓存。格式不合法或文件不存在时返回 false。
bool load_fixed_output(
    const std::string& path,
    std::string* out_fingerprint,
    std::vector<ModeResult>* out_results) {
  if (out_fingerprint == nullptr || out_results == nullptr) {
    return false;
  }

  std::ifstream in(path);
  if (!in.is_open()) {
    return false;
  }

  std::string key;
  std::string value;
  if (!(in >> key >> value)) {
    return false;
  }
  if (key != "magic" || value != "grouped_gemm_fixed_output_v1") {
    return false;
  }

  std::string fingerprint;
  size_t expected_count = 0;
  bool has_expected_count = false;
  std::vector<ModeResult> parsed_results;

  while (in >> key) {
    if (key == "fingerprint") {
      in >> fingerprint;
    } else if (key == "result_count") {
      in >> expected_count;
      has_expected_count = true;
    } else if (key == "result") {
      ModeResult r;
      in >> r.mode
         >> r.prepare_ms
         >> r.kernel_ms
         >> r.tflops
         >> r.total_tiles
         >> r.grid_ctas
         >> r.launch_count
         >> r.tail_work_share;
      if (!in.good()) {
        return false;
      }
      parsed_results.push_back(r);
    } else {
      std::string ignored;
      std::getline(in, ignored);
    }
  }

  if (fingerprint.empty() || parsed_results.empty()) {
    return false;
  }
  if (has_expected_count && parsed_results.size() != expected_count) {
    return false;
  }

  *out_fingerprint = fingerprint;
  *out_results = parsed_results;
  return true;
}

// 保存固定输出缓存；若目录不存在会自动创建。
bool save_fixed_output(
    const std::string& path,
    const std::string& fingerprint,
    const std::vector<ModeResult>& results) {
  std::filesystem::path file_path(path);
  if (file_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file_path.parent_path(), ec);
    if (ec) {
      std::cerr << "Failed to create fixed-output directory: " << ec.message() << std::endl;
      return false;
    }
  }

  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "Failed to open fixed-output file: " << path << std::endl;
    return false;
  }

  out << std::setprecision(9);
  out << "magic grouped_gemm_fixed_output_v1\n";
  out << "fingerprint " << fingerprint << "\n";
  out << "result_count " << results.size() << "\n";
  for (const auto& r : results) {
    out << "result "
        << r.mode << " "
        << r.prepare_ms << " "
        << r.kernel_ms << " "
        << r.tflops << " "
        << r.total_tiles << " "
        << r.grid_ctas << " "
        << r.launch_count << " "
        << r.tail_work_share << "\n";
  }
  return true;
}

// 按 workload 生成问题形状集合 (m, n, k)。
// small: 小矩阵；tail: 长尾；mixed: 宽范围混合。
std::vector<std::tuple<int, int, int>> generate_shapes(const BenchOptions& opt) {
  std::mt19937 rng(opt.seed);
  std::vector<std::tuple<int, int, int>> out;
  out.reserve(opt.problem_count);

  // 从候选值集合中均匀随机选一个。
  auto pick = [&](const std::vector<int>& values) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(values.size()) - 1);
    return values[dist(rng)];
  };

  if (opt.workload == "small") {
    // small：全部从小尺寸集合采样。
    const std::vector<int> m_values = {16, 24, 32, 48, 64};
    const std::vector<int> n_values = {16, 32, 48, 64, 96};
    const std::vector<int> k_values = {16, 24, 32, 48, 64};
    for (int i = 0; i < opt.problem_count; ++i) {
      out.emplace_back(pick(m_values), pick(n_values), pick(k_values));
    }
    return out;
  }

  if (opt.workload == "tail") {
    // tail：80% 小问题 + 20% 大问题，刻意制造负载不均衡。
    const std::vector<int> small_mn = {16, 24, 32, 48, 64};
    const std::vector<int> large_mn = {96, 128, 160, 192, 256};
    const std::vector<int> small_k = {16, 24, 32, 48, 64};
    const std::vector<int> large_k = {96, 128, 192, 256, 384, 512};

    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    for (int i = 0; i < opt.problem_count; ++i) {
      if (prob(rng) < 0.8f) {
        out.emplace_back(pick(small_mn), pick(small_mn), pick(small_k));
      } else {
        out.emplace_back(pick(large_mn), pick(large_mn), pick(large_k));
      }
    }
    return out;
  }

  // // mixed：在较宽尺寸空间均匀随机采样。
  // const std::vector<int> m_values = {16, 24, 32, 48, 64, 96, 128, 160};
  // const std::vector<int> n_values = {16, 32, 48, 64, 96, 128, 192};
  // const std::vector<int> k_values = {16, 24, 32, 48, 64, 96, 128, 192, 256};
  // for (int i = 0; i < opt.problem_count; ++i) {
  //   out.emplace_back(pick(m_values), pick(n_values), pick(k_values));
  // }
  // return out;

  // // mixed：在较宽尺寸空间均匀随机采样。
  // const std::vector<int> m_values = {64, 96, 128, 256, 512};
  // const std::vector<int> n_values = {64, 96, 128, 256, 512};
  // const std::vector<int> k_values = {64, 96, 128, 256, 512};
  // for (int i = 0; i < opt.problem_count; ++i) {
  //     out.emplace_back(pick(m_values), pick(n_values), pick(k_values));
  // }
  // return out;

  // // mixed：在较宽尺寸空间均匀随机采样。
  // const std::vector<int> m_values = {66, 98, 148, 300, 550};
  // const std::vector<int> n_values = {66, 98, 148, 300, 550};
  // const std::vector<int> k_values = {66, 98, 148, 300, 550};
  // for (int i = 0; i < opt.problem_count; ++i) {
  //     out.emplace_back(pick(m_values), pick(n_values), pick(k_values));
  // }
  // return out;

  // mixed：在较宽尺寸空间均匀随机采样。
  const std::vector<int> m_values = {63, 94, 122, 240, 500};
  const std::vector<int> n_values = {63, 94, 122, 240, 500};
  const std::vector<int> k_values = {63, 94, 122, 240, 500};
  for (int i = 0; i < opt.problem_count; ++i) {
      out.emplace_back(pick(m_values), pick(n_values), pick(k_values));
  }
  return out;
}

// 初始化所有问题的 host/device 缓冲，并构建 ProblemDesc 列表。
// 该函数完成随机赋值、显存分配、H2D 拷贝和描述符打包。
void init_problem_buffers(
    const std::vector<std::tuple<int, int, int>>& shapes,
    int seed,
    std::vector<ProblemBuffer>* out_buffers,
    std::vector<grouped_gemm::ProblemDesc>* out_problems) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // 提前清空并预留容量，减少扩容开销。
  out_buffers->clear();
  out_buffers->reserve(shapes.size());
  out_problems->clear();
  out_problems->reserve(shapes.size());

  for (const auto& [m, n, k] : shapes) {
    // 创建并获取当前问题缓冲引用。
    out_buffers->emplace_back();
    ProblemBuffer& pb = out_buffers->back();
    pb.m = m;
    pb.n = n;
    pb.k = k;

    // 计算 leading dimension，至少为 1。
    pb.lda = std::max(1, k);
    pb.ldb = std::max(1, n);
    pb.ldc = std::max(1, n);
    pb.ldd = std::max(1, n);

    // 按 (rows, ld) 分配主机输入矩阵。
    pb.h_a.resize(safe_matrix_elems(pb.m, pb.lda));
    pb.h_b.resize(safe_matrix_elems(pb.k, pb.ldb));
    pb.h_c.resize(safe_matrix_elems(pb.m, pb.ldc));

    // 随机初始化 A/B/C。
    for (auto& x : pb.h_a) {
      x = __float2half_rn(dist(rng));
    }
    for (auto& x : pb.h_b) {
      x = __float2half_rn(dist(rng));
    }
    for (auto& x : pb.h_c) {
      x = dist(rng);
    }

    // 分配设备缓冲：A/B/C 输入，D 输出。
    CHECK_CUDA_OR_EXIT(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_a), pb.h_a.size() * sizeof(__half)));
    CHECK_CUDA_OR_EXIT(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_b), pb.h_b.size() * sizeof(__half)));
    CHECK_CUDA_OR_EXIT(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_c), pb.h_c.size() * sizeof(float)));
    CHECK_CUDA_OR_EXIT(cudaMalloc(
        reinterpret_cast<void**>(&pb.d_d),
        safe_matrix_elems(pb.m, pb.ldd) * sizeof(float)));

    // 把主机输入同步到设备。
    CHECK_CUDA_OR_EXIT(cudaMemcpy(
        pb.d_a,
        pb.h_a.data(),
        pb.h_a.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_OR_EXIT(cudaMemcpy(
        pb.d_b,
        pb.h_b.data(),
        pb.h_b.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_OR_EXIT(cudaMemcpy(
        pb.d_c,
        pb.h_c.data(),
        pb.h_c.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    // 填充 ProblemDesc，供 grouped_gemm 执行器消费。
    grouped_gemm::ProblemDesc p{};
    p.m = pb.m;
    p.n = pb.n;
    p.k = pb.k;
    p.a = pb.d_a;
    p.b = pb.d_b;
    p.c = pb.d_c;
    p.d = pb.d_d;
    p.lda = pb.lda;
    p.ldb = pb.ldb;
    p.ldc = pb.ldc;
    p.ldd = pb.ldd;
    p.alpha = 1.0f;
    p.beta = 1.0f;
    out_problems->push_back(p);
  }
}

// 通用计时函数：先 warmup，再统计 iters 次平均耗时。
// launch_fn 负责发射 kernel，成功时返回 cudaSuccess。
template <typename LaunchFn>
float benchmark_ms(int warmup, int iters, LaunchFn launch_fn) {
  // 预热以减少首次运行带来的测量抖动。
  for (int i = 0; i < warmup; ++i) {
    CHECK_CUDA_OR_EXIT(launch_fn());
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK_CUDA_OR_EXIT(cudaEventCreate(&start));
  CHECK_CUDA_OR_EXIT(cudaEventCreate(&stop));

  // 用 CUDA Event 仅测量设备端执行时长。
  CHECK_CUDA_OR_EXIT(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA_OR_EXIT(launch_fn());
  }
  CHECK_CUDA_OR_EXIT(cudaEventRecord(stop));
  CHECK_CUDA_OR_EXIT(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA_OR_EXIT(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CHECK_CUDA_OR_EXIT(cudaEventDestroy(start));
  CHECK_CUDA_OR_EXIT(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(iters);
}

// 统计总 FLOPs：每个问题按 2*m*n*k 估算。
// 用于后续换算 TFLOP/s。
double total_flops(const std::vector<grouped_gemm::ProblemDesc>& problems) {
  double flops = 0.0;
  for (const auto& p : problems) {
    flops += 2.0 * static_cast<double>(p.m) * p.n * p.k;
  }
  return flops;
}

// 估计尾部工作占比。
// 方法：按当前排序，把最后 10% 问题工作量之和除以总工作量。
double tail_work_share(
    const grouped_gemm::PreparedProblemSet& prepared,
    const grouped_gemm::KernelConfig& cfg) {
  const int n = static_cast<int>(prepared.sorted_problems.size());
  if (n == 0) {
    return 0.0;
  }

  // 至少保留 1 个尾部样本，避免小样本时统计失真。
  const int tail_count = std::max(1, n / 10);

  double total_work = 0.0;
  double tail_work = 0.0;

  for (int i = 0; i < n; ++i) {
    // 该问题的 tile 数量。
    const int tile_count = prepared.tile_offsets[i + 1] - prepared.tile_offsets[i];
    // 该问题在 k 方向的迭代步数。
    const int k_steps = ceil_div(prepared.sorted_problems[i].k, cfg.cta_k);
    const double work = static_cast<double>(tile_count) * k_steps;
    total_work += work;
    if (i >= n - tail_count) {
      tail_work += work;
    }
  }

  return total_work > 0.0 ? tail_work / total_work : 0.0;
}

// 运行一种 grouped 模式的完整路径：prepare -> build -> benchmark。
ModeResult run_grouped_mode(
    const std::vector<grouped_gemm::ProblemDesc>& problems,
    grouped_gemm::SortMode mode,
    const grouped_gemm::KernelConfig& config,
    int warmup,
    int iters,
    double flops) {
  ModeResult result;
  result.mode = grouped_gemm::sort_mode_name(mode);
  // persistent grouped 路径通常只发射一次。
  result.launch_count = 1;

  const auto t0 = std::chrono::high_resolution_clock::now();
  // 预处理：按 mode 重排问题并生成 tile 偏移等元数据。
  const grouped_gemm::PreparedProblemSet prepared =
      grouped_gemm::prepare_problem_set(problems, config, mode);

  // 构建执行器：准备设备侧调度数据。
  grouped_gemm::GroupedGemmExecutor executor;
  CHECK_CUDA_OR_EXIT(executor.build(prepared, config));

  const auto t1 = std::chrono::high_resolution_clock::now();
  // 记录准备阶段开销。
  result.prepare_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

  result.total_tiles = prepared.total_tiles;
  result.grid_ctas = executor.grid_ctas();
  result.tail_work_share = tail_work_share(prepared, config);

  // 统计 kernel 平均耗时。
  result.kernel_ms = benchmark_ms(warmup, iters, [&]() {
    return executor.run(nullptr);
  });

  // 吞吐率换算：TFLOP/s = FLOPs / seconds / 1e12。
  result.tflops = (flops / 1e12) / (result.kernel_ms / 1e3);
  return result;
}

// 运行 baseline 对照路径：每个问题单独发射 kernel。
ModeResult run_baseline(
    const std::vector<grouped_gemm::ProblemDesc>& problems,
    const grouped_gemm::KernelConfig& config,
    int warmup,
    int iters,
    double flops,
    int total_tiles) {
  ModeResult result;
  result.mode = "baseline_per_problem";
  // baseline 的 launch 次数等于非空问题数。
  result.launch_count = grouped_gemm::count_non_empty_problems(problems, config);
  result.total_tiles = total_tiles;
  result.grid_ctas = 0;
  result.tail_work_share = 0.0;
  result.prepare_ms = 0.0f;

  result.kernel_ms = benchmark_ms(warmup, iters, [&]() {
    return grouped_gemm::run_baseline_per_problem(problems, config, nullptr);
  });

  result.tflops = (flops / 1e12) / (result.kernel_ms / 1e3);
  return result;
}

// 表格化打印所有模式结果，便于横向对比。
void print_results(const std::vector<ModeResult>& results) {
  std::cout << std::left << std::setw(24) << "mode"
            << std::setw(12) << "prep_ms"
            << std::setw(12) << "kernel_ms"
            << std::setw(12) << "tflops"
            << std::setw(12) << "tiles"
            << std::setw(12) << "grid_ctas"
            << std::setw(14) << "launches"
            << std::setw(14) << "tail_share"
            << std::endl;

  // 每一行对应一种执行模式。
  for (const auto& r : results) {
    std::cout << std::left << std::setw(24) << r.mode
              << std::setw(12) << std::fixed << std::setprecision(3) << r.prepare_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << r.kernel_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << r.tflops
              << std::setw(12) << r.total_tiles
              << std::setw(12) << r.grid_ctas
              << std::setw(14) << r.launch_count
              << std::setw(14) << std::fixed << std::setprecision(3) << r.tail_work_share
              << std::endl;
  }
}

}  // namespace

// 主入口流程：
// 1) 解析参数；2) 构造输入；3) 跑 grouped/baseline；4) 打印汇总。
int main(int argc, char** argv) {
  BenchOptions opt;
  if (!parse_options(argc, argv, &opt)) {
    print_usage(argv[0]);
    return 1;
  }

  // 先查询 occupancy，可用于结果展示和 fixed output 指纹。
  grouped_gemm::OccupancyInfo occupancy{};
  CHECK_CUDA_OR_EXIT(grouped_gemm::query_occupancy(opt.config, &occupancy));

  std::cout << "Workload: " << opt.workload << ", problems=" << opt.problem_count
            << ", warmup=" << opt.warmup << ", iters=" << opt.iters << std::endl;
  std::cout << "Kernel config: CTA(" << opt.config.cta_m << ", " << opt.config.cta_n
            << ", " << opt.config.cta_k << "), threads=" << opt.config.threads
            << ", persistent_ctas=" << opt.config.persistent_ctas
            << ", tensor_cores_requested=" << (opt.config.use_tensor_cores ? "on" : "off") << std::endl;
  std::cout << "Runtime mode: tensor_cores_enabled="
            << (occupancy.using_tensor_cores ? "yes" : "no") << std::endl;
  std::cout << "Occupancy: active_blocks_per_sm=" << occupancy.active_blocks_per_sm
            << ", max_blocks_per_sm=" << occupancy.max_blocks_per_sm
            << ", sm_count=" << occupancy.sm_count
            << ", theoretical=" << std::fixed << std::setprecision(2)
            << (occupancy.theoretical_occupancy * 100.0f) << "%" << std::endl;
  std::cout << "Register pressure: estimated_regs_per_thread="
            << occupancy.estimated_registers_per_thread
            << ", measured_regs_per_thread=" << occupancy.registers_per_thread
            << ", regs_per_block=" << occupancy.registers_per_block
            << ", register_limited_blocks_per_sm=" << occupancy.register_limited_blocks_per_sm
            << std::endl;
  std::cout << "Shared memory: static=" << occupancy.static_shared_bytes
            << "B, dynamic=" << occupancy.dynamic_shared_bytes << "B" << std::endl;

  const std::string fixed_output_fingerprint =
      make_fixed_output_fingerprint(opt, occupancy);

  if (!opt.fixed_output_path.empty()) {
    std::string cached_fingerprint;
    std::vector<ModeResult> cached_results;
    if (load_fixed_output(opt.fixed_output_path, &cached_fingerprint, &cached_results) &&
        cached_fingerprint == fixed_output_fingerprint) {
      print_results(cached_results);
      return 0;
    }
  }

  // 先生成问题形状。
  const std::vector<std::tuple<int, int, int>> shapes = generate_shapes(opt);

  // 再初始化 host/device 缓冲与 ProblemDesc。
  std::vector<ProblemBuffer> buffers;
  std::vector<grouped_gemm::ProblemDesc> problems;
  init_problem_buffers(shapes, opt.seed + 1, &buffers, &problems);

  // 计算总 FLOPs，作为所有模式共享的工作量基准。
  const double flops = total_flops(problems);

  std::vector<ModeResult> results;
  results.reserve(5);

  // 依次测试四种 grouped 排序模式。
  const std::vector<grouped_gemm::SortMode> modes = {
      grouped_gemm::SortMode::kNone,
      grouped_gemm::SortMode::kByK,
      grouped_gemm::SortMode::kByTiles,
      grouped_gemm::SortMode::kByTilesThenK,
  };

  // total_tiles 在同一问题集下与排序模式无关，记录一次供 baseline 复用 。
  int total_tiles = 0;
  for (grouped_gemm::SortMode mode : modes) {
    ModeResult result = run_grouped_mode(
        problems,
        mode,
        opt.config,
        opt.warmup,
        opt.iters,
        flops);
    total_tiles = result.total_tiles;
    results.push_back(result);
  }

  // 追加 baseline 对照结果。
  results.push_back(run_baseline(
      problems,
      opt.config,
      opt.warmup,
      opt.iters,
      flops,
      total_tiles));

  if (!opt.fixed_output_path.empty()) {
    if (!save_fixed_output(opt.fixed_output_path, fixed_output_fingerprint, results)) {
      std::cerr << "Warning: fixed-output cache was not written." << std::endl;
    }
  }

  // 打印最终汇总表。
  print_results(results);

  return 0;
}
