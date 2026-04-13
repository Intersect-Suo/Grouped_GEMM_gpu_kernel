#include "grouped_gemm/grouped_gemm.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

namespace {

// 测试路径下统一的 CUDA 调用检查宏。
// 一旦 CUDA API 返回错误，立即打印定位信息并返回 false，
// 让当前测试分支快速失败，便于定位问题来源。
#define CHECK_CUDA_OR_RETURN_FALSE(call)                                              \
  do {                                                                                 \
    cudaError_t _status = (call);                                                     \
    if (_status != cudaSuccess) {                                                     \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                << cudaGetErrorString(_status) << std::endl;                          \
      return false;                                                                    \
    }                                                                                  \
  } while (0)

// 正确性测试中每个 GEMM 子问题的完整存储单元。
// 同时保存主机侧输入/输出/参考结果与设备侧指针，
// 以及 m/n/k、步长和 alpha/beta 参数。
struct ProblemBuffer {
  int m = 0;
  int n = 0;
  int k = 0;

  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int ldd = 0;

  float alpha = 1.0f;
  float beta = 1.0f;

  std::vector<__half> h_a;
  std::vector<__half> h_b;
  std::vector<float> h_c;
  std::vector<float> h_d;
  std::vector<float> h_ref;

  __half* d_a = nullptr;
  __half* d_b = nullptr;
  float* d_c = nullptr;
  float* d_d = nullptr;

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

// 返回至少 1 个元素。
// 即使逻辑维度为 0，也能为分配与拷贝提供合法缓冲区，
// 让边界场景沿用统一的数据通路。
size_t safe_matrix_elems(int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    return 1;
  }
  return static_cast<size_t>(rows) * cols;
}

// 单个问题的 CPU 参考 GEMM：
// D = alpha * A * B + beta * C。
// 使用显式 leading dimension 做索引，用于与 GPU 输出逐元素比对。
void compute_reference(ProblemBuffer& pb) {
  if (pb.m <= 0 || pb.n <= 0) {
    return;
  }

  for (int i = 0; i < pb.m; ++i) {
    for (int j = 0; j < pb.n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < pb.k; ++kk) {
        const float a_val = __half2float(pb.h_a[static_cast<size_t>(i) * pb.lda + kk]);
        const float b_val = __half2float(pb.h_b[static_cast<size_t>(kk) * pb.ldb + j]);
        acc += a_val * b_val;
      }
      const float c_val = pb.h_c[static_cast<size_t>(i) * pb.ldc + j];
      pb.h_ref[static_cast<size_t>(i) * pb.ldd + j] = pb.alpha * acc + pb.beta * c_val;
    }
  }
}

// 为所有测试形状初始化主机/设备缓冲区，
// 填充随机输入并构建 grouped_gemm::ProblemDesc 列表。
bool init_problem_buffers(
    const std::vector<std::tuple<int, int, int>>& shapes,
    std::vector<ProblemBuffer>* out_buffers,
    std::vector<grouped_gemm::ProblemDesc>* out_problems) {
  if (out_buffers == nullptr || out_problems == nullptr) {
    return false;
  }

  std::mt19937 rng(20260328);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  out_buffers->clear();
  out_buffers->reserve(shapes.size());
  out_problems->clear();
  out_problems->reserve(shapes.size());

  for (const auto& [m, n, k] : shapes) {
    out_buffers->emplace_back();
    ProblemBuffer& pb = out_buffers->back();
    pb.m = m;
    pb.n = n;
    pb.k = k;
    pb.lda = std::max(1, k);
    pb.ldb = std::max(1, n);
    pb.ldc = std::max(1, n);
    pb.ldd = std::max(1, n);

    pb.alpha = dist(rng);
    pb.beta = dist(rng);

    pb.h_a.resize(safe_matrix_elems(pb.m, pb.lda));
    pb.h_b.resize(safe_matrix_elems(pb.k, pb.ldb));
    pb.h_c.resize(safe_matrix_elems(pb.m, pb.ldc));
    pb.h_d.resize(safe_matrix_elems(pb.m, pb.ldd), 0.0f);
    pb.h_ref.resize(safe_matrix_elems(pb.m, pb.ldd), 0.0f);

    for (auto& x : pb.h_a) {
      x = __float2half_rn(dist(rng));
    }
    for (auto& x : pb.h_b) {
      x = __float2half_rn(dist(rng));
    }
    for (auto& x : pb.h_c) {
      x = dist(rng);
    }

    compute_reference(pb);

    CHECK_CUDA_OR_RETURN_FALSE(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_a), pb.h_a.size() * sizeof(__half)));
    CHECK_CUDA_OR_RETURN_FALSE(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_b), pb.h_b.size() * sizeof(__half)));
    CHECK_CUDA_OR_RETURN_FALSE(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_c), pb.h_c.size() * sizeof(float)));
    CHECK_CUDA_OR_RETURN_FALSE(
        cudaMalloc(reinterpret_cast<void**>(&pb.d_d), pb.h_d.size() * sizeof(float)));

    CHECK_CUDA_OR_RETURN_FALSE(cudaMemcpy(
        pb.d_a,
        pb.h_a.data(),
        pb.h_a.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_OR_RETURN_FALSE(cudaMemcpy(
        pb.d_b,
        pb.h_b.data(),
        pb.h_b.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    CHECK_CUDA_OR_RETURN_FALSE(cudaMemcpy(
        pb.d_c,
        pb.h_c.data(),
        pb.h_c.size() * sizeof(float),
        cudaMemcpyHostToDevice));

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
    p.alpha = pb.alpha;
    p.beta = pb.beta;
    out_problems->push_back(p);
  }

  return true;
}

// 将每个问题的设备侧输出 D 拷回主机，供后续误差校验。
bool copy_device_outputs(std::vector<ProblemBuffer>* buffers) {
  if (buffers == nullptr) {
    return false;
  }
  for (ProblemBuffer& pb : *buffers) {
    CHECK_CUDA_OR_RETURN_FALSE(cudaMemcpy(
        pb.h_d.data(),
        pb.d_d,
        pb.h_d.size() * sizeof(float),
        cudaMemcpyDeviceToHost));
  }
  return true;
}

// 按绝对误差阈值比较 GPU 结果与 CPU 参考值。
// 一旦超阈值，立即输出问题编号与坐标并返回失败。
bool check_outputs(
    const std::vector<ProblemBuffer>& buffers,
    float tolerance,
    float* out_max_error) {
  float max_error = 0.0f;

  for (size_t problem_idx = 0; problem_idx < buffers.size(); ++problem_idx) {
    const ProblemBuffer& pb = buffers[problem_idx];
    if (pb.m <= 0 || pb.n <= 0) {
      continue;
    }

    for (int i = 0; i < pb.m; ++i) {
      for (int j = 0; j < pb.n; ++j) {
        const size_t offset = static_cast<size_t>(i) * pb.ldd + j;
        const float diff = std::fabs(pb.h_d[offset] - pb.h_ref[offset]);
        max_error = std::max(max_error, diff);
        if (diff > tolerance) {
          std::cerr << "Mismatch at problem " << problem_idx << ", (" << i << ", " << j
                    << "), got=" << pb.h_d[offset] << ", ref=" << pb.h_ref[offset]
                    << ", diff=" << diff << std::endl;
          return false;
        }
      }
    }
  }

  if (out_max_error != nullptr) {
    *out_max_error = max_error;
  }
  return true;
}

// 依次运行 grouped GEMM 的各排序模式并做正确性校验，
// 随后运行 baseline（逐问题发射）路径并同样校验。
bool run_modes_and_check(
    const std::vector<grouped_gemm::SortMode>& modes,
    const std::vector<grouped_gemm::ProblemDesc>& problems,
    std::vector<ProblemBuffer>* buffers,
    const grouped_gemm::KernelConfig& config,
    float tolerance) {
  for (const grouped_gemm::SortMode mode : modes) {
    CHECK_CUDA_OR_RETURN_FALSE(
        grouped_gemm::run_grouped_gemm_once(problems, mode, config, nullptr));
    CHECK_CUDA_OR_RETURN_FALSE(cudaDeviceSynchronize());
    if (!copy_device_outputs(buffers)) {
      return false;
    }

    float max_error = 0.0f;
    if (!check_outputs(*buffers, tolerance, &max_error)) {
      std::cerr << "Grouped GEMM failed for mode=" << grouped_gemm::sort_mode_name(mode)
                << std::endl;
      return false;
    }

    std::cout << "[PASS] grouped mode=" << grouped_gemm::sort_mode_name(mode)
              << ", max_abs_error=" << std::setprecision(6) << max_error << std::endl;
  }

  CHECK_CUDA_OR_RETURN_FALSE(
      grouped_gemm::run_baseline_per_problem(problems, config, nullptr));
  CHECK_CUDA_OR_RETURN_FALSE(cudaDeviceSynchronize());

  if (!copy_device_outputs(buffers)) {
    return false;
  }

  float baseline_error = 0.0f;
  if (!check_outputs(*buffers, tolerance, &baseline_error)) {
    std::cerr << "Baseline per-problem GEMM failed" << std::endl;
    return false;
  }

  std::cout << "[PASS] baseline per-problem, max_abs_error=" << std::setprecision(6)
            << baseline_error << std::endl;
  return true;
}

// 验证空问题集输入时，grouped 与 baseline 两条路径均可正常返回。
bool run_empty_problem_test(const grouped_gemm::KernelConfig& config) {
  std::vector<grouped_gemm::ProblemDesc> empty;
  CHECK_CUDA_OR_RETURN_FALSE(grouped_gemm::run_grouped_gemm_once(
      empty,
      grouped_gemm::SortMode::kByTiles,
      config,
      nullptr));
  CHECK_CUDA_OR_RETURN_FALSE(grouped_gemm::run_baseline_per_problem(empty, config, nullptr));
  CHECK_CUDA_OR_RETURN_FALSE(cudaDeviceSynchronize());
  std::cout << "[PASS] empty problem set" << std::endl;
  return true;
}

// 在单个 kernel 配置下执行完整正确性测试。
// 形状集合混合规则尺寸、非对齐尺寸和 0 维度问题，
// 用于覆盖常见与边界场景。
bool run_one_config(const grouped_gemm::KernelConfig& config) {
  std::vector<std::tuple<int, int, int>> shapes = {
      {64, 64, 64},
      {128, 64, 96},
      {45, 81, 37},
      {17, 23, 9},
      {192, 32, 64},
      {16, 256, 48},
      {0, 64, 16},
      {64, 0, 16},
  };

  std::vector<ProblemBuffer> buffers;
  std::vector<grouped_gemm::ProblemDesc> problems;
  if (!init_problem_buffers(shapes, &buffers, &problems)) {
    return false;
  }

  grouped_gemm::OccupancyInfo occupancy{};
  CHECK_CUDA_OR_RETURN_FALSE(grouped_gemm::query_occupancy(config, &occupancy));

  std::cout << "Running config: CTA(" << config.cta_m << ", " << config.cta_n << ", "
            << config.cta_k << "), threads=" << config.threads
            << ", tensor_cores_requested=" << (config.use_tensor_cores ? "on" : "off")
            << std::endl;
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
            << std::endl;

  const std::vector<grouped_gemm::SortMode> modes = {
      grouped_gemm::SortMode::kNone,
      grouped_gemm::SortMode::kByK,
      grouped_gemm::SortMode::kByTiles,
      grouped_gemm::SortMode::kByTilesThenK,
  };

  const float tolerance = 2e-2f;
  if (!run_modes_and_check(modes, problems, &buffers, config, tolerance)) {
    return false;
  }
  return run_empty_problem_test(config);
}

}  // namespace

int main() {
  grouped_gemm::KernelConfig config0{};
  config0.cta_m = 64;
  config0.cta_n = 64;
  config0.cta_k = 16;
  config0.threads = 256;
  config0.persistent_ctas = 0;
  config0.use_tensor_cores = true;

  grouped_gemm::KernelConfig config1{};
  config1.cta_m = 128;
  config1.cta_n = 64;
  config1.cta_k = 32;
  config1.threads = 256;
  config1.persistent_ctas = 0;
  config1.use_tensor_cores = true;

  bool ok = run_one_config(config0) && run_one_config(config1);
  if (!ok) {
    std::cerr << "Correctness test failed." << std::endl;
    return 1;
  }

  std::cout << "All correctness checks passed." << std::endl;
  return 0;
}
