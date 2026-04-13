#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace grouped_gemm {

enum class SortMode {
  kNone = 0,
  kByK = 1,
  kByTiles = 2,
  kByTilesThenK = 3,
};

struct ProblemDesc {
  int m;
  int n;
  int k;

  const __half* a;
  const __half* b;
  const float* c;
  float* d;

  int lda;
  int ldb;
  int ldc;
  int ldd;

  float alpha = 1.0f;
  float beta = 0.0f;
};

struct KernelConfig {
  int cta_m = 64;
  int cta_n = 64;
  int cta_k = 16;
  int threads = 256;

  int persistent_ctas = 0;
  bool use_tensor_cores = true;
};

struct PreparedProblemSet {
  std::vector<ProblemDesc> sorted_problems;
  std::vector<int> sorted_to_original;
  std::vector<int> tile_offsets;
  int total_tiles = 0;
};

// 运行时资源统计：包含 occupancy、共享内存和寄存器压力。
struct OccupancyInfo {
  int sm_count = 0;
  int max_blocks_per_sm = 0;
  int active_blocks_per_sm = 0;

  int threads_per_block = 0;
  int max_threads_per_sm = 0;
  int warps_per_block = 0;
  int max_warps_per_sm = 0;
  int active_warps_per_sm = 0;

  int max_registers_per_sm = 0;
  int max_registers_per_block = 0;

  // 公式估算值（实现内根据 CTA 与 MMA 参数计算）。
  int estimated_registers_per_thread = 0;
  int estimated_registers_per_block = 0;

  // 编译器实测值（cudaFuncGetAttributes）。
  int registers_per_thread = 0;
  int registers_per_block = 0;
  int register_limited_blocks_per_sm = 0;

  int static_shared_bytes = 0;
  int dynamic_shared_bytes = 0;

  float theoretical_occupancy = 0.0f;
  bool using_tensor_cores = false;
};

PreparedProblemSet prepare_problem_set(
    const std::vector<ProblemDesc>& problems,
    const KernelConfig& config,
    SortMode sort_mode);

const char* sort_mode_name(SortMode sort_mode);

cudaError_t query_occupancy(const KernelConfig& config, OccupancyInfo* info);

class GroupedGemmExecutor {
 public:
  GroupedGemmExecutor();
  ~GroupedGemmExecutor();

  GroupedGemmExecutor(const GroupedGemmExecutor&) = delete;
  GroupedGemmExecutor& operator=(const GroupedGemmExecutor&) = delete;

  GroupedGemmExecutor(GroupedGemmExecutor&& other) noexcept;
  GroupedGemmExecutor& operator=(GroupedGemmExecutor&& other) noexcept;

  cudaError_t build(const PreparedProblemSet& prepared, const KernelConfig& config);
  cudaError_t run(cudaStream_t stream = nullptr) const;

  int problem_count() const;
  int total_tiles() const;
  int grid_ctas() const;

 private:
  struct Impl;
  Impl* impl_;
};

cudaError_t run_grouped_gemm_once(
    const std::vector<ProblemDesc>& problems,
    SortMode sort_mode,
    const KernelConfig& config,
    cudaStream_t stream = nullptr);

cudaError_t run_baseline_per_problem(
    const std::vector<ProblemDesc>& problems,
    const KernelConfig& config,
    cudaStream_t stream = nullptr);

int count_non_empty_problems(
    const std::vector<ProblemDesc>& problems,
    const KernelConfig& config);

}  // namespace grouped_gemm
