#include "grouped_gemm/grouped_gemm.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mma.h>
#include <numeric>
#include <utility>

namespace grouped_gemm {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kThreadsPerBlock = 256;
constexpr int kThreadGroupsM = 16;
constexpr int kThreadGroupsN = 16;
constexpr int kPipelineStages = 2;
constexpr int kWarpSize = 32;

constexpr int kMmaM = 16;
constexpr int kMmaN = 16;
constexpr int kMmaK = 16;

struct DeviceProblemDesc {
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

  float alpha;
  float beta;

  int tiles_m;
  int tiles_n;
};

enum class KernelVariant {
  k64x64x16,
  k128x64x32,
  k64x128x32,
  kUnsupported,
};

__host__ __device__ inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

inline int tile_count_for_problem(const ProblemDesc& p, const KernelConfig& cfg) {
  if (p.m <= 0 || p.n <= 0) {
    return 0;
  }
  const int tiles_m = ceil_div(p.m, cfg.cta_m);
  const int tiles_n = ceil_div(p.n, cfg.cta_n);
  return tiles_m * tiles_n;
}

KernelVariant resolve_variant(const KernelConfig& config) {
  if (config.threads != kThreadsPerBlock) {
    return KernelVariant::kUnsupported;
  }
  if (config.cta_m == 64 && config.cta_n == 64 && config.cta_k == 16) {
    return KernelVariant::k64x64x16;
  }
  if (config.cta_m == 128 && config.cta_n == 64 && config.cta_k == 32) {
    return KernelVariant::k128x64x32;
  }
  if (config.cta_m == 64 && config.cta_n == 128 && config.cta_k == 32) {
    return KernelVariant::k64x128x32;
  }
  return KernelVariant::kUnsupported;
}

bool variant_supports_tensor_cores(KernelVariant variant) {
  switch (variant) {
    case KernelVariant::k64x64x16:
    case KernelVariant::k128x64x32:
    case KernelVariant::k64x128x32:
      return true;
    case KernelVariant::kUnsupported:
    default:
      return false;
  }
}

bool can_use_tensor_cores(const KernelConfig& config, KernelVariant variant, const cudaDeviceProp& prop) {
  if (!config.use_tensor_cores) {
    return false;
  }
  if (!variant_supports_tensor_cores(variant)) {
    return false;
  }
  if (prop.major < 7) {
    return false;
  }
  if (config.cta_m % kMmaM != 0 || config.cta_n % kMmaN != 0 || config.cta_k % kMmaK != 0) {
    return false;
  }
  return true;
}

cudaError_t resolve_runtime_mode(
    const KernelConfig& config,
    KernelVariant* out_variant,
    bool* out_use_tensor_cores,
    cudaDeviceProp* out_prop) {
  if (out_variant == nullptr || out_use_tensor_cores == nullptr) {
    return cudaErrorInvalidValue;
  }

  const KernelVariant variant = resolve_variant(config);
  if (variant == KernelVariant::kUnsupported) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  cudaDeviceProp prop{};
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) {
    return status;
  }

  *out_variant = variant;
  *out_use_tensor_cores = can_use_tensor_cores(config, variant, prop);
  if (out_prop != nullptr) {
    *out_prop = prop;
  }
  return cudaSuccess;
}

template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
constexpr size_t dynamic_shared_bytes_for_variant() {
  size_t bytes =
      kPipelineStages * (CTA_M * CTA_K + CTA_K * CTA_N) * sizeof(__half);
  if constexpr (UseTensorCores) {
    bytes += CTA_M * CTA_N * sizeof(float);
  }
  return bytes;
}

size_t shared_bytes_for(KernelVariant variant, bool use_tensor_cores) {
  switch (variant) {
    case KernelVariant::k64x64x16:
      return use_tensor_cores
          ? dynamic_shared_bytes_for_variant<64, 64, 16, true>()
          : dynamic_shared_bytes_for_variant<64, 64, 16, false>();
    case KernelVariant::k128x64x32:
      return use_tensor_cores
          ? dynamic_shared_bytes_for_variant<128, 64, 32, true>()
          : dynamic_shared_bytes_for_variant<128, 64, 32, false>();
    case KernelVariant::k64x128x32:
      return use_tensor_cores
          ? dynamic_shared_bytes_for_variant<64, 128, 32, true>()
          : dynamic_shared_bytes_for_variant<64, 128, 32, false>();
    case KernelVariant::kUnsupported:
    default:
      return 0;
  }
}

template <int CTA_M, int CTA_N>
constexpr int estimated_simt_registers_per_thread() {
  constexpr int rows_per_thread = CTA_M / kThreadGroupsM;
  constexpr int cols_per_thread = CTA_N / kThreadGroupsN;
  constexpr int accum_registers = rows_per_thread * cols_per_thread;
  constexpr int b_cache_registers = cols_per_thread;
  constexpr int loop_and_pointer_registers = 20;
  return accum_registers + b_cache_registers + loop_and_pointer_registers;
}

template <int CTA_M, int CTA_N>
constexpr int estimated_tensor_core_registers_per_thread() {
  constexpr int warp_count = kThreadsPerBlock / kWarpSize;
  constexpr int mma_tiles = (CTA_M / kMmaM) * (CTA_N / kMmaN);
  constexpr int max_tiles_per_warp = (mma_tiles + warp_count - 1) / warp_count;

  // 16x16x16 的 wmma accumulator fragment 会在每个 lane 上持有 8 个 FP32 元素。
  constexpr int accum_fragment_registers = max_tiles_per_warp * 8;
  // A/B 操作数 fragment 各约占 8 个寄存器量级。
  constexpr int operand_fragment_registers = 8 + 8;
  constexpr int loop_and_pointer_registers = 20;
  return accum_fragment_registers + operand_fragment_registers + loop_and_pointer_registers;
}

int estimated_registers_per_thread(KernelVariant variant, bool use_tensor_cores) {
  switch (variant) {
    case KernelVariant::k64x64x16:
      return use_tensor_cores
          ? estimated_tensor_core_registers_per_thread<64, 64>()
          : estimated_simt_registers_per_thread<64, 64>();
    case KernelVariant::k128x64x32:
      return use_tensor_cores
          ? estimated_tensor_core_registers_per_thread<128, 64>()
          : estimated_simt_registers_per_thread<128, 64>();
    case KernelVariant::k64x128x32:
      return use_tensor_cores
          ? estimated_tensor_core_registers_per_thread<64, 128>()
          : estimated_simt_registers_per_thread<64, 128>();
    case KernelVariant::kUnsupported:
    default:
      return 0;
  }
}

__device__ __forceinline__ int find_problem_by_tile(
    const int* tile_offsets,
    int problem_count,
    int tile_index) {
  int low = 0;
  int high = problem_count;
  while (low < high) {
    const int mid = (low + high) >> 1;
    if (tile_offsets[mid + 1] <= tile_index) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ void cp_async_cg_16(void* shared_dst, const void* global_src) {
  const unsigned int shared_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(shared_addr), "l"(global_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" : :);
}

__device__ __forceinline__ void cp_async_wait_group0() {
  asm volatile("cp.async.wait_group 0;\n" : :);
}

#else

__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
  (void)ptr;
  return false;
}

__device__ __forceinline__ void cp_async_cg_16(void* shared_dst, const void* global_src) {
  (void)shared_dst;
  (void)global_src;
}

__device__ __forceinline__ void cp_async_commit_group() {}
__device__ __forceinline__ void cp_async_wait_group0() {}

#endif

template <int CTA_M, int CTA_K>
__device__ void preload_a_stage(
    const DeviceProblemDesc& problem,
    int tile_m,
    int k_tile,
    __half* shared_a_stage) {
  static_assert(CTA_K % 8 == 0, "CTA_K must be divisible by 8");
  constexpr int kHalfPerVec = 8;
  constexpr int kVecCols = CTA_K / kHalfPerVec;
  constexpr int kTotalVec = CTA_M * kVecCols;

  const int tid = static_cast<int>(threadIdx.x);
  for (int vec_idx = tid; vec_idx < kTotalVec; vec_idx += blockDim.x) {
    const int row = vec_idx / kVecCols;
    const int vec_col = vec_idx - row * kVecCols;
    const int col = vec_col * kHalfPerVec;

    const int g_row = tile_m * CTA_M + row;
    const int g_col = k_tile * CTA_K + col;

    __half* dst = shared_a_stage + row * CTA_K + col;

    const bool full_in_bounds =
        (g_row < problem.m) && (g_col + kHalfPerVec <= problem.k);
    const __half* src = problem.a;
    if (full_in_bounds) {
      const int64_t offset = static_cast<int64_t>(g_row) * problem.lda + g_col;
      src = problem.a + offset;
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_in_bounds && is_aligned_16(dst) && is_aligned_16(src)) {
      cp_async_cg_16(dst, src);
      continue;
    }
#endif

#pragma unroll
    for (int i = 0; i < kHalfPerVec; ++i) {
      const int g_col_i = g_col + i;
      __half value = __float2half_rn(0.0f);
      if (g_row < problem.m && g_col_i < problem.k) {
        const int64_t offset = static_cast<int64_t>(g_row) * problem.lda + g_col_i;
        value = problem.a[offset];
      }
      dst[i] = value;
    }
  }
}

template <int CTA_N, int CTA_K>
__device__ void preload_b_stage(
    const DeviceProblemDesc& problem,
    int tile_n,
    int k_tile,
    __half* shared_b_stage) {
  static_assert(CTA_N % 8 == 0, "CTA_N must be divisible by 8");
  constexpr int kHalfPerVec = 8;
  constexpr int kVecCols = CTA_N / kHalfPerVec;
  constexpr int kTotalVec = CTA_K * kVecCols;

  const int tid = static_cast<int>(threadIdx.x);
  for (int vec_idx = tid; vec_idx < kTotalVec; vec_idx += blockDim.x) {
    const int row = vec_idx / kVecCols;
    const int vec_col = vec_idx - row * kVecCols;
    const int col = vec_col * kHalfPerVec;

    const int g_row = k_tile * CTA_K + row;
    const int g_col = tile_n * CTA_N + col;

    __half* dst = shared_b_stage + row * CTA_N + col;

    const bool full_in_bounds =
        (g_row < problem.k) && (g_col + kHalfPerVec <= problem.n);
    const __half* src = problem.b;
    if (full_in_bounds) {
      const int64_t offset = static_cast<int64_t>(g_row) * problem.ldb + g_col;
      src = problem.b + offset;
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (full_in_bounds && is_aligned_16(dst) && is_aligned_16(src)) {
      cp_async_cg_16(dst, src);
      continue;
    }
#endif

#pragma unroll
    for (int i = 0; i < kHalfPerVec; ++i) {
      const int g_col_i = g_col + i;
      __half value = __float2half_rn(0.0f);
      if (g_row < problem.k && g_col_i < problem.n) {
        const int64_t offset = static_cast<int64_t>(g_row) * problem.ldb + g_col_i;
        value = problem.b[offset];
      }
      dst[i] = value;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K>
__device__ void compute_gemm_tile_simt(
    const DeviceProblemDesc& problem,
    int tile_m,
    int tile_n) {
  static_assert(CTA_M % kThreadGroupsM == 0, "CTA_M must be divisible by 16");
  static_assert(CTA_N % kThreadGroupsN == 0, "CTA_N must be divisible by 16");
  static_assert(CTA_K % 8 == 0, "CTA_K must be divisible by 8");
  static_assert(CTA_N % 8 == 0, "CTA_N must be divisible by 8");

  constexpr int RowsPerThread = CTA_M / kThreadGroupsM;
  constexpr int ColsPerThread = CTA_N / kThreadGroupsN;

  const int tid = static_cast<int>(threadIdx.x);
  const int thread_group_m = tid / kThreadGroupsN;
  const int thread_group_n = tid % kThreadGroupsN;
  const int row_base = thread_group_m * RowsPerThread;
  const int col_base = thread_group_n * ColsPerThread;

  float accum[RowsPerThread][ColsPerThread];
#pragma unroll
  for (int i = 0; i < RowsPerThread; ++i) {
#pragma unroll
    for (int j = 0; j < ColsPerThread; ++j) {
      accum[i][j] = 0.0f;
    }
  }

  extern __shared__ __align__(16) unsigned char shared_raw[];
  constexpr int a_stage_elems = CTA_M * CTA_K;
  constexpr int b_stage_elems = CTA_K * CTA_N;

  __half* shared_a = reinterpret_cast<__half*>(shared_raw);
  __half* shared_b = shared_a + kPipelineStages * a_stage_elems;

  const int k_tiles = ceil_div(problem.k, CTA_K);
  if (k_tiles > 0) {
    preload_a_stage<CTA_M, CTA_K>(problem, tile_m, 0, shared_a + 0 * a_stage_elems);
    preload_b_stage<CTA_N, CTA_K>(problem, tile_n, 0, shared_b + 0 * b_stage_elems);
    cp_async_commit_group();

    for (int kt = 0; kt < k_tiles; ++kt) {
      const int read_stage = kt % kPipelineStages;
      __half* shared_a_stage = shared_a + read_stage * a_stage_elems;
      __half* shared_b_stage = shared_b + read_stage * b_stage_elems;

      cp_async_wait_group0();
      __syncthreads();

      const int next_k = kt + 1;
      if (next_k < k_tiles) {
        const int write_stage = next_k % kPipelineStages;
        preload_a_stage<CTA_M, CTA_K>(
            problem,
            tile_m,
            next_k,
            shared_a + write_stage * a_stage_elems);
        preload_b_stage<CTA_N, CTA_K>(
            problem,
            tile_n,
            next_k,
            shared_b + write_stage * b_stage_elems);
        cp_async_commit_group();
      }

#pragma unroll
      for (int kk = 0; kk < CTA_K; ++kk) {
        float b_values[ColsPerThread];
#pragma unroll
        for (int j = 0; j < ColsPerThread; ++j) {
          b_values[j] = __half2float(shared_b_stage[kk * CTA_N + col_base + j]);
        }

#pragma unroll
        for (int i = 0; i < RowsPerThread; ++i) {
          const float a_value = __half2float(shared_a_stage[(row_base + i) * CTA_K + kk]);
#pragma unroll
          for (int j = 0; j < ColsPerThread; ++j) {
            accum[i][j] += a_value * b_values[j];
          }
        }
      }

      __syncthreads();
    }
  }

#pragma unroll
  for (int i = 0; i < RowsPerThread; ++i) {
    const int g_row = tile_m * CTA_M + row_base + i;
    if (g_row >= problem.m) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < ColsPerThread; ++j) {
      const int g_col = tile_n * CTA_N + col_base + j;
      if (g_col >= problem.n) {
        continue;
      }

      const int64_t out_offset = static_cast<int64_t>(g_row) * problem.ldd + g_col;
      float c_value = 0.0f;
      if (problem.c != nullptr && problem.beta != 0.0f) {
        const int64_t c_offset = static_cast<int64_t>(g_row) * problem.ldc + g_col;
        c_value = problem.c[c_offset];
      }
      problem.d[out_offset] = problem.alpha * accum[i][j] + problem.beta * c_value;
    }
  }
}

template <int CTA_M, int CTA_N, int CTA_K>
__device__ void compute_gemm_tile_tensor_core(
    const DeviceProblemDesc& problem,
    int tile_m,
    int tile_n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  static_assert(CTA_M % kMmaM == 0, "CTA_M must be divisible by 16");
  static_assert(CTA_N % kMmaN == 0, "CTA_N must be divisible by 16");
  static_assert(CTA_K % kMmaK == 0, "CTA_K must be divisible by 16");

  constexpr int kWarpCount = kThreadsPerBlock / kWarpSize;
  constexpr int kMmaTilesM = CTA_M / kMmaM;
  constexpr int kMmaTilesN = CTA_N / kMmaN;
  constexpr int kTotalMmaTiles = kMmaTilesM * kMmaTilesN;
  constexpr int kMaxTilesPerWarp = (kTotalMmaTiles + kWarpCount - 1) / kWarpCount;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / kWarpSize;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  constexpr int a_stage_elems = CTA_M * CTA_K;
  constexpr int b_stage_elems = CTA_K * CTA_N;

  __half* shared_a = reinterpret_cast<__half*>(shared_raw);
  __half* shared_b = shared_a + kPipelineStages * a_stage_elems;
  float* shared_c = reinterpret_cast<float*>(shared_b + kPipelineStages * b_stage_elems);

  int tile_indices[kMaxTilesPerWarp];
  int local_tile_count = 0;
  for (int tile_idx = warp_id; tile_idx < kTotalMmaTiles; tile_idx += kWarpCount) {
    tile_indices[local_tile_count++] = tile_idx;
  }

  wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> c_frags[kMaxTilesPerWarp];
#pragma unroll
  for (int i = 0; i < kMaxTilesPerWarp; ++i) {
    if (i < local_tile_count) {
      wmma::fill_fragment(c_frags[i], 0.0f);
    }
  }

  const int k_tiles = ceil_div(problem.k, CTA_K);
  if (k_tiles > 0) {
    preload_a_stage<CTA_M, CTA_K>(problem, tile_m, 0, shared_a + 0 * a_stage_elems);
    preload_b_stage<CTA_N, CTA_K>(problem, tile_n, 0, shared_b + 0 * b_stage_elems);
    cp_async_commit_group();

    for (int kt = 0; kt < k_tiles; ++kt) {
      const int read_stage = kt % kPipelineStages;
      __half* shared_a_stage = shared_a + read_stage * a_stage_elems;
      __half* shared_b_stage = shared_b + read_stage * b_stage_elems;

      cp_async_wait_group0();
      __syncthreads();

      const int next_k = kt + 1;
      if (next_k < k_tiles) {
        const int write_stage = next_k % kPipelineStages;
        preload_a_stage<CTA_M, CTA_K>(
            problem,
            tile_m,
            next_k,
            shared_a + write_stage * a_stage_elems);
        preload_b_stage<CTA_N, CTA_K>(
            problem,
            tile_n,
            next_k,
            shared_b + write_stage * b_stage_elems);
        cp_async_commit_group();
      }

#pragma unroll
      for (int kk = 0; kk < CTA_K; kk += kMmaK) {
#pragma unroll
        for (int frag_idx = 0; frag_idx < kMaxTilesPerWarp; ++frag_idx) {
          if (frag_idx >= local_tile_count) {
            continue;
          }

          const int tile_idx = tile_indices[frag_idx];
          const int local_m = tile_idx / kMmaTilesN;
          const int local_n = tile_idx - local_m * kMmaTilesN;

          const __half* a_ptr = shared_a_stage + (local_m * kMmaM) * CTA_K + kk;
          const __half* b_ptr = shared_b_stage + kk * CTA_N + local_n * kMmaN;


          wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, __half, wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, __half, wmma::row_major> b_frag;

          // 把数据从共享内存搬运到Tensor core中
          wmma::load_matrix_sync(a_frag, a_ptr, CTA_K);
          wmma::load_matrix_sync(b_frag, b_ptr, CTA_N);

          // 计算
          wmma::mma_sync(c_frags[frag_idx], a_frag, b_frag, c_frags[frag_idx]);
        }
      }

      __syncthreads();
    }
  }

#pragma unroll
  // 把数据从Tensor core搬到共享内存
  for (int frag_idx = 0; frag_idx < kMaxTilesPerWarp; ++frag_idx) {
    if (frag_idx >= local_tile_count) {
      continue;
    }
    const int tile_idx = tile_indices[frag_idx];
    const int local_m = tile_idx / kMmaTilesN;
    const int local_n = tile_idx - local_m * kMmaTilesN;

    float* c_ptr = shared_c + (local_m * kMmaM) * CTA_N + local_n * kMmaN;
    wmma::store_matrix_sync(c_ptr, c_frags[frag_idx], CTA_N, wmma::mem_row_major);
  }
  __syncthreads();


  // 把数据从共享内存搬到全局内存
  for (int idx = tid; idx < CTA_M * CTA_N; idx += blockDim.x) {
    const int row = idx / CTA_N;
    const int col = idx - row * CTA_N;
    const int g_row = tile_m * CTA_M + row;
    const int g_col = tile_n * CTA_N + col;

    if (g_row >= problem.m || g_col >= problem.n) {
      continue;
    }

    const int64_t out_offset = static_cast<int64_t>(g_row) * problem.ldd + g_col;
    float c_value = 0.0f;
    if (problem.c != nullptr && problem.beta != 0.0f) {
      const int64_t c_offset = static_cast<int64_t>(g_row) * problem.ldc + g_col;
      c_value = problem.c[c_offset];
    }

    problem.d[out_offset] = problem.alpha * shared_c[idx] + problem.beta * c_value;
  }
#else
  compute_gemm_tile_simt<CTA_M, CTA_N, CTA_K>(problem, tile_m, tile_n);
#endif
}

template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
__device__ void compute_gemm_tile_dispatch(
    const DeviceProblemDesc& problem,
    int tile_m,
    int tile_n) {
  if constexpr (UseTensorCores) {
    compute_gemm_tile_tensor_core<CTA_M, CTA_N, CTA_K>(problem, tile_m, tile_n);
  } else {
    compute_gemm_tile_simt<CTA_M, CTA_N, CTA_K>(problem, tile_m, tile_n);
  }
}

template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
__global__ void grouped_gemm_kernel(
    const DeviceProblemDesc* problems,
    const int* tile_offsets,
    int problem_count,
    int total_tiles,
    int* global_tile_counter) {
  __shared__ int shared_tile_index;

  while (true) {
    if (threadIdx.x == 0) {
      shared_tile_index = atomicAdd(global_tile_counter, 1);
    }
    __syncthreads();

    const int tile_index = shared_tile_index;
    if (tile_index >= total_tiles) {
      return;
    }

    const int problem_index = find_problem_by_tile(tile_offsets, problem_count, tile_index);
    const DeviceProblemDesc problem = problems[problem_index];
    if (problem.tiles_m == 0 || problem.tiles_n == 0) {
      continue;
    }

    const int local_tile = tile_index - tile_offsets[problem_index];
    const int tile_m = local_tile / problem.tiles_n;
    const int tile_n = local_tile - tile_m * problem.tiles_n;

    if (tile_m < problem.tiles_m) {
      compute_gemm_tile_dispatch<CTA_M, CTA_N, CTA_K, UseTensorCores>(problem, tile_m, tile_n);
    }
    __syncthreads();
  }
}

template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
__global__ void single_problem_gemm_kernel(DeviceProblemDesc problem) {
  const int tile_index = static_cast<int>(blockIdx.x);
  const int tile_count = problem.tiles_m * problem.tiles_n;
  if (tile_index >= tile_count) {
    return;
  }

  const int tile_m = tile_index / problem.tiles_n;
  const int tile_n = tile_index - tile_m * problem.tiles_n;
  compute_gemm_tile_dispatch<CTA_M, CTA_N, CTA_K, UseTensorCores>(problem, tile_m, tile_n);
}


template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
cudaError_t configure_kernel_shared_memory() {
  const size_t shared_bytes = dynamic_shared_bytes_for_variant<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  constexpr int kLegacyDynamicSmemLimit = 48 * 1024;
  if (shared_bytes <= static_cast<size_t>(kLegacyDynamicSmemLimit)) {
    return cudaSuccess;
  }

  cudaError_t status = cudaFuncSetAttribute(
      grouped_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));
  if (status != cudaSuccess) {
    return status;
  }

  status = cudaFuncSetAttribute(
      single_problem_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared_bytes));
  if (status != cudaSuccess) {
    return status;
  }

  status = cudaFuncSetAttribute(
      grouped_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  if (status != cudaSuccess) {
    return status;
  }

  return cudaFuncSetAttribute(
      single_problem_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
}
template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
cudaError_t launch_grouped_variant(
    const DeviceProblemDesc* d_problems,
    const int* d_tile_offsets,
    int problem_count,
    int total_tiles,
    int* d_global_counter,
    int grid_ctas,
    cudaStream_t stream) {
  cudaError_t status = configure_kernel_shared_memory<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  if (status != cudaSuccess) {
    return status;
  }

  const size_t shared_bytes = dynamic_shared_bytes_for_variant<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  grouped_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>
      <<<grid_ctas, kThreadsPerBlock, shared_bytes, stream>>>(
          d_problems,
          d_tile_offsets,
          problem_count,
          total_tiles,
          d_global_counter);
  return cudaGetLastError();
}

template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
cudaError_t launch_single_problem_variant(const DeviceProblemDesc& problem, cudaStream_t stream) {
  const int tile_count = problem.tiles_m * problem.tiles_n;
  if (tile_count == 0) {
    return cudaSuccess;
  }

  cudaError_t status = configure_kernel_shared_memory<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  if (status != cudaSuccess) {
    return status;
  }

  const size_t shared_bytes = dynamic_shared_bytes_for_variant<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  single_problem_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>
      <<<tile_count, kThreadsPerBlock, shared_bytes, stream>>>(problem);
  return cudaGetLastError();
}

template <int CTA_M, int CTA_N, int CTA_K, bool UseTensorCores>
cudaError_t query_occupancy_variant(OccupancyInfo* info) {
  cudaError_t status = configure_kernel_shared_memory<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  if (status != cudaSuccess) {
    return status;
  }

  const size_t shared_bytes = dynamic_shared_bytes_for_variant<CTA_M, CTA_N, CTA_K, UseTensorCores>();
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &info->active_blocks_per_sm,
      grouped_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>,
      kThreadsPerBlock,
      shared_bytes);
  if (status != cudaSuccess) {
    return status;
  }

  cudaFuncAttributes attrs{};
  status = cudaFuncGetAttributes(&attrs, grouped_gemm_kernel<CTA_M, CTA_N, CTA_K, UseTensorCores>);
  if (status != cudaSuccess) {
    return status;
  }

  info->dynamic_shared_bytes = static_cast<int>(shared_bytes);
  info->static_shared_bytes = attrs.sharedSizeBytes;
  info->registers_per_thread = attrs.numRegs;
  return cudaSuccess;
}

cudaError_t launch_grouped_by_variant(
    KernelVariant variant,
    bool use_tensor_cores,
    const DeviceProblemDesc* d_problems,
    const int* d_tile_offsets,
    int problem_count,
    int total_tiles,
    int* d_global_counter,
    int grid_ctas,
    cudaStream_t stream) {
  switch (variant) {
    case KernelVariant::k64x64x16:
      return use_tensor_cores
          ? launch_grouped_variant<64, 64, 16, true>(
                d_problems,
                d_tile_offsets,
                problem_count,
                total_tiles,
                d_global_counter,
                grid_ctas,
                stream)
          : launch_grouped_variant<64, 64, 16, false>(
                d_problems,
                d_tile_offsets,
                problem_count,
                total_tiles,
                d_global_counter,
                grid_ctas,
                stream);
    case KernelVariant::k128x64x32:
      return use_tensor_cores
          ? launch_grouped_variant<128, 64, 32, true>(
                d_problems,
                d_tile_offsets,
                problem_count,
                total_tiles,
                d_global_counter,
                grid_ctas,
                stream)
          : launch_grouped_variant<128, 64, 32, false>(
                d_problems,
                d_tile_offsets,
                problem_count,
                total_tiles,
                d_global_counter,
                grid_ctas,
                stream);
    case KernelVariant::k64x128x32:
      return use_tensor_cores
          ? launch_grouped_variant<64, 128, 32, true>(
                d_problems,
                d_tile_offsets,
                problem_count,
                total_tiles,
                d_global_counter,
                grid_ctas,
                stream)
          : launch_grouped_variant<64, 128, 32, false>(
                d_problems,
                d_tile_offsets,
                problem_count,
                total_tiles,
                d_global_counter,
                grid_ctas,
                stream);
    case KernelVariant::kUnsupported:
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t launch_single_by_variant(
    KernelVariant variant,
    bool use_tensor_cores,
    const DeviceProblemDesc& problem,
    cudaStream_t stream) {
  switch (variant) {
    case KernelVariant::k64x64x16:
      return use_tensor_cores ? launch_single_problem_variant<64, 64, 16, true>(problem, stream)
                              : launch_single_problem_variant<64, 64, 16, false>(problem, stream);
    case KernelVariant::k128x64x32:
      return use_tensor_cores ? launch_single_problem_variant<128, 64, 32, true>(problem, stream)
                              : launch_single_problem_variant<128, 64, 32, false>(problem, stream);
    case KernelVariant::k64x128x32:
      return use_tensor_cores ? launch_single_problem_variant<64, 128, 32, true>(problem, stream)
                              : launch_single_problem_variant<64, 128, 32, false>(problem, stream);
    case KernelVariant::kUnsupported:
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t query_occupancy_by_variant(
    KernelVariant variant,
    bool use_tensor_cores,
    OccupancyInfo* info) {
  switch (variant) {
    case KernelVariant::k64x64x16:
      return use_tensor_cores ? query_occupancy_variant<64, 64, 16, true>(info)
                              : query_occupancy_variant<64, 64, 16, false>(info);
    case KernelVariant::k128x64x32:
      return use_tensor_cores ? query_occupancy_variant<128, 64, 32, true>(info)
                              : query_occupancy_variant<128, 64, 32, false>(info);
    case KernelVariant::k64x128x32:
      return use_tensor_cores ? query_occupancy_variant<64, 128, 32, true>(info)
                              : query_occupancy_variant<64, 128, 32, false>(info);
    case KernelVariant::kUnsupported:
    default:
      return cudaErrorInvalidValue;
  }
}

DeviceProblemDesc to_device_problem(const ProblemDesc& p, const KernelConfig& cfg) {
  DeviceProblemDesc out{};
  out.m = p.m;
  out.n = p.n;
  out.k = p.k;
  out.a = p.a;
  out.b = p.b;
  out.c = p.c;
  out.d = p.d;
  out.lda = p.lda;
  out.ldb = p.ldb;
  out.ldc = p.ldc;
  out.ldd = p.ldd;
  out.alpha = p.alpha;
  out.beta = p.beta;
  out.tiles_m = (p.m > 0 && p.n > 0) ? ceil_div(p.m, cfg.cta_m) : 0;
  out.tiles_n = (p.m > 0 && p.n > 0) ? ceil_div(p.n, cfg.cta_n) : 0;
  return out;
}

void release_device_buffers(
    DeviceProblemDesc*& d_problems,
    int*& d_tile_offsets,
    int*& d_counter) {
  if (d_problems != nullptr) {
    cudaFree(d_problems);
    d_problems = nullptr;
  }
  if (d_tile_offsets != nullptr) {
    cudaFree(d_tile_offsets);
    d_tile_offsets = nullptr;
  }
  if (d_counter != nullptr) {
    cudaFree(d_counter);
    d_counter = nullptr;
  }
}

}  // namespace

const char* sort_mode_name(SortMode sort_mode) {
  switch (sort_mode) {
    case SortMode::kNone:
      return "none";
    case SortMode::kByK:
      return "sort_by_k";
    case SortMode::kByTiles:
      return "sort_by_tiles";
    case SortMode::kByTilesThenK:
      return "sort_by_tiles_then_k";
    default:
      return "unknown";
  }
}

PreparedProblemSet prepare_problem_set(
    const std::vector<ProblemDesc>& problems,
    const KernelConfig& config,
    SortMode sort_mode) {
  PreparedProblemSet prepared;

  const int problem_count = static_cast<int>(problems.size());
  prepared.sorted_to_original.resize(problem_count);
  std::iota(prepared.sorted_to_original.begin(), prepared.sorted_to_original.end(), 0);

  auto compare_k = [&problems](int lhs, int rhs) {
    if (problems[lhs].k != problems[rhs].k) {
      return problems[lhs].k > problems[rhs].k;
    }
    const int64_t lhs_area = static_cast<int64_t>(problems[lhs].m) * problems[lhs].n;
    const int64_t rhs_area = static_cast<int64_t>(problems[rhs].m) * problems[rhs].n;
    return lhs_area > rhs_area;
  };

  auto compare_tiles = [&problems, &config, &compare_k](int lhs, int rhs) {
    const int lhs_tiles = tile_count_for_problem(problems[lhs], config);
    const int rhs_tiles = tile_count_for_problem(problems[rhs], config);
    if (lhs_tiles != rhs_tiles) {
      return lhs_tiles > rhs_tiles;
    }
    return compare_k(lhs, rhs);
  };

  if (sort_mode == SortMode::kByK) {
    std::stable_sort(
        prepared.sorted_to_original.begin(),
        prepared.sorted_to_original.end(),
        compare_k);
  } else if (sort_mode == SortMode::kByTiles || sort_mode == SortMode::kByTilesThenK) {
    std::stable_sort(
        prepared.sorted_to_original.begin(),
        prepared.sorted_to_original.end(),
        compare_tiles);
  }

  prepared.sorted_problems.reserve(problem_count);
  prepared.tile_offsets.reserve(problem_count + 1);
  prepared.tile_offsets.push_back(0);

  for (int sorted_index = 0; sorted_index < problem_count; ++sorted_index) {
    const int original_index = prepared.sorted_to_original[sorted_index];
    prepared.sorted_problems.push_back(problems[original_index]);

    const int tile_count = tile_count_for_problem(problems[original_index], config);
    const int next_offset = prepared.tile_offsets.back() + tile_count;
    prepared.tile_offsets.push_back(next_offset);
  }

  prepared.total_tiles = prepared.tile_offsets.back();
  return prepared;
}

cudaError_t query_occupancy(const KernelConfig& config, OccupancyInfo* info) {
  if (info == nullptr) {
    return cudaErrorInvalidValue;
  }

  KernelVariant variant = KernelVariant::kUnsupported;
  bool use_tensor_cores = false;
  cudaDeviceProp prop{};
  cudaError_t status = resolve_runtime_mode(config, &variant, &use_tensor_cores, &prop);
  if (status != cudaSuccess) {
    return status;
  }

  info->sm_count = prop.multiProcessorCount;
  info->max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
  info->active_blocks_per_sm = 0;

  info->threads_per_block = kThreadsPerBlock;
  info->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  info->warps_per_block = kThreadsPerBlock / prop.warpSize;
  info->max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
  info->active_warps_per_sm = 0;

  info->max_registers_per_sm = prop.regsPerMultiprocessor;
  info->max_registers_per_block = prop.regsPerBlock;

  info->estimated_registers_per_thread = estimated_registers_per_thread(variant, use_tensor_cores);
  info->estimated_registers_per_block = info->estimated_registers_per_thread * kThreadsPerBlock;

  info->registers_per_thread = 0;
  info->registers_per_block = 0;
  info->register_limited_blocks_per_sm = 0;

  info->static_shared_bytes = 0;
  info->dynamic_shared_bytes = static_cast<int>(shared_bytes_for(variant, use_tensor_cores));

  info->theoretical_occupancy = 0.0f;
  info->using_tensor_cores = use_tensor_cores;

  status = query_occupancy_by_variant(variant, use_tensor_cores, info);
  if (status != cudaSuccess) {
    return status;
  }

  info->registers_per_block = info->registers_per_thread * info->threads_per_block;

  if (info->registers_per_block > 0) {
    const int register_limit = info->max_registers_per_sm / info->registers_per_block;
    info->register_limited_blocks_per_sm = std::min(info->max_blocks_per_sm, register_limit);
  } else {
    info->register_limited_blocks_per_sm = info->max_blocks_per_sm;
  }

  info->active_warps_per_sm = info->active_blocks_per_sm * info->warps_per_block;
  if (info->max_warps_per_sm > 0) {
    info->theoretical_occupancy = static_cast<float>(info->active_warps_per_sm) /
                                  static_cast<float>(info->max_warps_per_sm);
  }
  return cudaSuccess;
}

struct GroupedGemmExecutor::Impl {
  DeviceProblemDesc* d_problems = nullptr;
  int* d_tile_offsets = nullptr;
  int* d_global_counter = nullptr;

  int problem_count = 0;
  int total_tiles = 0;
  int grid_ctas = 0;

  KernelConfig config{};
  KernelVariant variant = KernelVariant::kUnsupported;
  bool use_tensor_cores = false;
};

GroupedGemmExecutor::GroupedGemmExecutor() : impl_(new Impl()) {}

GroupedGemmExecutor::~GroupedGemmExecutor() {
  if (impl_ != nullptr) {
    release_device_buffers(
        impl_->d_problems,
        impl_->d_tile_offsets,
        impl_->d_global_counter);
    delete impl_;
    impl_ = nullptr;
  }
}

GroupedGemmExecutor::GroupedGemmExecutor(GroupedGemmExecutor&& other) noexcept
    : impl_(other.impl_) {
  other.impl_ = nullptr;
}

GroupedGemmExecutor& GroupedGemmExecutor::operator=(GroupedGemmExecutor&& other) noexcept {
  if (this != &other) {
    if (impl_ != nullptr) {
      release_device_buffers(
          impl_->d_problems,
          impl_->d_tile_offsets,
          impl_->d_global_counter);
      delete impl_;
    }
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

cudaError_t GroupedGemmExecutor::build(
    const PreparedProblemSet& prepared,
    const KernelConfig& config) {
  if (impl_ == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  KernelVariant variant = KernelVariant::kUnsupported;
  bool use_tensor_cores = false;
  cudaError_t status = resolve_runtime_mode(config, &variant, &use_tensor_cores, nullptr);
  if (status != cudaSuccess) {
    return status;
  }

  const int problem_count = static_cast<int>(prepared.sorted_problems.size());
  if (static_cast<int>(prepared.tile_offsets.size()) != problem_count + 1) {
    return cudaErrorInvalidValue;
  }
  if (!prepared.tile_offsets.empty() && prepared.tile_offsets.back() != prepared.total_tiles) {
    return cudaErrorInvalidValue;
  }

  release_device_buffers(
      impl_->d_problems,
      impl_->d_tile_offsets,
      impl_->d_global_counter);

  impl_->problem_count = problem_count;
  impl_->total_tiles = prepared.total_tiles;
  impl_->grid_ctas = 0;
  impl_->config = config;
  impl_->variant = variant;
  impl_->use_tensor_cores = use_tensor_cores;

  std::vector<DeviceProblemDesc> device_problems;
  device_problems.reserve(problem_count);
  for (const ProblemDesc& p : prepared.sorted_problems) {
    if (p.m < 0 || p.n < 0 || p.k < 0) {
      return cudaErrorInvalidValue;
    }
    if (p.a == nullptr || p.b == nullptr || p.d == nullptr) {
      return cudaErrorInvalidDevicePointer;
    }
    device_problems.push_back(to_device_problem(p, config));
  }

  if (problem_count > 0) {
    status = cudaMalloc(
        reinterpret_cast<void**>(&impl_->d_problems),
        problem_count * sizeof(DeviceProblemDesc));
    if (status != cudaSuccess) {
      return status;
    }

    status = cudaMemcpy(
        impl_->d_problems,
        device_problems.data(),
        problem_count * sizeof(DeviceProblemDesc),
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      return status;
    }

    status = cudaMalloc(
        reinterpret_cast<void**>(&impl_->d_tile_offsets),
        (problem_count + 1) * sizeof(int));
    if (status != cudaSuccess) {
      return status;
    }

    status = cudaMemcpy(
        impl_->d_tile_offsets,
        prepared.tile_offsets.data(),
        (problem_count + 1) * sizeof(int),
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      return status;
    }
  }

  if (impl_->total_tiles > 0) {
    status = cudaMalloc(
        reinterpret_cast<void**>(&impl_->d_global_counter),
        sizeof(int));
    if (status != cudaSuccess) {
      return status;
    }
  }

  OccupancyInfo occupancy{};
  status = query_occupancy(config, &occupancy);
  if (status != cudaSuccess) {
    return status;
  }
  impl_->use_tensor_cores = occupancy.using_tensor_cores;

  if (impl_->total_tiles > 0) {
    if (config.persistent_ctas > 0) {
      impl_->grid_ctas = config.persistent_ctas;
    } else {
      const int max_resident_ctas = occupancy.sm_count * std::max(1, occupancy.active_blocks_per_sm);
      impl_->grid_ctas = std::min(max_resident_ctas, std::max(impl_->total_tiles, occupancy.sm_count));
    }
    impl_->grid_ctas = std::max(1, impl_->grid_ctas);
  }

  return cudaSuccess;
}

cudaError_t GroupedGemmExecutor::run(cudaStream_t stream) const {
  if (impl_ == nullptr) {
    return cudaErrorInvalidDevicePointer;
  }

  if (impl_->total_tiles == 0 || impl_->problem_count == 0) {
    return cudaSuccess;
  }

  cudaError_t status = cudaMemsetAsync(impl_->d_global_counter, 0, sizeof(int), stream);
  if (status != cudaSuccess) {
    return status;
  }

  return launch_grouped_by_variant(
      impl_->variant,
      impl_->use_tensor_cores,
      impl_->d_problems,
      impl_->d_tile_offsets,
      impl_->problem_count,
      impl_->total_tiles,
      impl_->d_global_counter,
      impl_->grid_ctas,
      stream);
}

int GroupedGemmExecutor::problem_count() const {
  return impl_ != nullptr ? impl_->problem_count : 0;
}

int GroupedGemmExecutor::total_tiles() const {
  return impl_ != nullptr ? impl_->total_tiles : 0;
}

int GroupedGemmExecutor::grid_ctas() const {
  return impl_ != nullptr ? impl_->grid_ctas : 0;
}

cudaError_t run_grouped_gemm_once(
    const std::vector<ProblemDesc>& problems,
    SortMode sort_mode,
    const KernelConfig& config,
    cudaStream_t stream) {
  const PreparedProblemSet prepared = prepare_problem_set(problems, config, sort_mode);
  GroupedGemmExecutor executor;
  cudaError_t status = executor.build(prepared, config);
  if (status != cudaSuccess) {
    return status;
  }
  return executor.run(stream);
}

cudaError_t run_baseline_per_problem(
    const std::vector<ProblemDesc>& problems,
    const KernelConfig& config,
    cudaStream_t stream) {
  KernelVariant variant = KernelVariant::kUnsupported;
  bool use_tensor_cores = false;
  cudaError_t status = resolve_runtime_mode(config, &variant, &use_tensor_cores, nullptr);
  if (status != cudaSuccess) {
    return status;
  }

  for (const ProblemDesc& p : problems) {
    if (p.a == nullptr || p.b == nullptr || p.d == nullptr) {
      return cudaErrorInvalidDevicePointer;
    }

    DeviceProblemDesc device_problem = to_device_problem(p, config);
    status = launch_single_by_variant(variant, use_tensor_cores, device_problem, stream);
    if (status != cudaSuccess) {
      return status;
    }
  }

  return cudaSuccess;
}

int count_non_empty_problems(
    const std::vector<ProblemDesc>& problems,
    const KernelConfig& config) {
  int count = 0;
  for (const ProblemDesc& p : problems) {
    if (tile_count_for_problem(p, config) > 0) {
      ++count;
    }
  }
  return count;
}

}  // namespace grouped_gemm
