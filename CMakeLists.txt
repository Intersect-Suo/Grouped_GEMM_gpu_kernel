cmake_minimum_required(VERSION 3.22)
project(grouped_gemm_gpu_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_library(grouped_gemm STATIC
  src/grouped_gemm.cu
)

target_include_directories(grouped_gemm
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(grouped_gemm
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

set_target_properties(grouped_gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

add_executable(test_grouped_gemm
  tests/test_correctness.cu
)

target_link_libraries(test_grouped_gemm
  PRIVATE
    grouped_gemm
)

set_target_properties(test_grouped_gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

add_executable(bench_grouped_gemm
  bench/benchmark_grouped_gemm.cu
)

target_link_libraries(bench_grouped_gemm
  PRIVATE
    grouped_gemm
)

set_target_properties(bench_grouped_gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)
