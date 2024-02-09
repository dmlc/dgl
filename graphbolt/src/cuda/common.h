/**
 *  Copyright (c) 2017-2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/common.h
 * @brief Common utilities for CUDA
 */
#ifndef GRAPHBOLT_CUDA_COMMON_H_
#define GRAPHBOLT_CUDA_COMMON_H_

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/script.h>

#include <memory>
#include <unordered_map>

namespace graphbolt {
namespace cuda {

/**
 * @brief This class is designed to allocate workspace storage
 * and to get a nonblocking thrust execution policy
 * that uses torch's CUDA memory pool and the current cuda stream:
 *
 * cuda::CUDAWorkspaceAllocator allocator;
 * const auto stream = torch::cuda::getDefaultCUDAStream();
 * const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);
 *
 * Now, one can pass exec_policy to thrust functions
 *
 * To get an integer array of size 1000 whose lifetime is managed by unique_ptr,
 * use:
 *
 * auto int_array = allocator.AllocateStorage<int>(1000);
 *
 * int_array.get() gives the raw pointer.
 */
struct CUDAWorkspaceAllocator {
  // Required by thrust to satisfy allocator requirements.
  using value_type = char;

  explicit CUDAWorkspaceAllocator() { at::globalContext().lazyInitCUDA(); }

  CUDAWorkspaceAllocator& operator=(const CUDAWorkspaceAllocator&) = default;

  void operator()(void* ptr) const {
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  }

  // Required by thrust to satisfy allocator requirements.
  value_type* allocate(std::ptrdiff_t size) const {
    return reinterpret_cast<value_type*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(size));
  }

  // Required by thrust to satisfy allocator requirements.
  void deallocate(value_type* ptr, std::size_t) const { operator()(ptr); }

  template <typename T>
  std::unique_ptr<T, CUDAWorkspaceAllocator> AllocateStorage(
      std::size_t size) const {
    return std::unique_ptr<T, CUDAWorkspaceAllocator>(
        reinterpret_cast<T*>(allocate(sizeof(T) * size)), *this);
  }
};

inline auto GetAllocator() { return CUDAWorkspaceAllocator{}; }

inline auto GetCurrentStream() { return c10::cuda::getCurrentCUDAStream(); }

template <typename T>
inline bool is_zero(T size) {
  return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
  return size.x == 0 || size.y == 0 || size.z == 0;
}

#define CUDA_CALL(func) C10_CUDA_CHECK((func))

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, ...)          \
  {                                                                 \
    if (!graphbolt::cuda::is_zero((nblks)) &&                       \
        !graphbolt::cuda::is_zero((nthrs))) {                       \
      auto stream = graphbolt::cuda::GetCurrentStream();            \
      (kernel)<<<(nblks), (nthrs), (shmem), stream>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                               \
    }                                                               \
  }

#define CUB_CALL(fn, ...)                                                     \
  {                                                                           \
    auto allocator = graphbolt::cuda::GetAllocator();                         \
    auto stream = graphbolt::cuda::GetCurrentStream();                        \
    size_t workspace_size = 0;                                                \
    CUDA_CALL(cub::fn(nullptr, workspace_size, __VA_ARGS__, stream));         \
    auto workspace = allocator.AllocateStorage<char>(workspace_size);         \
    CUDA_CALL(cub::fn(workspace.get(), workspace_size, __VA_ARGS__, stream)); \
  }

#define THRUST_CALL(fn, ...)                                                 \
  [&] {                                                                      \
    auto allocator = graphbolt::cuda::GetAllocator();                        \
    auto stream = graphbolt::cuda::GetCurrentStream();                       \
    const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream); \
    return thrust::fn(exec_policy, __VA_ARGS__);                             \
  }()

/**
 * @brief This class is designed to handle the copy operation of a single
 * scalar_t item from a given CUDA device pointer. Later, if the object is cast
 * into scalar_t, the value can be read.
 *
 * auto num_edges = cuda::CopyScalar(indptr.data_ptr<scalar_t>() +
 *     indptr.size(0) - 1);
 * // Perform many operations here, they will run as normal.
 * // We finally need to read num_edges.
 * auto indices = torch::empty(static_cast<scalar_t>(num_edges));
 */
template <typename scalar_t>
struct CopyScalar {
  CopyScalar() : is_ready_(true) { init_pinned_storage(); }

  void record(at::cuda::CUDAStream stream = GetCurrentStream()) {
    copy_event_.record(stream);
    is_ready_ = false;
  }

  scalar_t* get() {
    return reinterpret_cast<scalar_t*>(pinned_scalar_.data_ptr());
  }

  CopyScalar(const scalar_t* device_ptr) {
    init_pinned_storage();
    auto stream = GetCurrentStream();
    CUDA_CALL(cudaMemcpyAsync(
        reinterpret_cast<scalar_t*>(pinned_scalar_.data_ptr()), device_ptr,
        sizeof(scalar_t), cudaMemcpyDeviceToHost, stream));
    record(stream);
  }

  operator scalar_t() {
    if (!is_ready_) {
      copy_event_.synchronize();
      is_ready_ = true;
    }
    return *get();
  }

 private:
  void init_pinned_storage() {
    pinned_scalar_ = torch::empty(
        sizeof(scalar_t),
        c10::TensorOptions().dtype(torch::kBool).pinned_memory(true));
  }

  torch::Tensor pinned_scalar_;
  at::cuda::CUDAEvent copy_event_;
  bool is_ready_;
};

// This includes all integer, float and boolean types.
#define GRAPHBOLT_DISPATCH_CASE_ALL_TYPES(...)            \
  AT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Bool, __VA_ARGS__)

#define GRAPHBOLT_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, GRAPHBOLT_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

#define GRAPHBOLT_DISPATCH_ELEMENT_SIZES(element_size, name, ...)             \
  [&] {                                                                       \
    switch (element_size) {                                                   \
      case 1: {                                                               \
        using element_size_t = uint8_t;                                       \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 2: {                                                               \
        using element_size_t = uint16_t;                                      \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 4: {                                                               \
        using element_size_t = uint32_t;                                      \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 8: {                                                               \
        using element_size_t = uint64_t;                                      \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case 16: {                                                              \
        using element_size_t = float4;                                        \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default:                                                                \
        TORCH_CHECK(false, name, " with the element_size is not supported!"); \
        using element_size_t = uint8_t;                                       \
        return __VA_ARGS__();                                                 \
    }                                                                         \
  }()

}  // namespace cuda
}  // namespace graphbolt
#endif  // GRAPHBOLT_CUDA_COMMON_H_
