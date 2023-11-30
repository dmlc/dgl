/**
 *  Copyright (c) 2017-2023 by Contributors
 * @file cuda/common.h
 * @brief Common utilities for CUDA
 */
#ifndef GRAPHBOLT_CUDA_COMMON_H_
#define GRAPHBOLT_CUDA_COMMON_H_

#include <c10/cuda/CUDAException.h>
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

 * cuda::CUDAWorkspaceAllocator allocator;
 * const auto stream = torch::cuda::getDefaultCUDAStream();
 * const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

 * Now, one can pass exec_policy to thrust functions

 * To get an integer array of size 1000 whose lifetime is managed by unique_ptr,
 * use:

 * auto int_array = allocator.AllocateStorage<int>(1000);

 * int_array.get() gives the raw pointer.
 */
class CUDAWorkspaceAllocator {
  using TensorPtrMapType = std::unordered_map<void*, torch::Tensor>;
  std::shared_ptr<TensorPtrMapType> ptr_map_;

 public:
  // Required by thrust to satisfy allocator requirements.
  using value_type = char;

  explicit CUDAWorkspaceAllocator()
      : ptr_map_(std::make_shared<TensorPtrMapType>()) {}

  CUDAWorkspaceAllocator& operator=(const CUDAWorkspaceAllocator&) = default;

  void operator()(void* ptr) const { ptr_map_->erase(ptr); }

  template <typename T>
  std::unique_ptr<T, CUDAWorkspaceAllocator> AllocateStorage(
      std::size_t size) const {
    auto tensor = torch::empty(
        sizeof(T) * size, torch::TensorOptions()
                              .dtype(torch::kByte)
                              .device(c10::DeviceType::CUDA));
    ptr_map_->operator[](tensor.data_ptr()) = tensor;
    return std::unique_ptr<T, CUDAWorkspaceAllocator>(
        reinterpret_cast<T*>(tensor.data_ptr()), *this);
  }

  // Required by thrust to satisfy allocator requirements.
  value_type* allocate(std::ptrdiff_t size) const {
    auto tensor = torch::empty(
        size, torch::TensorOptions()
                  .dtype(torch::kByte)
                  .device(c10::DeviceType::CUDA));
    ptr_map_->operator[](tensor.data_ptr()) = tensor;
    return reinterpret_cast<value_type*>(tensor.data_ptr());
  }

  // Required by thrust to satisfy allocator requirements.
  void deallocate(value_type* ptr, std::size_t) const { operator()(ptr); }
};

template <typename T>
inline bool is_zero(T size) {
  return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
  return size.x == 0 || size.y == 0 || size.z == 0;
}

#define CUDA_CALL(func) C10_CUDA_CHECK((func))

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...)    \
  {                                                                   \
    if (!graphbolt::cuda::is_zero((nblks)) &&                         \
        !graphbolt::cuda::is_zero((nthrs))) {                         \
      (kernel)<<<(nblks), (nthrs), (shmem), (stream)>>>(__VA_ARGS__); \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
    }                                                                 \
  }

}  // namespace cuda
}  // namespace graphbolt
#endif  // GRAPHBOLT_CUDA_COMMON_H_
