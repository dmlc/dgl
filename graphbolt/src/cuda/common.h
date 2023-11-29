/**
 *  Copyright (c) 2017-2023 by Contributors
 * @file cuda/common.h
 * @brief Common utilities for CUDA
 */
#ifndef GRAPHBOLT_CUDA_COMMON_H_
#define GRAPHBOLT_CUDA_COMMON_H_

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <memory>
#include <unordered_map>

namespace graphbolt {
namespace cuda {

/*
How to use this class to get a nonblocking thrust execution policy that uses
torch's memory pool and the current cuda stream:

cuda::CUDAWorkspaceAllocator allocator;
const auto stream = torch::cuda::getDefaultCUDAStream();
const auto exec_policy = thrust::cuda::par_nosync(allocator).on(stream);

Now, one can pass exec_policy to thrust functions

To get an integer array of size 1000 whose lifetime is managed by unique_ptr,
use: auto int_array = allocator.alloc_unique<int>(1000); int_array.get() gives
the raw pointer.
*/
class CUDAWorkspaceAllocator {
  using ptr_map_t = std::unordered_map<void*, torch::Tensor>;
  std::shared_ptr<ptr_map_t> pointers;

 public:
  using value_type = char;

  void operator()(void* ptr) const { pointers->erase(ptr); }

  explicit CUDAWorkspaceAllocator() : pointers(std::make_shared<ptr_map_t>()) {}

  CUDAWorkspaceAllocator& operator=(const CUDAWorkspaceAllocator&) = default;

  template <typename T>
  std::unique_ptr<T, CUDAWorkspaceAllocator> AllocateStorage(
      std::size_t size) const {
    auto t = torch::empty(
        sizeof(T) * size, torch::TensorOptions()
                              .dtype(torch::kByte)
                              .device(c10::DeviceType::CUDA));
    pointers->operator[](t.data_ptr()) = t;
    return std::unique_ptr<T, CUDAWorkspaceAllocator>(
        reinterpret_cast<T*>(t.data_ptr()), *this);
  }

  char* allocate(std::ptrdiff_t size) const {
    auto t = torch::empty(
        size, torch::TensorOptions()
                  .dtype(torch::kByte)
                  .device(c10::DeviceType::CUDA));
    pointers->operator[](t.data_ptr()) = t;
    return reinterpret_cast<char*>(t.data_ptr());
  }

  void deallocate(char* ptr, std::size_t) const { pointers->erase(ptr); }
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
