/**
 *  Copyright (c) 2020-2022 by Contributors
 * @file torch/torch.cpp
 * @brief Implementation of PyTorch adapter library.
 */

#include <c10/core/CPUAllocator.h>
#include <tensoradapter_exports.h>
#ifdef DGL_USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif  // DGL_USE_CUDA

namespace tensoradapter {

extern "C" {

TA_EXPORTS void* CPURawAlloc(size_t nbytes) {
  return c10::GetCPUAllocator()->raw_allocate(nbytes);
}

TA_EXPORTS void CPURawDelete(void* ptr) {
  c10::GetCPUAllocator()->raw_deallocate(ptr);
}

#ifdef DGL_USE_CUDA
TA_EXPORTS void* CUDARawAlloc(size_t nbytes, cudaStream_t stream) {
  at::globalContext().lazyInitCUDA();
  return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(nbytes, stream);
}

TA_EXPORTS void CUDARawDelete(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

TA_EXPORTS cudaStream_t CUDACurrentStream() {
  return at::cuda::getCurrentCUDAStream();
}

TA_EXPORTS void RecordStream(void* ptr, cudaStream_t stream, int device_id) {
  c10::DataPtr data_ptr{
      ptr, ptr, c10::cuda::CUDACachingAllocator::get()->raw_deleter(),
      c10::Device(c10::DeviceType::CUDA, device_id)};
  c10::cuda::CUDACachingAllocator::recordStream(
      data_ptr,
      // getStreamFromExternal doesn't exist before PyTorch 1.10, just copy it
      // here
      c10::cuda::CUDAStream(
          c10::cuda::CUDAStream::UNCHECKED,
          c10::Stream(
              c10::Stream::UNSAFE,
              c10::Device(c10::DeviceType::CUDA, device_id),
              reinterpret_cast<int64_t>(stream))));
  data_ptr.release_context();
}

class CUDAHostDeleter {
 public:
  explicit CUDAHostDeleter(std::unique_ptr<void, c10::DeleterFnPtr> ptr)
      : ptr_(std::move(ptr)) {}

 private:
  std::unique_ptr<void, c10::DeleterFnPtr> ptr_;
};

TA_EXPORTS void* CUDARawHostAlloc(
    size_t nbytes, void** ctx, void** raw_deleter) {
  auto data_ptr = at::cuda::getCachingHostAllocator()->allocate(nbytes);
  auto raw = data_ptr.get();
  // Return the raw ctx ptr for recording event.
  *ctx = data_ptr.get_context();

  // Transfer ownership to raw_deleter.
  auto* data_deleter = new CUDAHostDeleter(data_ptr.move_context());
  *raw_deleter = static_cast<void*>(data_deleter);
  return raw;
}

// Designated CUDAHostDeleter for CUDARawHostAlloc.
TA_EXPORTS void CUDARawHostDelete(void** raw_deleter) {
  delete static_cast<CUDAHostDeleter*>(*raw_deleter);
  *raw_deleter = nullptr;
}

TA_EXPORTS void CUDARecordHostAlloc(
    void* ptr, void* ctx, cudaStream_t stream, int device_id) {
  at::cuda::CachingHostAllocator_recordEvent(
      ptr, ctx,
      c10::cuda::CUDAStream(
          c10::cuda::CUDAStream::UNCHECKED,
          c10::Stream(
              c10::Stream::UNSAFE,
              c10::Device(c10::DeviceType::CUDA, device_id),
              reinterpret_cast<int64_t>(stream))));
}

TA_EXPORTS void CUDAHostAllocatorEmptyCache() {
  at::cuda::CachingHostAllocator_emptyCache();
}
#endif  // DGL_USE_CUDA
};

};  // namespace tensoradapter
