
#include <torch/script.h>

namespace dgl {
namespace graphbolt {

__global__ void _AddOneKernel(int64_t nnz, float* value) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nnz) {
    value[i] += 1;
  }
}

template <typename c10::DeviceType XPU>
torch::Tensor AddOneImpl(torch::Tensor tensor) {
  int64_t nnz = tensor.numel();
  int n_threads_per_block = 1024;
  int n_blocks = (nnz + n_threads_per_block - 1) / n_threads_per_block;
  _AddOneKernel<<<n_blocks, n_threads_per_block>>>(
      nnz, tensor.data_ptr<float>());
  return tensor;
}

template torch::Tensor AddOneImpl<c10::DeviceType::CUDA>(torch::Tensor tensor);

}  // namespace graphbolt
}  // namespace dgl