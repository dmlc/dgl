#ifndef DGL_KERNEL_CUDA_UTILS_CUH_
#define DGL_KERNEL_CUDA_UTILS_CUH_

namespace dgl {
namespace kernel {
namespace cuda {
namespace utils {
// Find the number of threads that is:
//  - power of two
//  - smaller or equal to dim
//  - smaller or equal to max_nthrs
inline int FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

template <typename DType>
__global__ void _FillKernel(DType* ptr, size_t length, DType val) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    ptr[tx] = val;
    tx += stride_x;
  }
}

template <typename DType>
void Fill(cudaStream_t stream, DType* ptr, size_t length, DType val) {
  int nt = FindNumThreads(length, 1024);
  int nb = (length + nt - 1) / nt;
  _FillKernel<<<nb, nt, 0, stream>>>(ptr, length, val);
}

}  // namespace utils
}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_UTILS_CUH_
