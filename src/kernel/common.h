#ifndef DGL_KERNEL_COMMON_H_
#define DGL_KERNEL_COMMON_H_

namespace dgl {
namespace kernel {

#ifdef __CUDACC__
#define DGLDEVICE __device__
#define DGLINLINE __forceinline__
#else
#define DGLDEVICE
#define DGLINLINE __inline__
#endif  // __CUDACC__

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_COMMON_H_
