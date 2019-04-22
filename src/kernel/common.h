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

#define DGL_XPU_SWITCH(val, XPU, ...)  \
  if (val == kDLCPU) {                 \
    const int XPU = kDLCPU;            \
    {__VA_ARGS__}                      \
  } else if (val == kDLGPU) {          \
    const int XPU = kDLGPU;            \
    {__VA_ARGS__}                      \
  } else {                             \
    LOG(FATAL) << "Unsupported device type: " << val;  \
  }

#define DGL_DTYPE_SWITCH(val, DType, ...)                   \
  if (val.code == kDLFloat && val.bits == 32) {             \
    typedef float DType;                                    \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLFloat && val.bits == 64) {      \
    typedef double DType;                                   \
    {__VA_ARGS__}                                           \
  } else {                                                  \
    LOG(FATAL) << "Unsupported dtype: " << val.code << "_"  \
               << val.bits;                                 \
  }


#if 0
#define DGL_DTYPE_SWITCH(val, DType, ...)                   \
  if (val.code == kDLInt && val.bits == 32) {               \
    typedef int32_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLInt && val.bits == 64) {        \
    typedef int64_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLUInt && val.bits == 32) {       \
    typedef uint32_t DType;                                 \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLUInt && val.bits == 64) {       \
    typedef uint64_t DType;                                 \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLFloat && val.bits == 32) {      \
    typedef float DType;                                    \
    {__VA_ARGS__}                                           \
  } else if (val.code == kDLFloat && val.bits == 64) {      \
    typedef double DType;                                   \
    {__VA_ARGS__}                                           \
  } else {                                                  \
    LOG(FATAL) << "Unsupported dtype: " << val.code << "_"  \
               << val.bits;                                 \
  }
#endif


}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_COMMON_H_
