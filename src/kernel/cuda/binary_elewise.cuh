#ifndef DGL_KERNEL_CUDA_BINARY_ELEWISE_CUH_
#define DGL_KERNEL_CUDA_BINARY_ELEWISE_CUH_

#include "../functor.h"
#include "./functor.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

template <typename DType>
struct GData {
  // length along x dimension
  int64_t x_length;
  // input data
  DType* src_data, edge_data, dst_data;
  // shuffle edge ids
  int64_t* edge_ids;
  // output data
  DType* out_data;
};

template <typename DType,
          typename Functors>
struct BinaryElewise {
  static __device__ __forceinline__ bool CondEdge(
      int64_t src, int64_t dst, int64_t eid, GData<DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int64_t src, int64_t dst, int64_t eid, GData<DType>* gdata) {
    const int64_t D = gdata->x_length;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    eid = Functors::GetEid(eid, gdata->edge_ids);
    DType* srcoff = gdata->src_data + src * D;
    DType* edgeoff = gdata->edge_data + eid * D;
    DType* dstoff = gdata->dst_data + dst * D;
    DType* outoff = gdata->out_data + Functors::SelectOut(src, eid, dst);
    while (tx < D) {
      DType v1 = Functors::InputLeft(srcoff + tx, edge_off + tx, dstoff + tx);
      DType v2 = Functors::InputRight(srcoff + tx, edge_off + tx, dstoff + tx);
      DType out = Functors::Call(v1, v2);
      Functors::Output(outoff + tx, out);
      tx += stride_x;
    }
  }
}

// functors
template <typename DType, typename EidGetter, typename Selector,
          typename LeftReader, typename RightReader,
          typename BinaryOp, typename OutputWriter>
struct FunctorsTempl {
  static __device__ __forceinline__ int64_t GetEid(int64_t eid, int64_t* id_map) {
    return EidGetter::Call(eid, id_map);
  }
  static __device__ __forceinline__ int64_t SelectOut(
      int64_t src, int64_t eid, int64_t dst) {
    return Selector::Call(src, eid, dst);
  }
  static __device__ __forceinline__ int64_t InputLeft(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return LeftReader::Call(src_data, eid_data, dst_data);
  }
  static __device__ __forceinline__ int64_t InputRight(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return RightReader::Call(src_data, eid_data, dst_data);
  }
  static __device__ __forceinline__ DType Call(DType lhs, DType rhs) {
    return BinaryOp::Call(lhs, rhs);
  }
  static __device__ __forceinline__ void Output(DType* addr, DType val) {
    OutputWriter::Call(addr, val);
  }
};

// common aliasing
template <typename DType, typename EidGetter,
          typename BinaryOp, typename OutputWriter>
using SrcOpEdgeReduce = FunctorsTempl<DType, EidGetter, SelectDst<int64_t>,
                                      ReadSrc<DType, LDGReader>, ReadEdge<DType, LDGReader>,
                                      BinaryOp, OutputWriter>;

template <typename DType, typename EidGetter, typename BinaryOp>
using SrcOpEdgeToEdge = FunctorsTempl<DType, EidGetter, SelectEdge<int64_t>,
                                      ReadSrc<DType, LDGReader>, ReadDst<DType, LDGReader>,
                                      BinaryOp, StoreWriter<DType>>;

template <typename DType, typename EidGetter,
          typename BinaryOp, typename OutputWriter>
using SrcOpDstReduce = FunctorsTempl<DType, EidGetter, SelectDst<int64_t>,
                                     ReadSrc<DType, LDGReader>, ReadDst<DType, LDGReader>,
                                     BinaryOp, OutputWriter>;

template <typename DType, typename EidGetter, typename BinaryOp>
using SrcOpDstToEdge = FunctorsTempl<DType, EidGetter, SelectEdge<int64_t>,
                                     ReadSrc<DType, LDGReader>, ReadDst<DType, LDGReader>,
                                     BinaryOp, StoreWriter<DType>>;

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BINARY_ELEWISE_CUH_
