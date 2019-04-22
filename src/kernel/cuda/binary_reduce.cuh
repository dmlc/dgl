#ifndef DGL_KERNEL_CUDA_BINARY_REDUCE_CUH_
#define DGL_KERNEL_CUDA_BINARY_REDUCE_CUH_

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
    DType* lhs = Functors::SelectLeft(srcoff, edgeoff, dstoff);
    DType* rhs = Functors::SelectLeft(srcoff, edgeoff, dstoff);
    DType* outoff = gdata->out_data + Functors::SelectOut(src, eid, dst);
    while (tx < D) {
      DType v1 = Functors::Read(lhs + tx);
      DType v2 = Functors::Read(rhs + tx);
      DType out = Functors::Call(v1, v2);
      Functors::Write(outoff + tx, out);
      tx += stride_x;
    }
  }
}

// functors
template <typename DType, typename EidGetter,
          typename OutSelector, typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct FunctorsTempl {
  static __device__ __forceinline__ int64_t GetEid(int64_t eid, int64_t* id_map) {
    return EidGetter::Call(eid, id_map);
  }
  static __device__ __forceinline__ int64_t SelectOut(
      int64_t src, int64_t eid, int64_t dst) {
    return OutSelector::Call(src, eid, dst);
  }
  static __device__ __forceinline__ DType* SelectLeft(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return LeftSelector::Call(src_data, eid_data, dst_data);
  }
  static __device__ __forceinline__ DType* SelectRight(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return RightSelector::Call(src_data, eid_data, dst_data);
  }
  static __device__ __forceinline__ DType Call(DType lhs, DType rhs) {
    return BinaryOp::Call(lhs, rhs);
  }
  static __device__ __forceinline__ DType Read(DType* addr) {
    return LDGReader<DType>::Call(addr);
  }
  static __device__ __forceinline__ void Write(DType* addr, DType val) {
    Reducer::Call(addr, val);
  }
};

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BINARY_REDUCE_CUH_
