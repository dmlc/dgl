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
  DType* lhs_data, rhs_data;
  // input id mappings
  int64_t* lhs_mapping, rhs_mapping;
  // output data
  DType* out_data;
  // output id mapping
  int64_t* out_mapping;
};

template <typename DType,
          typename Functors>
struct BinaryReduce {
  static __device__ __forceinline__ bool CondEdge(
      int64_t src, int64_t dst, int64_t eid, GData<DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      int64_t src, int64_t dst, int64_t eid, GData<DType>* gdata) {
    const int64_t D = gdata->x_length;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int64_t lid = Functors::SelectLeft(src, eid, dst);
    int64_t rid = Functors::SelectRight(src, eid, dst);
    int64_t oid = Functors::SelectOut(src, eid, dst);
    lid = Functors::MapLeft(lid, gdata->lhs_mapping);
    rid = Functors::MapRight(rid, gdata->rhs_mapping);
    oid = Functors::MapOut(oid, gdata->out_mapping);
    DType* lhsoff = gdata->lhs_data + lid * D;
    DType* rhsoff = gdata->rhs_data + rid * D;
    DType* outoff = gdata->out_data + oid * D;
    while (tx < D) {
      DType lhs = Functors::Read(lhsoff + tx);
      DType rhs = Functors::Read(rhsoff + tx);
      DType out = Functors::Call(lhs, rhs);
      Functors::Write(outoff + tx, out);
      tx += stride_x;
    }
  }
};

// functors
template <typename DType,
          typename OutIdGetter, typename LeftIdGetter, typename RightIdGetter,
          typename OutSelector, typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct FunctorsTempl {
  static __device__ __forceinline__ int64_t SelectOut(
      int64_t src, int64_t edge, int64_t dst) {
    return OutSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ int64_t SelectLeft(
      int64_t src, int64_t edge, int64_t dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ int64_t SelectRight(
      int64_t src, int64_t edge, int64_t dst) {
    return RightSelector::Call(src, edge, dst);
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
  static __device__ __forceinline__ int64_t MapLeft(int64_t id, int64_t* id_map) {
    return LeftIdGetter::Call(id, id_map);
  }
  static __device__ __forceinline__ int64_t MapRight(int64_t id, int64_t* id_map) {
    return RightIdGetter::Call(id, id_map);
  }
  static __device__ __forceinline__ int64_t MapOut(int64_t id, int64_t* id_map) {
    return OutIdGetter::Call(id, id_map);
  }
};

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BINARY_REDUCE_CUH_
