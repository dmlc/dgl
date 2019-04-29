#pragma once

#include <minigun/minigun.h>

#include "./binary_reduce_impl.h"
#include "./functor.cuh"

namespace dgl {
namespace kernel {
namespace cuda {
template <int Mode, typename DType, typename Functors>
struct BackwardBinaryReduce {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, BackwardGData<DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, BackwardGData<DType>* gdata) {
    const int64_t D = gdata->x_length;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
    int64_t lid = Functors::SelectLeft(src, eid, dst);
    int64_t rid = Functors::SelectRight(src, eid, dst);
    int64_t oid = Functors::SelectOut(src, eid, dst);
    lid = Functors::GetId(lid, gdata->lhs_mapping);
    rid = Functors::GetId(rid, gdata->rhs_mapping);
    oid = Functors::GetId(oid, gdata->out_mapping);
    DType* lhsoff = gdata->lhs_data + lid * D;
    DType* rhsoff = gdata->rhs_data + rid * D;
    DType* outoff = gdata->out_data + oid * D;
    DType* gradoutoff = gdata->grad_out_data + oid * D;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * D;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * D;
    while (tx < D) {
      DType lhs = Functors::Read(lhsoff + tx);
      DType rhs = Functors::Read(rhsoff + tx);
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhs, rhs);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);
      if (Mode == binary_op::kGradLhs || Mode == binary_op::kGradBoth) {
        DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
        AtomicAdd(gradlhsoff + tx, grad_lhs);
      }
      if (Mode == binary_op::kGradRhs || Mode == binary_op::kGradBoth) {
        DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
        AtomicAdd(gradrhsoff + tx, grad_rhs);
      }
      tx += stride_x;
    }
  }
};

template <typename DType, typename IdGetter,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BackwardFunctorsTempl {
  static __device__ __forceinline__ mg_int SelectOut(
      mg_int src, mg_int edge, mg_int dst) {
    return GradOutSelector<Reducer>::Type::Call(src, edge, dst);
  }
  static __device__ __forceinline__ mg_int SelectLeft(
      mg_int src, mg_int edge, mg_int dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ mg_int SelectRight(
      mg_int src, mg_int edge, mg_int dst) {
    return RightSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ DType Op(DType lhs, DType rhs) {
    return BinaryOp::Call(lhs, rhs);
  }
  static __device__ __forceinline__ DType Read(DType* addr) {
    return LDGReader<DType>::Call(addr);
  }
  static __device__ __forceinline__ void Write(DType* addr, DType val) {
    Reducer::Call(addr, val);
  }
  static __device__ __forceinline__ int64_t GetId(int64_t id, int64_t* id_map) {
    return IdGetter::Call(id, id_map);
  }
  static __device__ __forceinline__ DType BackwardWrite(DType val, DType accum) {
    return Reducer::BackwardCall(val, accum);
  }
  static __device__ __forceinline__ DType BackwardOpLhs(DType lhs, DType rhs, DType out) {
    return BinaryOp::BackwardLhs(lhs, rhs, out);
  }
  static __device__ __forceinline__ DType BackwardOpRhs(DType lhs, DType rhs, DType out) {
    return BinaryOp::BackwardRhs(lhs, rhs, out);
  }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N> AdvanceConfig;

template <int Mode, typename DType, typename IdGetter,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    BackwardGData<DType>* gdata) {
  using minigun::IntArray1D;
  typedef BackwardFunctorsTempl<DType, IdGetter, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef BackwardBinaryReduce<Mode, DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<kDLGPU, AdvanceConfig, BackwardGData<DType>, UDF>(
        rtcfg, csr, gdata, IntArray1D());
}

#define GEN_BACKWARD_DEFINE(mode, dtype, lhs_tgt, rhs_tgt, op) \
  template void CallBackwardBinaryReduce< \
                    mode, dtype, GETID<XPU, int64_t>, \
                    lhs_tgt, rhs_tgt, \
                    op<dtype>, REDUCER<XPU, dtype>>( \
      const minigun::advance::RuntimeConfig& rtcfg, \
      const minigun::Csr& csr, \
      BackwardGData<dtype>* gdata);

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl
