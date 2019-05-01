#ifndef DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>

#include "../binary_reduce_impl_decl.h"
#include "./functor.h"

namespace dgl {
namespace kernel {
namespace cpu {

template <int Mode, typename DType, typename Functors>
struct BackwardBinaryReduce {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, BackwardGData<DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, BackwardGData<DType>* gdata) {
    const int64_t D = gdata->x_length;
    int64_t lid = Functors::SelectLeft(src, eid, dst);
    int64_t rid = Functors::SelectRight(src, eid, dst);
    int64_t oid = Functors::SelectOut(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * D;
    DType* rhsoff = gdata->rhs_data + rid * D;
    DType* outoff = gdata->out_data + oid * D;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * D;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * D;
    DType* gradoutoff = gdata->grad_out_data + oid * D;
    for (int64_t tx = 0; tx < D; ++tx) {
      DType lhs = Functors::Read(lhsoff + tx);
      DType rhs = Functors::Read(rhsoff + tx);
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhs, rhs);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);
      if (Mode == binary_op::kGradLhs || Mode == binary_op::kGradBoth) {
        DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
#pragma omp atomic
        gradlhsoff[tx] += grad_lhs;
      }
      if (Mode == binary_op::kGradRhs || Mode == binary_op::kGradBoth) {
        DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
#pragma omp atomic
        gradrhsoff[tx] += grad_rhs;
      }
    }
  }
};

template <int Mode, int NDim, typename DType, typename Functors>
struct BackwardBinaryReduceBcast {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, BackwardBcastGData<NDim, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, BackwardBcastGData<NDim, DType>* gdata) {
    int64_t lid = Functors::SelectLeft(src, eid, dst);
    int64_t rid = Functors::SelectRight(src, eid, dst);
    int64_t oid = Functors::SelectOut(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len;
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * gdata->out_len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * gdata->out_len;
    DType* gradoutoff = gdata->grad_out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.
    for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType lhs = Functors::Read(lhsoff +
          Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride));
      DType rhs = Functors::Read(rhsoff +
          Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride));
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhs, rhs);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);
      if (Mode == binary_op::kGradLhs || Mode == binary_op::kGradBoth) {
        DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs, rhs, e);
#pragma omp atomic
        gradlhsoff[tx] += grad_lhs;
      }
      if (Mode == binary_op::kGradRhs || Mode == binary_op::kGradBoth) {
        DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs, rhs, e);
#pragma omp atomic
        gradrhsoff[tx] += grad_rhs;
      }
    }
  }
};

template <typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BackwardFunctorsTempl {
  static inline mg_int SelectOut(
      mg_int src, mg_int edge, mg_int dst) {
    return GradOutSelector<Reducer>::Type::Call(src, edge, dst);
  }
  static inline mg_int SelectLeft(
      mg_int src, mg_int edge, mg_int dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static inline mg_int SelectRight(
      mg_int src, mg_int edge, mg_int dst) {
    return RightSelector::Call(src, edge, dst);
  }
  static inline DType Op(DType lhs, DType rhs) {
    return BinaryOp::Call(lhs, rhs);
  }
  static inline DType Read(DType* addr) {
    return *addr;
  }
  static inline void Write(DType* addr, DType val) {
    Reducer::Call(addr, val);
  }
  static inline int64_t GetId(int64_t id, int64_t* id_map) {
    return *(id_map + id);
  }
  static inline DType BackwardWrite(DType val, DType accum) {
    return Reducer::BackwardCall(val, accum);
  }
  static inline DType BackwardOpLhs(DType lhs, DType rhs, DType out) {
    return BinaryOp::BackwardLhs(lhs, rhs, out);
  }
  static inline DType BackwardOpRhs(DType lhs, DType rhs, DType out) {
    return BinaryOp::BackwardRhs(lhs, rhs, out);
  }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N> AdvanceConfig;

}  // namespace cpu

template <int XPU, int Mode, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr, const minigun::Csr& rev_csr,
    BackwardGData<DType>* gdata) {
  using minigun::IntArray1D;
  typedef cpu::BackwardFunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BackwardBinaryReduce<Mode, DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, cpu::AdvanceConfig, BackwardGData<DType>, UDF>(
        rtcfg, rev_csr, gdata, IntArray1D());
}

#define GEN_BACKWARD_DEFINE(mode, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduce<XPU,                   \
                    mode, dtype,                                \
                    lhs_tgt, rhs_tgt,                           \
                    op<dtype>, REDUCER<XPU, dtype>>(            \
      const minigun::advance::RuntimeConfig& rtcfg,             \
      const minigun::Csr& csr,                                  \
      const minigun::Csr& rev_csr,                              \
      BackwardGData<dtype>* gdata);

template <int XPU, int Mode, int NDim, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr, const minigun::Csr& rev_csr,
    BackwardBcastGData<NDim, DType>* gdata) {
  using minigun::IntArray1D;
  typedef cpu::BackwardFunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, cpu::AdvanceConfig,
    BackwardBcastGData<NDim, DType>, UDF>(
        rtcfg, rev_csr, gdata, IntArray1D());
}

#define GEN_BACKWARD_BCAST_DEFINE(mode, ndim, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduceBcast<XPU,                          \
                    mode, ndim, dtype,                                      \
                    lhs_tgt, rhs_tgt,                                       \
                    op<dtype>, REDUCER<XPU, dtype>>(                        \
      const minigun::advance::RuntimeConfig& rtcfg,                         \
      const minigun::Csr& csr,                                              \
      const minigun::Csr& rev_csr,                                          \
      BackwardBcastGData<ndim, dtype>* gdata);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_
