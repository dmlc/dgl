/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_impl.h
 * \brief Minigun CPU UDFs for binary reduce
 */
#ifndef DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>
#include <dgl/immutable_graph.h>

#include <algorithm>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.h"

namespace dgl {
namespace kernel {
namespace cpu {

template <typename DType,
          typename Functors>
struct BinaryReduce {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData<DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData<DType>* gdata) {
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
    for (int64_t tx = 0; tx < D; ++tx) {
      DType lhs = Functors::Read(lhsoff + tx);
      DType rhs = Functors::Read(rhsoff + tx);
      DType out = Functors::Op(lhs, rhs);
      Functors::Write(outoff + tx, out);
    }
  }
};

inline void Unravel(int64_t idx, int ndim,
    const int64_t* shape, const int64_t* stride, int64_t* out) {
  for (int d = 0; d < ndim; ++d) {
    out[d] = (idx / stride[d]) % shape[d];
  }
}

inline int64_t Ravel(const int64_t* idx, int ndim,
    const int64_t* shape, const int64_t* stride) {
  int64_t out = 0;
  for (int d = 0; d < ndim; ++d) {
    out += std::min(idx[d], shape[d] - 1) * stride[d];
  }
  return out;
}

template <int NDim, typename DType, typename Functors>
struct BinaryReduceBcast {
  static inline bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, BcastGData<NDim, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, BcastGData<NDim, DType>* gdata) {
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
    int64_t tmp[NDim];  // store unraveled idx.
    for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType lhs = Functors::Read(lhsoff +
          Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride));
      DType rhs = Functors::Read(rhsoff +
          Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride));
      DType out = Functors::Op(lhs, rhs);
      Functors::Write(outoff + tx, out);
    }
  }
};

template <typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct FunctorsTempl {
  static inline mg_int SelectOut(
      mg_int src, mg_int edge, mg_int dst) {
    return OutSelector<Reducer>::Type::Call(src, edge, dst);
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
};

typedef minigun::advance::Config<true, minigun::advance::kV2N> AdvanceConfig;

}  // namespace cpu

template <int XPU, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce_v2(const minigun::advance::RuntimeConfig& rtcfg,
                         const ImmutableGraph* graph,
                         GData<DType>* gdata) {
  using minigun::IntArray1D;
  typedef cpu::FunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BinaryReduce<DType, Functors> UDF;
  // csr
  auto outcsr = graph->GetOutCSR();
  minigun::Csr csr = utils::CreateCsr(outcsr->indptr(), outcsr->indices());
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<int64_t*>(outcsr->edge_ids()->data);
  }
  if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<int64_t*>(outcsr->edge_ids()->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<int64_t*>(outcsr->edge_ids()->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, cpu::AdvanceConfig, GData<DType>, UDF>(
        rtcfg, csr, gdata, IntArray1D());
}

template <int XPU, int NDim, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast_v2(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BcastGData<NDim, DType>* gdata) {
  using minigun::IntArray1D;
  typedef cpu::FunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BinaryReduceBcast<NDim, DType, Functors> UDF;
  // csr
  auto outcsr = graph->GetOutCSR();
  minigun::Csr csr = utils::CreateCsr(outcsr->indptr(), outcsr->indices());
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<int64_t*>(outcsr->edge_ids()->data);
  }
  if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<int64_t*>(outcsr->edge_ids()->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<int64_t*>(outcsr->edge_ids()->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, cpu::AdvanceConfig,
    BcastGData<NDim, DType>, UDF>(
        rtcfg, csr, gdata, IntArray1D());
}

#define GEN_DEFINE(dtype, lhs_tgt, rhs_tgt, op)                    \
  template void CallBinaryReduce_v2<XPU,                           \
        dtype, lhs_tgt, rhs_tgt, op<dtype>, REDUCER<XPU, dtype>>(  \
      const minigun::advance::RuntimeConfig& rtcfg,                \
      const ImmutableGraph* graph,                                 \
      GData<dtype>* gdata);

#define GEN_BCAST_DEFINE(ndim, dtype, lhs_tgt, rhs_tgt, op)              \
  template void CallBinaryReduceBcast_v2<XPU, ndim, dtype,               \
                                 lhs_tgt, rhs_tgt,                       \
                                 op<dtype>, REDUCER<XPU, dtype>>(        \
      const minigun::advance::RuntimeConfig& rtcfg,                      \
      const ImmutableGraph* graph,                                       \
      BcastGData<ndim, dtype>* gdata);

#define EVAL(F, ...) F(__VA_ARGS__)

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_
