/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/backward_binary_reduce_impl.h
 * \brief Minigun CPU UDFs for bacward binary reduce
 */
#ifndef DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.h"
#include "../csr_interface.h"

namespace dgl {
namespace kernel {
namespace cpu {

// Minigun UDF to compute backward binary reduce.
template <int Mode, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduce {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    const int64_t D = gdata->x_length;
    const int64_t len = gdata->data_len;
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    Idx oid = Functors::SelectOut(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * D * len;
    DType* rhsoff = gdata->rhs_data + rid * D * len;
    DType* outoff = gdata->out_data + oid * D;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * D * len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * D * len;
    DType* gradoutoff = gdata->grad_out_data + oid * D;
    for (int64_t tx = 0; tx < D; ++tx) {
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);

      DType* lhs_base = lhsoff + tx * len;
      DType* rhs_base = rhsoff + tx * len;
      if (Mode == binary_op::kGradLhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
          gradlhsoff[tx * len + i] += grad_lhs;
        }
      } else if (Mode == binary_op::kGradRhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
          gradrhsoff[tx * len + i] += grad_rhs;
        }
      }
    }
  }
};

// Minigun UDF to compute backward binary reduce with broadcasting.
template <int Mode, int NDim,
          typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduceBcast {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    const int64_t len = gdata->data_len;
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    Idx oid = Functors::SelectOut(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len;
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * gdata->out_len * len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * gdata->out_len * len;
    DType* gradoutoff = gdata->grad_out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.
    for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(
        lhsoff + Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len,
        rhsoff + Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len,
        len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);

      DType* lhs_base = lhsoff +
          Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride) * len;
      DType* rhs_base = rhsoff +
          Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride) * len;
      if (Mode == binary_op::kGradLhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
          gradlhsoff[tx * len + i] += grad_lhs;
        }
      } else if (Mode == binary_op::kGradRhs) {
        for (int64_t i = 0; i < len; ++i) {
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
          gradrhsoff[tx * len + i] += grad_rhs;
        }
      }
    }
  }
};

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BackwardFunctorsTempl {
  static inline Idx SelectOut(
      Idx src, Idx edge, Idx dst) {
    typedef typename OutSelector<Reducer>::Type OutTarget;
    return SwitchSrcDst<OutTarget>::Type::Call(src, edge, dst);
  }
  static inline Idx SelectLeft(
      Idx src, Idx edge, Idx dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static inline Idx SelectRight(
      Idx src, Idx edge, Idx dst) {
    return RightSelector::Call(src, edge, dst);
  }
  static inline DType Op(DType* lhs, DType* rhs, int64_t len) {
    return BinaryOp::Call(lhs, rhs, len);
  }
  static inline DType Read(DType* addr) {
    return *addr;
  }
  static inline void Write(DType* addr, DType val) {
    Reducer::Call(addr, val);
  }
  static inline Idx GetId(Idx id, Idx* id_map) {
    return *(id_map + id);
  }
  static inline DType BackwardWrite(DType val, DType accum) {
    return Reducer::BackwardCall(val, accum);
  }
  static inline DType BackwardOpLhs(DType* lhs_base,
                                    DType* rhs_base,
                                    int64_t i,
                                    DType out) {
    DType lhs;
    DType rhs;
    switch (BinaryOp::BackwardLhsReadMode) {
      case binary_op::kBackReadRhs:
        rhs = Read(rhs_base + i);
        break;
      case binary_op::kBackReadLhs:
        lhs = Read(lhs_base + i);
        break;
      case binary_op::kGradBoth:
        lhs = Read(lhs_base + i);
        rhs = Read(rhs_base + i);
        break;
      default:
    }

    return BinaryOp::BackwardLhs(lhs, rhs, out);
  }
  static inline DType BackwardOpRhs(DType* lhs_base,
                                    DType* rhs_base,
                                    int64_t i,
                                    DType out) {
    DType lhs;
    DType rhs;
    switch (BinaryOp::BackwardRhsReadMode) {
      case binary_op::kBackReadRhs:
        rhs = Read(rhs_base + i);
        break;
      case binary_op::kBackReadLhs:
        lhs = Read(lhs_base + i);
        break;
      case binary_op::kGradBoth:
        lhs = Read(lhs_base + i);
        rhs = Read(rhs_base + i);
        break;
      default:
    }
    return BinaryOp::BackwardRhs(lhs, rhs, out);
  }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> AdvanceConfig;
typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kSrc> SrcAdvanceConfig;
typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kDst> DstAdvanceConfig;
typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> EdgeAdvanceConfig;

}  // namespace cpu

// Template implementation of BackwardBinaryReduce operator.
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<Idx, DType>* gdata) {
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph.GetInCSRMatrix();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
  typedef cpu::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cpu::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
    if (Mode == binary_op::kGradLhs) {
    if (LeftSelector::target == binary_op::kEdge) {
      // Out Target is Edge, we need use COO format
      auto coo_matrix = graph.GetCOOMatrix();
      minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::EdgeAdvanceConfig, 
        BackwardGData<Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (LeftSelector::target == binary_op::kSrc) {
      // Out Target is source Node, we need use CSR format
      // so data are aggregated in rows
      auto outcsr = graph.GetOutCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {&csr, NULL, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::SrcAdvanceConfig, 
        BackwardGData<Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (LeftSelector::target == binary_op::kDst) {
      // Out Target is destination Node, we need use CSR_t format
      // so data are aggregated in columns
      auto incsr = graph.GetInCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::DstAdvanceConfig, 
        BackwardGData<Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    }
  } else if (Mode == binary_op::kGradRhs) {
    if (RightSelector::target == binary_op::kEdge) {
      // Out Target is Edge, we need use COO format
      auto coo_matrix = graph.GetCOOMatrix();
      minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);
      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::EdgeAdvanceConfig, 
        BackwardGData<Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (RightSelector::target == binary_op::kSrc) {
      // Out Target is source Node, we need use CSR format
      // so data are aggregated in rows
      auto outcsr = graph.GetOutCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {&csr, NULL, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::SrcAdvanceConfig, 
        BackwardGData<Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (RightSelector::target == binary_op::kDst) {
      // Out Target is destination Node, we need use CSR_t format
      // so data are aggregated in columns
      auto incsr = graph.GetInCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::DstAdvanceConfig, 
        BackwardGData<Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    }
  } else if (Mode == binary_op::kGradBoth) {
    CHECK(false) << "Do not support kGradBoth now";
  } else {
    CHECK(false) << "BackwardBinaryReduce Mode not implemented";
  }
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_DEFINE(mode, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduce<XPU,                \
                    mode, IDX, dtype,                           \
                    lhs_tgt, rhs_tgt,                           \
                    op<dtype>, REDUCER<XPU, dtype>>(            \
      const minigun::advance::RuntimeConfig& rtcfg,             \
      const CSRWrapper& graph,                                  \
      BackwardGData<IDX, dtype>* gdata);

// Template implementation of BackwardBinaryReduce with broadcasting operator.
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata) {
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph.GetInCSRMatrix();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
  typedef cpu::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cpu::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;

  if (Mode == binary_op::kGradLhs) {
    if (LeftSelector::target == binary_op::kEdge) {
      // Out Target is Edge, we need use COO format
      auto coo_matrix = graph.GetCOOMatrix();
      minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::EdgeAdvanceConfig,
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (LeftSelector::target == binary_op::kSrc) {
      // Out Target is source Node, we need use CSR format
      // so data are aggregated in rows
      auto outcsr = graph.GetOutCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {&csr, NULL, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::SrcAdvanceConfig,
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (LeftSelector::target == binary_op::kDst) {
      // Out Target is destination Node, we need use CSR_t format
      // so data are aggregated in columns
      auto incsr = graph.GetInCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::DstAdvanceConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    }
  } else if (Mode == binary_op::kGradRhs) {
    if (RightSelector::target == binary_op::kEdge) {
      // Out Target is Edge, we need use COO format
      auto coo_matrix = graph.GetCOOMatrix();
      minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (RightSelector::target == binary_op::kEdge) {
        if (gdata->rhs_mapping == nullptr) {
          gdata->rhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->rhs);
          gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(coo_matrix.data->data);
        } else {
          out_map = aten::MergeIDMapping(coo_matrix.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::EdgeAdvanceConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (RightSelector::target == binary_op::kSrc) {
      // Out Target is source Node, we need use CSR format
      // so data are aggregated in rows
      auto outcsr = graph.GetOutCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(outcsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(outcsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {&csr, NULL, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::SrcAdvanceConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    } else if (RightSelector::target == binary_op::kDst) {
      // Out Target is destination Node, we need use CSR_t format
      // so data are aggregated in columns
      auto incsr = graph.GetInCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);

      // If the user-given mapping is none and the target is edge data, we need to
      // replace the mapping by the edge ids in the csr graph so that the edge
      // data is correctly read/written.
      runtime::NDArray out_map;
      if (LeftSelector::target == binary_op::kEdge) {
        if (gdata->lhs_mapping == nullptr) {
          gdata->lhs_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->lhs);
          gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
        }
      }
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
        if (gdata->out_mapping == nullptr) {
          gdata->out_mapping = static_cast<Idx*>(incsr.data->data);
        } else {
          out_map = aten::MergeIDMapping(incsr.data, gdata->out);
          gdata->out_mapping = static_cast<Idx*>(out_map->data);
        }
      }

      minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
      // TODO(minjie): allocator
      minigun::advance::Advance<XPU, Idx, DType, cuda::DstAdvanceConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
    }
  } else if (Mode == binary_op::kGradBoth) {
    CHECK(false) << "Do not support kGradBoth now";
  } else {
    CHECK(false) << "BackwardBinaryReduce Mode not implemented";
  }
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_BCAST_DEFINE(mode, ndim, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduceBcast<XPU,                       \
                    mode, ndim, IDX, dtype,                                 \
                    lhs_tgt, rhs_tgt,                                       \
                    op<dtype>, REDUCER<XPU, dtype>>(                        \
      const minigun::advance::RuntimeConfig& rtcfg,                         \
      const CSRWrapper& graph,                                              \
      BackwardBcastGData<ndim, IDX, dtype>* gdata);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_BACKWARD_BINARY_REDUCE_IMPL_H_
