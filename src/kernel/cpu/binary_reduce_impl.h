/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_impl.h
 * \brief Minigun CPU UDFs for binary reduce
 */
#ifndef DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>

#include <algorithm>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.h"
#include "../csr_interface.h"

namespace dgl {
namespace kernel {
namespace cpu {

// Minigun UDF to compute binary reduce.
template <typename Idx, typename DType, typename Functors>
struct BinaryReduce {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
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
    for (int64_t tx = 0; tx < D; ++tx) {
      DType out = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
      DType* outaddr = outoff + tx;
      DType outval = *outaddr;
      Functors::Write(outval, out);
      *outaddr = outval;
    }
  }

  static void ApplyEdgeReduce(
    Idx src, Idx dst, Idx eid, Idx feat_idx, DType &outval, GData<Idx, DType>* gdata) {
    const int64_t D = gdata->x_length;
    const int64_t len = gdata->data_len;
    // LOG(INFO) << "inside ApplyEdgeReduce";
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }

    DType* lhsoff = gdata->lhs_data + lid * D * len;
    DType* rhsoff = gdata->rhs_data + rid * D * len;
    DType out = Functors::Op(lhsoff + feat_idx * len, rhsoff + feat_idx * len, len);
    Functors::Write(outval, out);
  }

  static inline Idx GetFeatSize(GData<Idx, DType> *gdata) {
    return gdata->x_length;
  }

  static inline DType *GetOutBuf(GData<Idx, DType> *gdata) {
    return gdata->out_data;
  }

  static inline Idx GetOutOffset(Idx oid, GData<Idx, DType> *gdata) {
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }

    return oid;
  }
};

/*
 * This func do the followings:
 *   1. Convert flattened index to multi-dimension index
 *      according to output shape (assume row-major).
 *   2. Convert multi-dimension index to flattened index for lhs.
 *   3. Convert multi-dimension index to flattened index for rhs.
 */
inline void UnravelRavel(
    const int64_t idx, const int ndim, const int64_t* out_shape, const int64_t* out_stride,
    const int64_t* lhs_shape, const int64_t* lhs_stride,
    const int64_t* rhs_shape, const int64_t* rhs_stride, int64_t *lhs_out, int64_t *rhs_out) {
  if (out_stride[0] == lhs_stride[0]) {
    for (int d = 0; d < ndim; ++d) {
      int64_t o_sh = out_shape[d];
      int64_t o_st = out_stride[d];
      int64_t rhs_sh = rhs_shape[d];
      int64_t rhs_st = rhs_stride[d];
      int64_t i = (idx / o_st) % o_sh;
      /*
       * Simplfied for rhs_out += min(i, rhs_sh - 1) * rhs_st;
       * rhs_sh be o_sh or 1
       */
      if (rhs_sh > i) {
        *rhs_out += i * rhs_st;
      }
    }
    *lhs_out = idx;
  } else {
    for (int d = 0; d < ndim; ++d) {
      int64_t o_sh = out_shape[d];
      int64_t o_st = out_stride[d];
      int64_t lhs_sh = lhs_shape[d];
      int64_t lhs_st = lhs_stride[d];

      int64_t i = (idx / o_st) % o_sh;
      /*
       * Simplfied for lhs_out += min(i, lhs_sh - 1) * lhs_st;
       * lhs_sh be o_sh or 1
       */
      if (lhs_sh > i) {
        *lhs_out += i * lhs_st;
      }
    }
    *rhs_out = idx;
  }
}

// Minigun UDF to compute binary reduce with broadcasting.
template <int NDim, typename Idx, typename DType, typename Functors>
struct BinaryReduceBcast {
  static inline bool CondEdge(
      Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
    return true;
  }
  static inline void ApplyEdge(
      Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
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
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len;  // data with len size
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.
    for (int64_t tx = 0; tx < gdata->out_len; ++tx) {
      int64_t lhs_add = 0;
      int64_t rhs_add = 0;
      UnravelRavel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride,
          gdata->lhs_shape, gdata->lhs_stride,
          gdata->rhs_shape, gdata->rhs_stride, &lhs_add, &rhs_add);
      DType out = Functors::Op(lhsoff + lhs_add * len, rhsoff + rhs_add * len, len);

      DType* outaddr = outoff + tx;
      DType outval = *outaddr;
      Functors::Write(outval, out);
      *outaddr = outval;
    }
  }

  static inline void ApplyEdgeReduce(
    Idx src, Idx dst, Idx eid, Idx feat_idx, DType &outval, BcastGData<NDim, Idx, DType>* gdata) {
    const int64_t len = gdata->data_len;
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len; //data with len size
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;

    int64_t lhs_add = 0;
    int64_t rhs_add = 0;
    UnravelRavel(feat_idx, gdata->ndim, gdata->out_shape, gdata->out_stride,
        gdata->lhs_shape, gdata->lhs_stride,
        gdata->rhs_shape, gdata->rhs_stride, &lhs_add, &rhs_add);
    DType out = Functors::Op(lhsoff + lhs_add * len, rhsoff + rhs_add * len, len);
    Functors::Write(outval, out);
  }

  static inline Idx GetFeatSize(BcastGData<NDim, Idx, DType> *gdata) {
    return gdata->out_len;
  }

  static inline DType *GetOutBuf(BcastGData<NDim, Idx, DType> *gdata) {
    return gdata->out_data;
  }

  static inline Idx GetOutOffset(Idx oid, BcastGData<NDim, Idx, DType> *gdata) {
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }

    return oid;
  }
};

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct FunctorsTempl {
  static inline Idx SelectOut(
      Idx src, Idx edge, Idx dst) {
    return OutSelector<Reducer>::Type::Call(src, edge, dst);
  }
  static inline Idx SelectLeft(
      Idx src, Idx edge, Idx dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static inline Idx SelectRight(
      Idx src, Idx edge, Idx dst) {
    return RightSelector::Call(src, edge, dst);
  }
  static inline DType Op(DType *lhs, DType *rhs, int64_t len) {
    return BinaryOp::Call(lhs, rhs, len);
  }
  static inline void Write(DType &addr, DType val) {
    Reducer::Call(addr, val);
  }
  static inline Idx GetId(Idx id, Idx* id_map) {
    return *(id_map + id);
  }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kSrc> SrcAdvanceConfig;
typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kDst> DstAdvanceConfig;
typedef minigun::advance::Config<true, minigun::advance::kV2N, minigun::advance::kEdge> EdgeAdvanceConfig;

}  // namespace cpu

// Template implementation of BinaryReduce operator.
template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(const minigun::advance::RuntimeConfig& rtcfg,
                      const CSRWrapper& graph,
                      GData<Idx, DType>* gdata) {
  typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BinaryReduce<Idx, DType, Functors> UDF;
  
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
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
    if (gdata->out_mapping == nullptr) {
      gdata->out_mapping = static_cast<Idx*>(coo_matrix.data->data);
    } else {
      out_map = aten::MergeIDMapping(coo_matrix.data, gdata->out);
      gdata->out_mapping = static_cast<Idx*>(out_map->data);
    }

    minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
    // TODO(minjie): allocator
    minigun::advance::Advance<XPU, Idx, DType, cpu::EdgeAdvanceConfig, 
      GData<Idx, DType>, UDF>(
          rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
  } else if (OutSelector<Reducer>::Type::target == binary_op::kSrc) {
    CHECK(false) << "BinaryReduce target should not be kSrc";
  } else if (OutSelector<Reducer>::Type::target == binary_op::kDst) {
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
    if (RightSelector::target == binary_op::kEdge) {
      if (gdata->rhs_mapping == nullptr) {
        gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
      } else {
        out_map = aten::MergeIDMapping(incsr.data, gdata->rhs);
        gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
      }
    }

    minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
    // TODO(minjie): allocator
    minigun::advance::Advance<XPU, Idx, DType, cpu::DstAdvanceConfig, 
      GData<Idx, DType>, UDF>(
          rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
  }
}

// Template implementation of BinaryReduce broadcasting operator.
template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
  const minigun::advance::RuntimeConfig& rtcfg,
  const CSRWrapper& graph,
  BcastGData<NDim, Idx, DType>* gdata) {
  typedef cpu::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cpu::BinaryReduceBcast<NDim, Idx, DType, Functors> UDF;

  if (OutSelector<Reducer>::Type::target == binary_op::kEdge) {
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
        auto target_mapping = coo_matrix.data;
        out_map = aten::MergeIDMapping(target_mapping, gdata->lhs);
        gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
      }
    }
    if (RightSelector::target == binary_op::kEdge) {
      if (gdata->rhs_mapping == nullptr) {
        gdata->rhs_mapping = static_cast<Idx*>(coo_matrix.data->data);
      } else {
        auto target_mapping = coo_matrix.data;
        out_map = aten::MergeIDMapping(target_mapping, gdata->rhs);
        gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
      }
    }
    if (gdata->out_mapping == nullptr) {
      gdata->out_mapping = static_cast<Idx*>(coo_matrix.data->data);
    } else {
      auto target_mapping = coo_matrix.data;
      out_map = aten::MergeIDMapping(target_mapping, gdata->out);
      gdata->out_mapping = static_cast<Idx*>(out_map->data);
    }

    minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
    // TODO(minjie): allocator
    minigun::advance::Advance<XPU, Idx, DType, cpu::EdgeAdvanceConfig, 
      BcastGData<NDim, Idx, DType>, UDF>(
          rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
  } else if (OutSelector<Reducer>::Type::target == binary_op::kSrc) {
    CHECK(false) << "BinaryReduceBcast target should not be kSrc";
  } else if (OutSelector<Reducer>::Type::target == binary_op::kDst) {
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
        auto target_mapping = incsr.data;
        out_map = aten::MergeIDMapping(target_mapping, gdata->lhs);
        gdata->lhs_mapping = static_cast<Idx*>(out_map->data);
      }
    }
    if (RightSelector::target == binary_op::kEdge) {
      if (gdata->rhs_mapping == nullptr) {
        gdata->rhs_mapping = static_cast<Idx*>(incsr.data->data);
      } else {
        auto target_mapping = incsr.data;
        out_map = aten::MergeIDMapping(target_mapping, gdata->rhs);
        gdata->rhs_mapping = static_cast<Idx*>(out_map->data);
      }
    }

    minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
    // TODO(minjie): allocator
    minigun::advance::Advance<XPU, Idx, DType, cpu::DstAdvanceConfig, 
      BcastGData<NDim, Idx, DType>, UDF>(
          rtcfg, spmat, gdata, minigun::IntArray1D<Idx>());
  }
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_DEFINE(dtype, lhs_tgt, rhs_tgt, op)                    \
  template void CallBinaryReduce<XPU, IDX,                      \
        dtype, lhs_tgt, rhs_tgt, op<dtype>, REDUCER<XPU, dtype>>(  \
      const minigun::advance::RuntimeConfig& rtcfg,                \
      const CSRWrapper& graph,                                     \
      GData<IDX, dtype>* gdata);

#define GEN_BCAST_DEFINE(ndim, dtype, lhs_tgt, rhs_tgt, op)         \
  template void CallBinaryReduceBcast<XPU, ndim, IDX, dtype,     \
                                 lhs_tgt, rhs_tgt,                  \
                                 op<dtype>, REDUCER<XPU, dtype>>(   \
      const minigun::advance::RuntimeConfig& rtcfg,                 \
      const CSRWrapper& graph,                                      \
      BcastGData<ndim, IDX, dtype>* gdata);

#define EVAL(F, ...) MSVC_EXPAND(F(__VA_ARGS__))

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_BINARY_REDUCE_IMPL_H_
