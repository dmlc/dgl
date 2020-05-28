/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/backward_binary_reduce_impl.cuh
 * \brief Minigun CUDA UDFs for bacward binary reduce
 */
#ifndef DGL_KERNEL_CUDA_BACKWARD_BINARY_REDUCE_IMPL_CUH_
#define DGL_KERNEL_CUDA_BACKWARD_BINARY_REDUCE_IMPL_CUH_

#include <minigun/minigun.h>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.cuh"
#include "../spmat_interface.h"

namespace dgl {
namespace kernel {
namespace cuda {

// Minigun UDF to compute backward binary reduce.
template <int Mode, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduce {
  static __device__ __forceinline__ void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    const int64_t D = gdata->x_length;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
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
    while (tx < D) {
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);

      DType* lhs_base = lhsoff + tx * len;
      DType* rhs_base = rhsoff + tx * len;
      if (Mode == binary_op::kGradLhs) {
#pragma unroll
        for (int64_t i = 0; i < len; ++i) {
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
          Functors::Write(gradlhsoff + tx * len + i, grad_lhs);
        }
      } else if (Mode == binary_op::kGradRhs) {
#pragma unroll
        for (int64_t i = 0; i < len; ++i) {
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs_base, rhs_base, i, e);
          Functors::Write(gradrhsoff + tx * len + i, grad_rhs);
        }
      }
      tx += stride_x;
    }
  }

  static __device__ __forceinline__ void ApplyEdgeReduce(
      Idx src, Idx dst, Idx eid, Idx feat_idx, DType *outval, BackwardGData<Idx, DType>* gdata) {
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
    DType* gradoutoff = gdata->grad_out_data + oid * D;

    Idx tx = feat_idx / len;
    DType out = Functors::Read(outoff + tx);
    DType grad_out = Functors::Read(gradoutoff + tx);
    DType e = Functors::Op(lhsoff + feat_idx, rhsoff + feat_idx, len);
    DType grad_e = grad_out * Functors::BackwardWrite(e, out);

    DType* lhs_base = lhsoff + tx * len;
    DType* rhs_base = rhsoff + tx * len;
    int64_t i = feat_idx % len;
    if (Mode == binary_op::kGradLhs) {
      DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
      *outval += grad_lhs;
    } else if (Mode == binary_op::kGradRhs) {
      DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs_base, rhs_base, i, e);
      *outval += grad_rhs;
    }
  }

  static __device__ __forceinline__ Idx GetFeatSize(BackwardGData<Idx, DType> *gdata) {
    return gdata->x_length * gdata->data_len;
  }

  static __device__ __forceinline__ DType * GetOutBuf(BackwardGData<Idx, DType> *gdata) {
    if (Mode == binary_op::kGradLhs) {
      return gdata->grad_lhs_data;
    } else { // (Mode == binary_op::kGradRhs)
      return gdata->grad_rhs_data;
    }
  }

  static __device__ __forceinline__ Idx GetOutOffset(Idx id, BackwardGData<Idx, DType> *gdata) {
    if (Mode == binary_op::kGradLhs) {
      if (gdata->lhs_mapping) {
        return Functors::GetId(id, gdata->lhs_mapping);
      }

      return id;
    } else {// (Mode == binary_op::kGradRhs)
      if (gdata->rhs_mapping) {
        return Functors::GetId(id, gdata->rhs_mapping);
      }
      return id;
    }
  }
};

// Minigun UDF to compute backward binary reduce with broadcasting.
template <int Mode, int NDim, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduceBcast {
  static __device__ __forceinline__ void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
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
    while (tx < gdata->out_len) {
      int64_t lhs_add = 0;
      int64_t rhs_add = 0;
      UnravelRavel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride,
          gdata->lhs_shape, gdata->lhs_stride,
          gdata->rhs_shape, gdata->rhs_stride, &lhs_add, &rhs_add);
      DType out = Functors::Read(outoff + tx);
      DType grad_out = Functors::Read(gradoutoff + tx);
      DType e = Functors::Op(lhsoff + lhs_add * len, rhsoff + rhs_add * len, len);
      DType grad_e = grad_out * Functors::BackwardWrite(e, out);

      DType* lhs_base = lhsoff + lhs_add * len;
      DType* rhs_base = rhsoff + rhs_add * len;
      if (Mode == binary_op::kGradLhs) {
#pragma unroll
        for (int64_t i = 0; i < len; ++i) {
          DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
          Functors::Write(gradlhsoff + tx * len + i, grad_lhs);
        }
      } else if (Mode == binary_op::kGradRhs) {
#pragma unroll
        for (int64_t i = 0; i < len; ++i) {
          DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs_base, rhs_base, i, e);
          Functors::Write(gradrhsoff + tx * len + i, grad_rhs);
        }
      }
      tx += stride_x;
    }
  }

  static __device__ __forceinline__ void ApplyEdgeReduce(
      Idx src, Idx dst, Idx eid, Idx feat_idx, DType *outval, BackwardBcastGData<NDim, Idx, DType>* gdata) {
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
    DType* gradoutoff = gdata->grad_out_data + oid * gdata->out_len;

    int64_t lhs_add = 0;
    int64_t rhs_add = 0;
    Idx tx = feat_idx/len;
    UnravelRavel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride,
        gdata->lhs_shape, gdata->lhs_stride,
        gdata->rhs_shape, gdata->rhs_stride, &lhs_add, &rhs_add);
    DType out = Functors::Read(outoff + tx);
    DType grad_out = Functors::Read(gradoutoff + tx);
    DType e = Functors::Op(lhsoff + lhs_add * len, rhsoff + rhs_add * len, len);
    DType grad_e = grad_out * Functors::BackwardWrite(e, out);

    DType* lhs_base = lhsoff + lhs_add * len;
    DType* rhs_base = rhsoff + rhs_add * len;
    int64_t i = feat_idx%len;
    if (Mode == binary_op::kGradLhs) {
      DType grad_lhs = grad_e * Functors::BackwardOpLhs(lhs_base, rhs_base, i, e);
      *outval += grad_lhs;
    } else if (Mode == binary_op::kGradRhs) {
      DType grad_rhs = grad_e * Functors::BackwardOpRhs(lhs_base, rhs_base, i, e);
      *outval += grad_rhs;
    }
  }

  static __device__ __forceinline__ Idx GetFeatSize(BackwardBcastGData<NDim, Idx, DType> *gdata) {
    return gdata->out_len * gdata->data_len;
  }

  static __device__ __forceinline__ DType * GetOutBuf(BackwardBcastGData<NDim, Idx, DType> *gdata) {
    if (Mode == binary_op::kGradLhs) {
      return gdata->grad_lhs_data;
    } else { // (Mode == binary_op::kGradRhs)
      return gdata->grad_rhs_data;
    }
  }

  static __device__ __forceinline__ Idx GetOutOffset(Idx id, BackwardBcastGData<NDim, Idx, DType> *gdata) {
    if (Mode == binary_op::kGradLhs) {
      if (gdata->lhs_mapping) {
        return Functors::GetId(id, gdata->lhs_mapping);
      }

      return id;
    } else { // (Mode == binary_op::kGradRhs)
      if (gdata->rhs_mapping) {
        return Functors::GetId(id, gdata->rhs_mapping);
      }
      return id;
    }
  }
};

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer,
          bool Atomic=false>
struct BackwardFunctorsTempl {
  static __device__ __forceinline__ Idx SelectOut(
      Idx src, Idx edge, Idx dst) {
    typedef typename OutSelector<Reducer>::Type OutTarget;
    // return SwitchSrcDst<OutTarget>::Type::Call(src, edge, dst);
    return OutTarget::Call(src, edge, dst);
  }
  static __device__ __forceinline__ Idx SelectLeft(
      Idx src, Idx edge, Idx dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ Idx SelectRight(
      Idx src, Idx edge, Idx dst) {
    return RightSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ DType Op(DType* lhs, DType* rhs, int64_t len) {
    return BinaryOp::Call(lhs, rhs, len);
  }
  static __device__ __forceinline__ DType Read(DType* addr) {
    return LDGReader<DType>::Call(addr);
  }
  static __device__ __forceinline__ void Write(DType* addr, DType val) {
    if (!Atomic) {
      *addr += val;
    } else {
      AtomicAdd(addr, val);
    }
  }
  static __device__ __forceinline__ Idx GetId(Idx id, Idx* id_map) {
    return LDGReader<Idx>::Call(id_map + id);
  }
  static __device__ __forceinline__ DType BackwardWrite(DType val, DType accum) {
    return Reducer::BackwardCall(val, accum);
  }
  static __device__ __forceinline__ DType BackwardOpLhs(DType* lhs_base,
                                                        DType* rhs_base,
                                                        int64_t i,
                                                        DType out) {
    DType lhs = 0;
    DType rhs = 0;
    switch (BinaryOp::BackwardLhsReadMode()) {
      case binary_op::kBackReadRhs:
        rhs = Read(rhs_base + i);
        break;
      case binary_op::kBackReadLhs:
        lhs = Read(lhs_base + i);
        break;
      case binary_op::kBackReadBoth:
        lhs = Read(lhs_base + i);
        rhs = Read(rhs_base + i);
        break;
      default:
        break;
    }
    return BinaryOp::BackwardLhs(lhs, rhs, out);
  }

  static __device__ __forceinline__ DType BackwardOpRhs(DType* lhs_base,
                                                        DType* rhs_base,
                                                        int64_t i,
                                                        DType out) {
    DType lhs = 0;
    DType rhs = 0;
    switch (BinaryOp::BackwardRhsReadMode()) {
      case binary_op::kBackReadRhs:
        rhs = Read(rhs_base + i);
        break;
      case binary_op::kBackReadLhs:
        lhs = Read(lhs_base + i);
        break;
      case binary_op::kBackReadBoth:
        lhs = Read(lhs_base + i);
        rhs = Read(rhs_base + i);
        break;
      default:
        break;
    }
    return BinaryOp::BackwardRhs(lhs, rhs, out);
  }
};

typedef minigun::advance::Config<minigun::advance::kSrc> AdvanceSrcConfig;
typedef minigun::advance::Config<minigun::advance::kEdge> AdvanceEdgeConfig;
typedef minigun::advance::Config<minigun::advance::kDst> AdvanceDstConfig;

}  // namespace cuda

// Template implementation of BackwardBinaryReduce operator.
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const SparseMatrixWrapper& graph,
    BackwardGData<Idx, DType>* gdata) {
  typedef BackwardGData<Idx, DType> GDataType;
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          LeftSelector, RightSelector,
          BinaryOp, Reducer, false> NonAtomicFunctor;
  typedef cuda::BackwardBinaryReduce<Mode, Idx, DType, NonAtomicFunctor> NonAtomicUDF;
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          LeftSelector, RightSelector,
          BinaryOp, Reducer, false> AtomicFunctor;
  typedef cuda::BackwardBinaryReduce<Mode, Idx, DType, AtomicFunctor> AtomicUDF;
  ADVANCE_DISPATCH(graph, AtomicUDF, NonAtomicUDF, Mode, GDataType);
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_DEFINE(mode, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduce<XPU,                \
                    mode, IDX, dtype,                           \
                    lhs_tgt, rhs_tgt,                           \
                    op<dtype>, REDUCER<XPU, dtype>>(            \
      const minigun::advance::RuntimeConfig& rtcfg,             \
      const SparseMatrixWrapper& graph,                                  \
      BackwardGData<IDX, dtype>* gdata);

// Template implementation of BackwardBinaryReduce with broadcasting operator.
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const SparseMatrixWrapper& graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata) {
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          LeftSelector, RightSelector,
          BinaryOp, Reducer, false> Functors;
  typedef cuda::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
  if (Mode == binary_op::kGradLhs) {
    if (LeftSelector::target == binary_op::kEdge) {
      // Out Target is Edge, we need use COO format
      auto coo_matrix = graph.GetCOOMatrix();
      minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);
      if (RightSelector::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->rhs_mapping), gdata->rhs, coo_matrix.data);
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, coo_matrix.data);
      utils::ComputeEdgeMapping<Idx>(&(gdata->lhs_mapping), gdata->lhs, coo_matrix.data);

      minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
      minigun::advance::Advance<XPU, Idx, DType, cuda::AdvanceEdgeConfig,
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata);
    } else if (LeftSelector::target == binary_op::kSrc) {
      // Out Target is source Node, we need use CSR format
      // so data are aggregated in rows
      auto outcsr = graph.GetOutCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
      if (RightSelector::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->rhs_mapping), gdata->rhs, outcsr.data);
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, outcsr.data);

      minigun::SpMat<Idx> spmat = {&csr, NULL, NULL};
      minigun::advance::Advance<XPU, Idx, DType, cuda::AdvanceSrcConfig,
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata);
    } else if (LeftSelector::target == binary_op::kDst) {
      // Out Target is destination Node, we need use CSR_t format
      // so data are aggregated in columns
      auto incsr = graph.GetInCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
      if (RightSelector::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->rhs_mapping), gdata->rhs, incsr.data);
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, incsr.data);

      minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
      minigun::advance::Advance<XPU, Idx, DType, cuda::AdvanceDstConfig,
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata);
    }
  } else if (Mode == binary_op::kGradRhs) {
    if (RightSelector::target == binary_op::kEdge) {
      // Out Target is Edge, we need use COO format
      auto coo_matrix = graph.GetCOOMatrix();
      minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);
      if (LeftSelector::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->lhs_mapping), gdata->lhs, coo_matrix.data);
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, coo_matrix.data);
      utils::ComputeEdgeMapping<Idx>(&(gdata->rhs_mapping), gdata->rhs, coo_matrix.data);

      minigun::SpMat<Idx> spmat = {NULL, NULL, &coo};
      minigun::advance::Advance<XPU, Idx, DType, cuda::AdvanceEdgeConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata);
    } else if (RightSelector::target == binary_op::kSrc) {
      // Out Target is source Node, we need use CSR format
      // so data are aggregated in rows
      auto outcsr = graph.GetOutCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);
      if (LeftSelector::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->lhs_mapping), gdata->lhs, outcsr.data);
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, outcsr.data);

      minigun::SpMat<Idx> spmat = {&csr, NULL, NULL};
      minigun::advance::Advance<XPU, Idx, DType, cuda::AdvanceSrcConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata);
    } else if (RightSelector::target == binary_op::kDst) {
      // Out Target is destination Node, we need use CSR_t format
      // so data are aggregated in columns
      auto incsr = graph.GetInCSRMatrix();
      minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);
      if (LeftSelector::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->lhs_mapping), gdata->lhs, incsr.data);
      if (OutSelector<Reducer>::Type::target == binary_op::kEdge)
        utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, incsr.data);

      minigun::SpMat<Idx> spmat = {NULL, &csr, NULL};
      minigun::advance::Advance<XPU, Idx, DType, cuda::AdvanceDstConfig, 
        BackwardBcastGData<NDim, Idx, DType>, UDF>(
            rtcfg, spmat, gdata);
    }
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
      const SparseMatrixWrapper& graph,                                              \
      BackwardBcastGData<ndim, IDX, dtype>* gdata);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BACKWARD_BINARY_REDUCE_IMPL_CUH_
