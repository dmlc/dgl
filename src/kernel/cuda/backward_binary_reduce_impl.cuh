/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/backward_binary_reduce_impl.cuh
 * \brief Minigun CUDA UDFs for bacward binary reduce
 */
#ifndef DGL_KERNEL_CUDA_BACKWARD_BINARY_REDUCE_IMPL_CUH_
#define DGL_KERNEL_CUDA_BACKWARD_BINARY_REDUCE_IMPL_CUH_

#include <minigun/minigun.h>
#include <dgl/immutable_graph.h>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

// Minigun UDF to compute backward binary reduce.
template <int Mode, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduce {
  static __device__ __forceinline__ bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardGData<Idx, DType>* gdata) {
    const int64_t D = gdata->x_length;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
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
    DType* lhsoff = gdata->lhs_data + lid * D;
    DType* rhsoff = gdata->rhs_data + rid * D;
    DType* outoff = gdata->out_data + oid * D;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * D;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * D;
    DType* gradoutoff = gdata->grad_out_data + oid * D;
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

// Minigun UDF to compute backward binary reduce with broadcasting.
template <int Mode, int NDim, typename Idx, typename DType, typename Functors>
struct BackwardBinaryReduceBcast {
  static __device__ __forceinline__ bool CondEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      Idx src, Idx dst, Idx eid, BackwardBcastGData<NDim, Idx, DType>* gdata) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
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
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len;
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    DType* gradlhsoff = gdata->grad_lhs_data + lid * gdata->out_len;
    DType* gradrhsoff = gdata->grad_rhs_data + rid * gdata->out_len;
    DType* gradoutoff = gdata->grad_out_data + oid * gdata->out_len;
    int64_t tmp[NDim];  // store unraveled idx.
    while (tx < gdata->out_len) {
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

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BackwardFunctorsTempl {
  static __device__ __forceinline__ Idx SelectOut(
      Idx src, Idx edge, Idx dst) {
    typedef typename OutSelector<Reducer>::Type OutTarget;
    return SwitchSrcDst<OutTarget>::Type::Call(src, edge, dst);
  }
  static __device__ __forceinline__ Idx SelectLeft(
      Idx src, Idx edge, Idx dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ Idx SelectRight(
      Idx src, Idx edge, Idx dst) {
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
  static __device__ __forceinline__ Idx GetId(Idx id, Idx* id_map) {
    return LDGReader<Idx>::Call(id_map + id);
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

}  // namespace cuda

// Template implementation of BackwardBinaryReduce operator.
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardGData<Idx, DType>* gdata) {
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph->GetInCSR();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr->indptr(), incsr->indices());
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cuda::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge
      && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  if (RightSelector::target == binary_op::kEdge
      && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_DEFINE(mode, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduce<XPU,                \
                    mode, IDX, dtype,                           \
                    lhs_tgt, rhs_tgt,                           \
                    op<dtype>, REDUCER<XPU, dtype>>(            \
      const minigun::advance::RuntimeConfig& rtcfg,             \
      const ImmutableGraph* graph,                              \
      BackwardGData<IDX, dtype>* gdata);

// Template implementation of BackwardBinaryReduce with broadcasting operator.
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata) {
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph->GetInCSR();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr->indptr(), incsr->indices());
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cuda::BackwardBinaryReduceBcast<Mode, NDim, Idx, DType, Functors> UDF;
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge
      && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  if (RightSelector::target == binary_op::kEdge
      && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig,
    BackwardBcastGData<NDim, Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_BACKWARD_BCAST_DEFINE(mode, ndim, dtype, lhs_tgt, rhs_tgt, op)  \
  template void CallBackwardBinaryReduceBcast<XPU,                       \
                    mode, ndim, IDX, dtype,                                 \
                    lhs_tgt, rhs_tgt,                                       \
                    op<dtype>, REDUCER<XPU, dtype>>(                        \
      const minigun::advance::RuntimeConfig& rtcfg,                         \
      const ImmutableGraph* graph,                                          \
      BackwardBcastGData<ndim, IDX, dtype>* gdata);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BACKWARD_BINARY_REDUCE_IMPL_CUH_
