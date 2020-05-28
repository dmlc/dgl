/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_impl.cuh
 * \brief Minigun CUDA UDFs for binary reduce
 */
#ifndef DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_CUH_
#define DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_CUH_

#include <minigun/minigun.h>

#include "../binary_reduce_impl_decl.h"
#include "../utils.h"
#include "./functor.cuh"
#include "../spmat_interface.h"

namespace dgl {
namespace kernel {
namespace cuda {

// Minigun UDF to compute binary reduce.
template <typename Idx, typename DType, typename Functors>
struct BinaryReduce {
  static __device__ __forceinline__ void ApplyEdge(
      Idx src, Idx dst, Idx eid, GData<Idx, DType>* gdata) {
    // Edge parallel mode.
    const int64_t D = gdata->x_length;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
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
    while (tx < D) {
      DType out = Functors::Op(lhsoff + tx * len, rhsoff + tx * len, len);
      Functors::Write(outoff + tx, out);
      tx += stride_x;
    }
  }

  static __device__ __forceinline__ void ApplyEdgeReduce(
      Idx src, Idx dst, Idx eid, Idx feat_idx, DType *outval, GData<Idx, DType>* gdata) {
      // Node parallel mode.
      const int64_t D = gdata->x_length;
    const int64_t len = gdata->data_len;
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

  static __device__ __forceinline__ Idx GetFeatSize(GData<Idx, DType> *gdata) {
    // Feature size 
    return gdata->x_length;
  }

  static __device__ __forceinline__ DType * GetOutBuf(GData<Idx, DType> *gdata) {
    // Output buffer.
    return gdata->out_data;
  }

  static __device__ __forceinline__ Idx GetOutOffset(Idx oid, GData<Idx, DType> *gdata) {
    // Output offset.
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
 * TODO(zihao): we should not put it in advance function actually.
 */
__device__ __forceinline__ void UnravelRavel(
    const int64_t idx, const int ndim, const int64_t* out_shape, const int64_t* out_stride,
    const int64_t* lhs_shape, const int64_t* lhs_stride,
    const int64_t* rhs_shape, const int64_t* rhs_stride, int64_t *lhs_out, int64_t *rhs_out) {
  if (out_stride[0] == lhs_stride[0]) {
#pragma unroll
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
#pragma unroll
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
  static __device__ __forceinline__ void ApplyEdge(
      Idx src, Idx dst, Idx eid, BcastGData<NDim, Idx, DType>* gdata) {
    // Edge parallel mode.
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
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
    DType* lhsoff = gdata->lhs_data + lid * gdata->lhs_len * len; //data with len size
    DType* rhsoff = gdata->rhs_data + rid * gdata->rhs_len * len;
    DType* outoff = gdata->out_data + oid * gdata->out_len;
    while (tx < gdata->out_len) {
      int64_t lhs_add = 0;
      int64_t rhs_add = 0;
      UnravelRavel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride,
          gdata->lhs_shape, gdata->lhs_stride,
          gdata->rhs_shape, gdata->rhs_stride, &lhs_add, &rhs_add);
      DType out = Functors::Op(lhsoff + lhs_add * len, rhsoff + rhs_add * len, len);
      Functors::Write(outoff + tx, out);
      tx += stride_x;
    }
  }

  static __device__ __forceinline__ void ApplyEdgeReduce(
      Idx src, Idx dst, Idx eid, Idx feat_idx, DType *outval, BcastGData<NDim, Idx, DType>* gdata) {
    // Node parallel mode.
    const int64_t len = gdata->data_len; // data length, a single value or a vector
    Idx lid = Functors::SelectLeft(src, eid, dst);
    Idx rid = Functors::SelectRight(src, eid, dst);
    if (gdata->lhs_mapping) {
      lid = Functors::GetId(lid, gdata->lhs_mapping);
    }
    if (gdata->rhs_mapping) {
      rid = Functors::GetId(rid, gdata->rhs_mapping);
    }
    DType* lhs_off = gdata->lhs_data + lid * gdata->lhs_len * len;
    DType* rhs_off = gdata->rhs_data + rid * gdata->rhs_len * len;
    int64_t lhs_add = 0;
    int64_t rhs_add = 0;
    UnravelRavel(feat_idx, gdata->ndim, gdata->out_shape, gdata->out_stride,
        gdata->lhs_shape, gdata->lhs_stride,
        gdata->rhs_shape, gdata->rhs_stride,
        &lhs_add, &rhs_add);
    DType out = Functors::Op(lhs_off + lhs_add * len, rhs_off + rhs_add * len, len);
    Functors::Write(outval, out);
  }

  static __device__ __forceinline__ Idx GetFeatSize(BcastGData<NDim, Idx, DType> *gdata) {
    return gdata->out_len;
  }

  static __device__ __forceinline__ DType * GetOutBuf(BcastGData<NDim, Idx, DType> *gdata) {
    return gdata->out_data;
  }

  static __device__ __forceinline__ Idx GetOutOffset(Idx oid, BcastGData<NDim, Idx, DType> *gdata) {
    if (gdata->out_mapping) {
      oid = Functors::GetId(oid, gdata->out_mapping);
    }

    return oid;
  }
};

// Auxiliary template used in UDF.
template <typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer,
          bool Atomic=false>
struct FunctorsTempl {
  static __device__ __forceinline__ Idx SelectOut(
      Idx src, Idx edge, Idx dst) {
    return OutSelector<Reducer>::Type::Call(src, edge, dst);
  }
  static __device__ __forceinline__ Idx SelectLeft(
      Idx src, Idx edge, Idx dst) {
    return LeftSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ Idx SelectRight(
      Idx src, Idx edge, Idx dst) {
    return RightSelector::Call(src, edge, dst);
  }
  static __device__ __forceinline__ DType Op(DType *lhs, DType *rhs, int64_t len) {
    return BinaryOp::Call(lhs, rhs, len);
  }
  static __device__ __forceinline__ void Write(DType* addr, DType val) {
    // http://www.aerialmantis.co.uk/blog/2017/03/17/template-keywords/
    Reducer::template Call<Atomic>(addr, val);
  }
  static __device__ __forceinline__ Idx GetId(Idx id, Idx* id_map) {
    return LDGReader<Idx>::Call(id_map + id);
  }
};

}  // namespace cuda

// Template implementation of BinaryReduce operator.
template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(const minigun::advance::RuntimeConfig& rtcfg,
                      const SparseMatrixWrapper& graph,
                      GData<Idx, DType>* gdata) {
  LOG(INFO) << "Binary Reduce: " << ToStringSparseFormat(graph.GetRestrictFormat());
  typedef GData<Idx, DType> GDataType;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer, false>
          NonAtomicFunctor;
  typedef cuda::BinaryReduce<Idx, DType, NonAtomicFunctor> NonAtomicUDF;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer, true>
          AtomicFunctor;
  typedef cuda::BinaryReduce<Idx, DType, AtomicFunctor> AtomicUDF;
  auto udf_target = OutSelector<Reducer>::Type::target;
  ADVANCE_DISPATCH(graph, AtomicUDF, NonAtomicUDF, udf_target, GDataType);
}

// Template implementation of BinaryReduce broadcasting operator.
template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
  const minigun::advance::RuntimeConfig& rtcfg,
  const SparseMatrixWrapper& graph,
  BcastGData<NDim, Idx, DType>* gdata) {
  LOG(INFO) << "Binary Reduce Bcast: " << ToStringSparseFormat(graph.GetRestrictFormat());
  typedef BcastGData<NDim, Idx, DType> GDataType;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer, false>
          NonAtomicFunctor;
  typedef cuda::BinaryReduceBcast<NDim, Idx, DType, NonAtomicFunctor> NonAtomicUDF;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer, true>
          AtomicFunctor;
  typedef cuda::BinaryReduceBcast<NDim, Idx, DType, AtomicFunctor> AtomicUDF;
  auto udf_target = OutSelector<Reducer>::Type::target;
  ADVANCE_DISPATCH(graph, AtomicUDF, NonAtomicUDF, udf_target, GDataType);
}

// Following macro is used to generate explicit-specialization of the template
// operator.
#define GEN_DEFINE(dtype, lhs_tgt, rhs_tgt, op)                    \
  template void CallBinaryReduce<XPU, IDX,                      \
        dtype, lhs_tgt, rhs_tgt, op<dtype>, REDUCER<XPU, dtype>>(  \
      const minigun::advance::RuntimeConfig& rtcfg,                \
      const SparseMatrixWrapper& graph,                                     \
      GData<IDX, dtype>* gdata);

#define GEN_BCAST_DEFINE(ndim, dtype, lhs_tgt, rhs_tgt, op)         \
  template void CallBinaryReduceBcast<XPU, ndim, IDX, dtype,     \
                                 lhs_tgt, rhs_tgt,                  \
                                 op<dtype>, REDUCER<XPU, dtype>>(   \
      const minigun::advance::RuntimeConfig& rtcfg,                 \
      const SparseMatrixWrapper& graph,                                      \
      BcastGData<ndim, IDX, dtype>* gdata);

#define EVAL(F, ...) MSVC_EXPAND(F(__VA_ARGS__))

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_CUH_
