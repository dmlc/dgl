#ifndef DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_CUH_
#define DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_CUH_

#include <minigun/minigun.h>

#include "./binary_reduce_impl.h"
#include "./functor.cuh"

namespace dgl {
namespace kernel {
namespace cuda {

template <typename DType,
          typename Functors>
struct BinaryReduce {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, GData<DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, GData<DType>* gdata) {
    const int64_t D = gdata->x_length;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
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
    while (tx < D) {
      DType lhs = Functors::Read(lhsoff + tx);
      DType rhs = Functors::Read(rhsoff + tx);
      DType out = Functors::Op(lhs, rhs);
      Functors::Write(outoff + tx, out);
      tx += stride_x;
    }
  }
};

__device__ __forceinline__ void Unravel(
    int64_t idx, int ndim, const int64_t* shape, const int64_t* stride, int64_t* out) {
  for (int d = 0; d < ndim; ++d) {
    out[d] = (idx / stride[d]) % shape[d];
  }
}

__device__ __forceinline__ int64_t Ravel(
    const int64_t* idx, int ndim, const int64_t* shape, const int64_t* stride) {
  int64_t out = 0;
  for (int d = 0; d < ndim; ++d) {
    out += min(idx[d], shape[d] - 1) * stride[d];
  }
  return out;
}

template <int NDim, typename DType, typename Functors>
struct BinaryReduceBcast {
  static __device__ __forceinline__ bool CondEdge(
      mg_int src, mg_int dst, mg_int eid, BcastGData<NDim, DType>* gdata) {
    return true;
  }
  static __device__ __forceinline__ void ApplyEdge(
      mg_int src, mg_int dst, mg_int eid, BcastGData<NDim, DType>* gdata) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_x = blockDim.x * gridDim.x;
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
    while (tx < gdata->out_len) {
      Unravel(tx, gdata->ndim, gdata->out_shape, gdata->out_stride, tmp);
      DType lhs = Functors::Read(lhsoff +
          Ravel(tmp, gdata->ndim, gdata->lhs_shape, gdata->lhs_stride));
      DType rhs = Functors::Read(rhsoff +
          Ravel(tmp, gdata->ndim, gdata->rhs_shape, gdata->rhs_stride));
      DType out = Functors::Op(lhs, rhs);
      Functors::Write(outoff + tx, out);
      tx += stride_x;
    }
  }
};

template <typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct FunctorsTempl {
  static __device__ __forceinline__ mg_int SelectOut(
      mg_int src, mg_int edge, mg_int dst) {
    return OutSelector<Reducer>::Type::Call(src, edge, dst);
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
    return LDGReader<int64_t>::Call(id_map + id);
  }
};

typedef minigun::advance::Config<true, minigun::advance::kV2N> AdvanceConfig;

template <typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    const minigun::Csr& rev_csr,
    GData<DType>* gdata) {
  using minigun::IntArray1D;
  typedef FunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef BinaryReduce<DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<kDLGPU, AdvanceConfig, GData<DType>, UDF>(
        rtcfg, csr, gdata, IntArray1D());
}

template <int NDim, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    const minigun::Csr& rev_csr,
    BcastGData<NDim, DType>* gdata) {
  using minigun::IntArray1D;
  typedef FunctorsTempl<DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef BinaryReduceBcast<NDim, DType, Functors> UDF;
  // TODO(minjie): allocator
  minigun::advance::Advance<kDLGPU, AdvanceConfig,
    BcastGData<NDim, DType>, UDF>(
        rtcfg, csr, gdata, IntArray1D());
}

#define GEN_DEFINE(dtype, lhs_tgt, rhs_tgt, op)                    \
  template void CallBinaryReduce<dtype,                            \
                                 lhs_tgt, rhs_tgt,                 \
                                 op<dtype>, REDUCER<XPU, dtype>>(  \
      const minigun::advance::RuntimeConfig& rtcfg,                \
      const minigun::Csr& csr,                                     \
      const minigun::Csr& rev_csr,                                 \
      GData<dtype>* gdata);

#define GEN_BCAST_DEFINE(ndim, dtype, lhs_tgt, rhs_tgt, op)              \
  template void CallBinaryReduceBcast<ndim, dtype,                       \
                                 lhs_tgt, rhs_tgt,                       \
                                 op<dtype>, REDUCER<XPU, dtype>>(        \
      const minigun::advance::RuntimeConfig& rtcfg,                      \
      const minigun::Csr& csr,                                           \
      const minigun::Csr& rev_csr,                                       \
      BcastGData<ndim, dtype>* gdata);

#define EVAL(F, ...) F(__VA_ARGS__)

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_CUH_
