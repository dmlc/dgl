/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/functor.cuh
 * @brief Functors for template on CUDA
 */
#ifndef DGL_ARRAY_CUDA_FUNCTOR_CUH_
#define DGL_ARRAY_CUDA_FUNCTOR_CUH_

#include <cmath>
#include <limits>

#include "./atomic.cuh"
#include "./fp16.cuh"
#include "bf16.cuh"

namespace dgl {
namespace aten {
namespace cuda {

/////////////////////////// CUDA binary operators //////////////////////////////
namespace binary {
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] + rhs[0];
  }
};
template <typename DType>
constexpr bool Add<DType>::use_lhs;
template <typename DType>
constexpr bool Add<DType>::use_rhs;
template <typename DType>
constexpr bool Add<DType>::reduce_last_dim;

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] - rhs[0];
  }
};
template <typename DType>
constexpr bool Sub<DType>::use_lhs;
template <typename DType>
constexpr bool Sub<DType>::use_rhs;
template <typename DType>
constexpr bool Sub<DType>::reduce_last_dim;

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] * rhs[0];
  }
};
template <typename DType>
constexpr bool Mul<DType>::use_lhs;
template <typename DType>
constexpr bool Mul<DType>::use_rhs;
template <typename DType>
constexpr bool Mul<DType>::reduce_last_dim;

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] / rhs[0];
  }
};
template <typename DType>
constexpr bool Div<DType>::use_lhs;
template <typename DType>
constexpr bool Div<DType>::use_rhs;
template <typename DType>
constexpr bool Div<DType>::reduce_last_dim;

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0];
  }
};
template <typename DType>
constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyLhs<DType>::use_rhs;
template <typename DType>
constexpr bool CopyLhs<DType>::reduce_last_dim;

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    return rhs[0];
  }
};
template <typename DType>
constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyRhs<DType>::use_rhs;
template <typename DType>
constexpr bool CopyRhs<DType>::reduce_last_dim;

template <typename DType>
struct Dot {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = true;
  static __device__ __forceinline__ DType
  Call(const DType *lhs, const DType *rhs, int64_t len = 1) {
    DType rst = static_cast<DType>(0.0f);
    for (int64_t i = 0; i < len; ++i) {
      rst += lhs[i] * rhs[i];
    }
    return rst;
  }
};
template <typename DType>
constexpr bool Dot<DType>::use_lhs;
template <typename DType>
constexpr bool Dot<DType>::use_rhs;
template <typename DType>
constexpr bool Dot<DType>::reduce_last_dim;

}  // end of namespace binary

/////////////////////////// CUDA reduce operators //////////////////////////////
namespace reduce {
template <typename Idx, typename DType, bool atomic>
struct _Sum {
  static constexpr __host__ __device__ __forceinline__ DType zero() {
    return 0.;
  }
  static constexpr bool require_arg = false;
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf, DType val, Idx uid,
      Idx eid) {
    if (!atomic) {
      *out_buf += val;
    } else {
      cuda::AtomicAdd(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_buf, DType val, Idx id) {
    if (!atomic) {
      *out_buf += val;
    } else {
      cuda::AtomicAdd(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(
      Idx fid, Idx *arg_u_buf, Idx *arg_e_buf, DType val, DType val_ref,
      Idx uid, Idx eid) {}
};

template <typename Idx, typename DType, bool atomic = false>
struct Sum : _Sum<Idx, DType, atomic> {};

template <typename Idx, bool atomic>
struct Sum<Idx, __half, atomic> : _Sum<Idx, __half, atomic> {
  static constexpr __host__ __device__ __forceinline__ __half zero() {
    return __float2half_rn(0.);
  }
  static __device__ __forceinline__ void Call(
      __half *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __half val, Idx uid, Idx eid) {
    _Sum<Idx, __half, atomic>::Call(
        out_buf, arg_u_buf, arg_e_buf, val, uid, eid);
  }
  static __device__ __forceinline__ void Call(
      __half *out_buf, Idx *arg_buf, __half val, Idx id) {
    _Sum<Idx, __half, atomic>::Call(out_buf, arg_buf, val, id);
  }
  // sometimes we have to use float in reduction for better precision
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __half val, Idx uid, Idx eid) {
    _Sum<Idx, float, atomic>::Call(out_buf, arg_u_buf, arg_e_buf,
        static_cast<float>(val), uid, eid);
  }
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_buf, __half val, Idx id) {
    _Sum<Idx, float, atomic>::Call(out_buf, arg_buf,
        static_cast<float>(val), id);
  }
};

#if BF16_ENABLED
template <typename Idx, bool atomic>
struct Sum<Idx, __nv_bfloat16, atomic> : _Sum<Idx, __nv_bfloat16, atomic> {
  static constexpr __host__ __device__ __forceinline__ __nv_bfloat16 zero() {
    return __float2bfloat16_rn(0.);
  }
  static __device__ __forceinline__ void Call(
      __nv_bfloat16 *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __nv_bfloat16 val, Idx uid, Idx eid) {
    _Sum<Idx, __nv_bfloat16, atomic>::Call(
        out_buf, arg_u_buf, arg_e_buf, val, uid, eid);
  }
  static __device__ __forceinline__ void Call(
      __nv_bfloat16 *out_buf, Idx *arg_buf, __nv_bfloat16 val, Idx id) {
    _Sum<Idx, __nv_bfloat16, atomic>::Call(out_buf, arg_buf, val, id);
  }
  // sometimes we have to use float in reduction for better precision
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __nv_bfloat16 val, Idx uid, Idx eid) {
    _Sum<Idx, float, atomic>::Call(out_buf, arg_u_buf, arg_e_buf,
        static_cast<float>(val), uid, eid);
  }
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_buf, __nv_bfloat16 val, Idx id) {
    _Sum<Idx, float, atomic>::Call(out_buf, arg_buf,
        static_cast<float>(val), id);
  }
};
#endif  // BF16_ENABLED

template <typename Idx, typename DType, bool atomic>
struct _Max {
  static constexpr __host__ __device__ __forceinline__ DType zero() {
    return -std::numeric_limits<DType>::infinity();
  }
  static constexpr bool require_arg = true;
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf, DType val, Idx uid,
      Idx eid) {
    if (!atomic) {
      if (*out_buf < val) {
        *out_buf = val;
        *arg_u_buf = uid;
        *arg_e_buf = eid;
      }
    } else {
      cuda::AtomicMax(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_buf, DType val, Idx id) {
    if (!atomic) {
      if (*out_buf < val) {
        *out_buf = val;
        *arg_buf = id;
      }
    } else {
      cuda::AtomicMax(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(
      Idx fid, Idx *arg_u_buf, Idx *arg_e_buf, DType val, DType val_ref,
      Idx uid, Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf) arg_u_buf[fid] = uid;
        if (arg_e_buf) arg_e_buf[fid] = eid;
      }
    }
  }
};

template <typename Idx, typename DType, bool atomic = false>
struct Max : _Max<Idx, DType, atomic> {};

template <typename Idx, bool atomic>
struct Max<Idx, __half, atomic> : _Max<Idx, __half, atomic> {
  static constexpr __host__ __device__ __forceinline__ __half zero() {
    return __float2half_rn(-6.550400e+04f);
  }
  static __device__ __forceinline__ void Call(
      __half *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __half val, Idx uid, Idx eid) {
    _Max<Idx, __half, atomic>::Call(
        out_buf, arg_u_buf, arg_e_buf, val, uid, eid);
  }
  static __device__ __forceinline__ void Call(
      __half *out_buf, Idx *arg_buf, __half val, Idx id) {
    _Max<Idx, __half, atomic>::Call(out_buf, arg_buf, val, id);
  }
  // sometimes we have to use float in reduction for better precision
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __half val, Idx uid, Idx eid) {
    _Max<Idx, float, atomic>::Call(out_buf, arg_u_buf, arg_e_buf,
        static_cast<float>(val), uid, eid);
  }
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_buf, __half val, Idx id) {
    _Max<Idx, float, atomic>::Call(out_buf, arg_buf,
        static_cast<float>(val), id);
  }
};

#if BF16_ENABLED
template <typename Idx, bool atomic>
struct Max<Idx, __nv_bfloat16, atomic> : _Max<Idx, __nv_bfloat16, atomic> {
  static constexpr __host__ __device__ __forceinline__ __nv_bfloat16 zero() {
    return __float2bfloat16_rn(-std::numeric_limits<float>::infinity());
  }
  static __device__ __forceinline__ void Call(
      __nv_bfloat16 *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __nv_bfloat16 val, Idx uid, Idx eid) {
    _Max<Idx, __nv_bfloat16, atomic>::Call(
        out_buf, arg_u_buf, arg_e_buf, val, uid, eid);
  }
  static __device__ __forceinline__ void Call(
      __nv_bfloat16 *out_buf, Idx *arg_buf, __nv_bfloat16 val, Idx id) {
    _Max<Idx, __nv_bfloat16, atomic>::Call(out_buf, arg_buf, val, id);
  }
  // sometimes we have to use float in reduction for better precision
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __nv_bfloat16 val, Idx uid, Idx eid) {
    _Max<Idx, float, atomic>::Call(out_buf, arg_u_buf, arg_e_buf,
        static_cast<float>(val), uid, eid);
  }
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_buf, __nv_bfloat16 val, Idx id) {
    _Max<Idx, float, atomic>::Call(out_buf, arg_buf,
        static_cast<float>(val), id);
  }
};
#endif  // BF16_ENABLED

template <typename Idx, typename DType, bool atomic>
struct _Min {
  static constexpr __host__ __device__ __forceinline__ DType zero() {
    return std::numeric_limits<DType>::infinity();
  }
  static constexpr bool require_arg = true;
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf, DType val, Idx uid,
      Idx eid) {
    if (!atomic) {
      if (*out_buf > val) {
        *out_buf = val;
        *arg_u_buf = uid;
        *arg_e_buf = eid;
      }
    } else {
      cuda::AtomicMin(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_buf, DType val, Idx id) {
    if (!atomic) {
      if (*out_buf > val) {
        *out_buf = val;
        *arg_buf = id;
      }
    } else {
      cuda::AtomicMin(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(
      Idx fid, Idx *arg_u_buf, Idx *arg_e_buf, DType val, DType val_ref,
      Idx uid, Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf) arg_u_buf[fid] = uid;
        if (arg_e_buf) arg_e_buf[fid] = eid;
      }
    }
  }
};

template <typename Idx, typename DType, bool atomic = false>
struct Min : _Min<Idx, DType, atomic> {};

template <typename Idx, bool atomic>
struct Min<Idx, __half, atomic> : _Min<Idx, __half, atomic> {
  static constexpr __host__ __device__ __forceinline__ __half zero() {
    return __float2half_rn(6.550400e+04f);
  }
  static __device__ __forceinline__ void Call(
      __half *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __half val, Idx uid, Idx eid) {
    _Min<Idx, __half, atomic>::Call(
        out_buf, arg_u_buf, arg_e_buf, val, uid, eid);
  }
  static __device__ __forceinline__ void Call(
      __half *out_buf, Idx *arg_buf, __half val, Idx id) {
    _Min<Idx, __half, atomic>::Call(out_buf, arg_buf, val, id);
  }
  // sometimes we have to use float in reduction for better precision
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __half val, Idx uid, Idx eid) {
    _Min<Idx, float, atomic>::Call(out_buf, arg_u_buf, arg_e_buf,
        static_cast<float>(val), uid, eid);
  }
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_buf, __half val, Idx id) {
    _Min<Idx, float, atomic>::Call(out_buf, arg_buf,
        static_cast<float>(val), id);
  }
};

#if BF16_ENABLED
template <typename Idx, bool atomic>
struct Min<Idx, __nv_bfloat16, atomic> : _Min<Idx, __nv_bfloat16, atomic> {
  static constexpr __host__ __device__ __forceinline__ __nv_bfloat16 zero() {
    return __float2bfloat16_rn(std::numeric_limits<float>::infinity());
  }
  static __device__ __forceinline__ void Call(
      __nv_bfloat16 *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __nv_bfloat16 val, Idx uid, Idx eid) {
    _Min<Idx, __nv_bfloat16, atomic>::Call(
        out_buf, arg_u_buf, arg_e_buf, val, uid, eid);
  }
  static __device__ __forceinline__ void Call(
      __nv_bfloat16 *out_buf, Idx *arg_buf, __nv_bfloat16 val, Idx id) {
    _Min<Idx, __nv_bfloat16, atomic>::Call(out_buf, arg_buf, val, id);
  }
  // sometimes we have to use float in reduction for better precision
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
      __nv_bfloat16 val, Idx uid, Idx eid) {
    _Min<Idx, float, atomic>::Call(out_buf, arg_u_buf, arg_e_buf,
        static_cast<float>(val), uid, eid);
  }
  static __device__ __forceinline__ void Call(
      float *out_buf, Idx *arg_buf, __nv_bfloat16 val, Idx id) {
    _Min<Idx, float, atomic>::Call(out_buf, arg_buf,
        static_cast<float>(val), id);
  }
};
#endif  // BF16_ENABLED

}  // namespace reduce

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_FUNCTOR_CUH_
