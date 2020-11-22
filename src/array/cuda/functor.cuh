/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/functor.cuh
 * \brief Functors for template on CUDA
 */
#ifndef DGL_ARRAY_CUDA_FUNCTOR_CUH_
#define DGL_ARRAY_CUDA_FUNCTOR_CUH_

#include "./atomic.cuh"

namespace dgl {
namespace aten {
namespace cuda {

/////////////////////////////// CUDA binary operators ///////////////////////////////
namespace binary {
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] + rhs[0];
  }
};
template <typename DType> constexpr bool Add<DType>::use_lhs;
template <typename DType> constexpr bool Add<DType>::use_rhs;
template <typename DType> constexpr bool Add<DType>::reduce_last_dim;

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] - rhs[0];
  }
};
template <typename DType> constexpr bool Sub<DType>::use_lhs;
template <typename DType> constexpr bool Sub<DType>::use_rhs;
template <typename DType> constexpr bool Sub<DType>::reduce_last_dim;

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] * rhs[0];
  }
};
template <typename DType> constexpr bool Mul<DType>::use_lhs;
template <typename DType> constexpr bool Mul<DType>::use_rhs;
template <typename DType> constexpr bool Mul<DType>::reduce_last_dim;

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0] / rhs[0];
  }
};
template <typename DType> constexpr bool Div<DType>::use_lhs;
template <typename DType> constexpr bool Div<DType>::use_rhs;
template <typename DType> constexpr bool Div<DType>::reduce_last_dim;

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0];
  }
};
template <typename DType> constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType> constexpr bool CopyLhs<DType>::use_rhs;
template <typename DType> constexpr bool CopyLhs<DType>::reduce_last_dim;

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return rhs[0];
  }
};
template <typename DType> constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType> constexpr bool CopyRhs<DType>::use_rhs;
template <typename DType> constexpr bool CopyRhs<DType>::reduce_last_dim;

template <typename DType>
struct Dot {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = true;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    DType rst = static_cast<DType>(0);
    for (int64_t i = 0; i < len; ++i) {
      rst += lhs[i] * rhs[i];
    }
    return rst;
  }
};
template <typename DType> constexpr bool Dot<DType>::use_lhs;
template <typename DType> constexpr bool Dot<DType>::use_rhs;
template <typename DType> constexpr bool Dot<DType>::reduce_last_dim;

}   // end of namespace binary

/////////////////////////////// CUDA reduce operators ///////////////////////////////
namespace reduce {
template <typename Idx,
          typename DType,
          bool atomic=false>
struct Sum {
  static constexpr DType zero = 0;
  static constexpr bool require_arg = false;
  static __device__ __forceinline__ void Call(
    DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, Idx uid, Idx eid) {
    if (!atomic) {
      *out_buf += val;
    } else {
      cuda::AtomicAdd(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(
      DType *out_buf, Idx *arg_buf,
      DType val, Idx id) {
    if (!atomic) {
      *out_buf += val;
    } else {
      cuda::AtomicAdd(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(Idx fid,
    Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {}
};
template <typename Idx, typename DType, bool atomic>
constexpr DType Sum<Idx, DType, atomic>::zero;
template <typename Idx, typename DType, bool atomic>
constexpr bool Sum<Idx, DType, atomic>::require_arg;

template <typename Idx,
          typename DType,
          bool atomic=false>
struct Max {
  static constexpr DType zero = -std::numeric_limits<DType>::infinity();
  static constexpr bool require_arg = true;
  static __device__ __forceinline__ void Call(
    DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, Idx uid, Idx eid) {
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
      DType *out_buf, Idx *arg_buf,
      DType val, Idx id) {
    if (!atomic) {
      if (*out_buf < val) {
        *out_buf = val;
        *arg_buf = id;
      }
    } else {
      cuda::AtomicMax(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(Idx fid,
    Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf)
          arg_u_buf[fid] = uid;
        if (arg_e_buf)
          arg_e_buf[fid] = eid;
      }
    }
  }
};
template <typename Idx, typename DType, bool atomic>
constexpr DType Max<Idx, DType, atomic>::zero;
template <typename Idx, typename DType, bool atomic>
constexpr bool Max<Idx, DType, atomic>::require_arg;

template <typename Idx,
          typename DType,
          bool atomic=false>
struct Min {
  static constexpr DType zero = std::numeric_limits<DType>::infinity();
  static constexpr bool require_arg = true;
  static __device__ __forceinline__ void Call(
    DType *out_buf, Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, Idx uid, Idx eid) {
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
      DType *out_buf, Idx *arg_buf,
      DType val, Idx id) {
    if (!atomic) {
      if (*out_buf > val) {
        *out_buf = val;
        *arg_buf = id;
      }
    } else {
      cuda::AtomicMin(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(Idx fid,
    Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf)
          arg_u_buf[fid] = uid;
        if (arg_e_buf)
          arg_e_buf[fid] = eid;
      }
    }
  }
};
template <typename Idx, typename DType, bool atomic>
constexpr DType Min<Idx, DType, atomic>::zero;
template <typename Idx, typename DType, bool atomic>
constexpr bool Min<Idx, DType, atomic>::require_arg;

}  // namespace reduce

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CUDA_FUNCTOR_CUH_
