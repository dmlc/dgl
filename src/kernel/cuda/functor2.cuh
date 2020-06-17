#ifndef DGL_KERNEL2_FUNCTOR_CUH_
#define DGL_KERNEL2_FUNCTOR_CUH_

namespace dgl {
namespace kernel {
namespace cuda {

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

template <typename DType>
struct CopyU {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return lhs[0];
  }
};

template <typename DType>
struct CopyE {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(
      const DType *lhs, const DType *rhs, int64_t len = 1) {
    return rhs[0];
  }
};

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


}   // end of namespace binary

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
  static __device__ __forceinline__ void CallArg(Idx fid,
    Idx *arg_u_buf, Idx *arg_e_buf,
    DType val, DType val_ref, Idx uid, Idx eid) {}
};

template <typename Idx,
          typename DType,
          bool atomic=false>
struct Max {
  static constexpr DType zero = std::numeric_limits<DType>::lowest();
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

template <typename Idx,
          typename DType,
          bool atomic=false>
struct Min {
  static constexpr DType zero = std::numeric_limits<DType>::max();
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

}   // end of namespace reduce

}
}
}

#endif  // DGL_KERNEL2_FUNCTOR_CUH_
