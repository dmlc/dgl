/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/sddmm.h
 * \brief SDDMM CPU kernel function header.
 */
#ifndef DGL_KERNEL_CPU_SDDMM_CUH_
#define DGL_KERNEL_CPU_SDDMM_CUH_

#include "../utils.h"
#include "../bcast.h"
#include <dgl/array.h>

namespace dgl {
namespace kernel {
namespace cpu {

template <typename IdType, typename DType, typename Op>
void SDDMMCsr(const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray ufeat, NDArray vfeat, NDArray out) {
  const bool has_idx = !aten::IsNullArray(csr.data);
  const IdType* indptr = utils::GetPtr<IdType>(csr.indptr);
  const IdType* indices = utils::GetPtr<IdType>(csr.indices);
  const IdType* edges = utils::GetPtr<IdType>(csr.data);
  const DType* X = utils::GetPtr<DType>(ufeat);
  const DType* Y = utils::GetPtr<DType>(vfeat);
  int64_t dim = bcast.out_len,
          lhs_dim = bcast.lhs_len,
          rhs_dim = bcast.rhs_len;
  DType* O = utils::GetPtr<DType>(out);
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx? edges[j] : j;
      DType* out_off = O + eid * dim;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + lhs_add : nullptr;
        const DType* rhs_off = Op::use_rhs? Y + cid * rhs_dim + rhs_add : nullptr;
        out_off[k] = Op::Call(lhs_off, rhs_off, bcast.reduce_size);
      }
    }
  }
}

template <typename IdType, typename DType, typename Op>
void SDDMMCoo(const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray ufeat, NDArray vfeat, NDArray out) {
  const bool has_idx = !aten::IsNullArray(coo.data);
  const IdType* row = utils::GetPtr<IdType>(coo.row);
  const IdType* col = utils::GetPtr<IdType>(coo.col);
  const IdType* edges = utils::GetPtr<IdType>(coo.data);
  const DType* X = utils::GetPtr<DType>(ufeat);
  const DType* Y = utils::GetPtr<DType>(vfeat);
  int64_t dim = bcast.out_len,
          lhs_dim = bcast.lhs_len,
          rhs_dim = bcast.rhs_len;
  DType* O = utils::GetPtr<DType>(out);
  const int64_t nnz = coo.row->shape[0];
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx? edges[i] : i;
    DType* out_off = O + eid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off = Op::use_rhs? Y + cid * rhs_dim + rhs_add : nullptr;
      out_off[k] = Op::Call(lhs_off, rhs_off, bcast.reduce_size);
    }
  }
}

namespace op {
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off + *rhs_off;
  }
};

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off * *rhs_off;
  }
};

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType*, int64_t len = 1) {
    return *lhs_off;
  }
};

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* , const DType* rhs_off, int64_t len = 1) {
    return *rhs_off;
  }
};

template <typename DType>
struct Dot {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    DType rst = 0;
    for (int64_t l = 0; l < len; ++l) {
      rst += lhs_off[l] * rhs_off[l]; 
    } 
    return rst;
  }
};

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef dgl::kernel::cpu::op::Add<DType> Op;                  \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef dgl::kernel::cpu::op::Mul<DType> Op;                  \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_u") {                                  \
      typedef dgl::kernel::cpu::op::CopyLhs<DType> Op;              \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_e") {                                  \
      typedef dgl::kernel::cpu::op::CopyRhs<DType> Op;              \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "dot") {                                     \
      typedef dgl::kernel::cpu::op::Dot<DType> Op;                  \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SDDMM binary operator: " << op;     \
    }                                                               \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_SDDMM_CUH_
