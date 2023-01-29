/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/sddmm.h
 * @brief SDDMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SDDMM_H_
#define DGL_ARRAY_CPU_SDDMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dgl/runtime/parallel_for.h>

#include "../selector.h"

namespace dgl {
namespace aten {
namespace cpu {

/**
 * @brief CPU kernel of g-SDDMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 * @note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <
    typename IdType, typename DType, typename Op, int LhsTarget = 0,
    int RhsTarget = 2>
void SDDMMCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray lhs, NDArray rhs,
    NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = lhs.Ptr<DType>();
  const DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
  DType* O = out.Ptr<DType>();
  runtime::parallel_for(0, csr.num_rows, [=](IdType b, IdType e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx ? edges[j] : j;
        DType* out_off = O + eid * dim;
        for (int64_t k = 0; k < dim; ++k) {
          const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* lhs_off =
              Op::use_lhs
                  ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim +
                        lhs_add * reduce_size
                  : nullptr;
          const DType* rhs_off =
              Op::use_rhs
                  ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim +
                        rhs_add * reduce_size
                  : nullptr;
          out_off[k] = Op::Call(lhs_off, rhs_off, reduce_size);
        }
      }
    }
  });
}

/**
 * @brief CPU kernel of g-SDDMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The COO matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 * @note it uses edge parallel strategy, different threads are responsible
 *       for the computation of different edges.
 */
template <
    typename IdType, typename DType, typename Op, int LhsTarget = 0,
    int RhsTarget = 2>
void SDDMMCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray lhs, NDArray rhs,
    NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  const DType* X = lhs.Ptr<DType>();
  const DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
  DType* O = out.Ptr<DType>();
#pragma omp parallel for
  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx ? edges[i] : i;
    DType* out_off = O + eid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off =
          Op::use_lhs ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim +
                            lhs_add * reduce_size
                      : nullptr;
      const DType* rhs_off =
          Op::use_rhs ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim +
                            rhs_add * reduce_size
                      : nullptr;
      out_off[k] = Op::Call(lhs_off, rhs_off, bcast.reduce_size);
    }
  }
}

namespace op {

////////////////////////// binary operators on CPU /////////////////////////////
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off + *rhs_off;
  }
};

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off - *rhs_off;
  }
};

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off * *rhs_off;
  }
};

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    return *lhs_off / *rhs_off;
  }
};

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(
      const DType* lhs_off, const DType*, int64_t len = 1) {
    return *lhs_off;
  }
};

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType*, const DType* rhs_off, int64_t len = 1) {
    return *rhs_off;
  }
};

template <typename DType>
struct Dot {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(
      const DType* lhs_off, const DType* rhs_off, int64_t len = 1) {
    DType rst = 0;
    for (int64_t l = 0; l < len; ++l) {
      rst += lhs_off[l] * rhs_off[l];
    }
    return rst;
  }
};

#define SWITCH_OP(op, Op, ...)                                   \
  do {                                                           \
    if ((op) == "add") {                                         \
      typedef dgl::aten::cpu::op::Add<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "sub") {                                  \
      typedef dgl::aten::cpu::op::Sub<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "mul") {                                  \
      typedef dgl::aten::cpu::op::Mul<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "div") {                                  \
      typedef dgl::aten::cpu::op::Div<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "copy_lhs") {                             \
      typedef dgl::aten::cpu::op::CopyLhs<DType> Op;             \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "copy_rhs") {                             \
      typedef dgl::aten::cpu::op::CopyRhs<DType> Op;             \
      { __VA_ARGS__ }                                            \
    } else if ((op) == "dot") {                                  \
      typedef dgl::aten::cpu::op::Dot<DType> Op;                 \
      { __VA_ARGS__ }                                            \
    } else {                                                     \
      LOG(FATAL) << "Unsupported SDDMM binary operator: " << op; \
    }                                                            \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SDDMM_H_
