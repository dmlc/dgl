/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/spmm.h
 * \brief SPMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SPMM_H_
#define DGL_ARRAY_CPU_SPMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <limits>
#include <algorithm>

namespace dgl {
namespace aten {
namespace cpu {

/*!
 * \brief CPU kernel of SpMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  int64_t dim = bcast.out_len,
          lhs_dim = bcast.lhs_len,
          rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType *out_off = O + rid * dim;
    std::fill(out_off, out_off + dim, 0);
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx ? edges[j] : j;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType *lhs_off =
            Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
        const DType *rhs_off =
            Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
        out_off[k] += Op::Call(lhs_off, rhs_off);
      }
    }
  }
}

/*!
 * \brief CPU kernel of SpMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes. To avoid possible data hazard,
 *       we use atomic operators in the reduction phase.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  int64_t dim = bcast.out_len,
          lhs_dim = bcast.lhs_len,
          rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  memset(O, 0, out.GetSize());
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx? edges[i] : i;
    DType* out_off = O + cid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
      if (val != 0) {
#pragma omp atomic
        out_off[k] += val;
      }
    }
  }
}

/*!
 * \brief CPU kernel of SpMM-Min/Max on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \note It uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 * \note The result will contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len,
                lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    IdType* argx_off = argX + rid * dim;
    IdType* argw_off = argW + rid * dim;
    std::fill(out_off, out_off + dim, Cmp::zero);
    if (Op::use_lhs)
      std::fill(argx_off, argx_off + dim, 0);
    if (Op::use_rhs)
      std::fill(argw_off, argw_off + dim, 0);
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx? edges[j] : j;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType* lhs_off = Op::use_lhs? X + cid * lhs_dim + lhs_add : nullptr;
        const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + rhs_add : nullptr;
        const DType val = Op::Call(lhs_off, rhs_off);
        if (Cmp::Call(out_off[k], val)) {
          out_off[k] = val;
          if (Op::use_lhs)
            argx_off[k] = cid;
          if (Op::use_rhs)
            argw_off[k] = eid;
        }
      }
    }
  }
}

/*!
 * \brief CPU kernel of SpMM-Min/Max on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices 
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes. To avoid possible data hazard,
 *       we use atomic operators in the reduction phase.
 * \note The result will contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len,
                lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  std::fill(O, O + out.NumElements(), Cmp::zero);
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx? edges[i] : i;
    DType* out_off = O + cid * dim;
    IdType* argx_off = Op::use_lhs? argX + cid * dim : nullptr;
    IdType* argw_off = Op::use_rhs? argW + cid * dim : nullptr;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
#pragma omp critical
      if (Cmp::Call(out_off[k], val)) {
        out_off[k] = val;
        if (Op::use_lhs)
          argx_off[k] = rid;
        if (Op::use_rhs)
          argw_off[k] = eid;
      }
    }
  }
}

namespace op {

//////////////////////////////// binary operators on CPU ////////////////////////////////
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off + *rhs_off;
  }
};
template <typename DType> constexpr bool Add<DType>::use_lhs;
template <typename DType> constexpr bool Add<DType>::use_rhs;

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off - *rhs_off;
  }
};
template <typename DType> constexpr bool Sub<DType>::use_lhs;
template <typename DType> constexpr bool Sub<DType>::use_rhs;

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off * *rhs_off;
  }
};
template <typename DType> constexpr bool Mul<DType>::use_lhs;
template <typename DType> constexpr bool Mul<DType>::use_rhs;

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off / *rhs_off;
  }
};
template <typename DType> constexpr bool Div<DType>::use_lhs;
template <typename DType> constexpr bool Div<DType>::use_rhs;

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType* ) {
    return *lhs_off;
  }
};
template <typename DType> constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType> constexpr bool CopyLhs<DType>::use_rhs;

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* , const DType* rhs_off) {
    return *rhs_off;
  }
};
template <typename DType> constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType> constexpr bool CopyRhs<DType>::use_rhs;

//////////////////////////////// Reduce operators on CPU ////////////////////////////////
template <typename DType>
struct Max {
  static constexpr DType zero = -std::numeric_limits<DType>::infinity();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum < val;
  }
};
template <typename DType> constexpr DType Max<DType>::zero;

template <typename DType>
struct Min {
  static constexpr DType zero = std::numeric_limits<DType>::infinity();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum > val;
  }
};
template <typename DType> constexpr DType Min<DType>::zero;

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef dgl::aten::cpu::op::Add<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef dgl::aten::cpu::op::Sub<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef dgl::aten::cpu::op::Mul<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef dgl::aten::cpu::op::Div<DType> Op;                    \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_lhs") {                                \
      typedef dgl::aten::cpu::op::CopyLhs<DType> Op;                \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_rhs") {                                \
      typedef dgl::aten::cpu::op::CopyRhs<DType> Op;                \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPMM_H_
