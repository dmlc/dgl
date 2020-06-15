#ifndef DGL_KERNEL_CPU_SPMM_CUH_
#define DGL_KERNEL_CPU_SPMM_CUH_

#include "../utils.h"
#include <dgl/array.h>
#include "../binary_reduce.h"

namespace dgl {
namespace kernel {
namespace cpu {

template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out) {
  const bool has_idx = !aten::IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  DType* O = static_cast<DType*>(out->data);
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      DType accum = 0;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx? edges[j] : j;
        const DType* lhs_off = Op::use_lhs? X + cid * dim + k : nullptr;
        const DType* rhs_off = Op::use_rhs? W + eid * dim + k : nullptr;
        accum += Op::Call(lhs_off, rhs_off);
      }
      out_off[k] = accum;
    }
  }
}

template <typename IdType, typename DType, typename Op>
void SpMMSumCoo(const aten::COOMatrix& coo,
                NDArray ufeat, NDArray efeat,
                NDArray out) {
  const bool has_idx = !aten::IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  DType* O = static_cast<DType*>(out->data);
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
      const DType* lhs_off = Op::use_lhs? X + rid * dim + k : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * dim + k : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
#pragma omp atomic
      out_off[k] += val;
    }
  }
}

template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsr(const aten::CSRMatrix& csr,
                NDArray ufeat, NDArray efeat,
                NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !aten::IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    IdType* argx_off = argX + rid * dim;
    IdType* argw_off = argW + rid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      DType accum = Cmp::zero;
      IdType ax = 0, aw = 0;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx? edges[j] : j;
        const DType* lhs_off = Op::use_lhs? X + cid * dim + k : nullptr;
        const DType* rhs_off = Op::use_rhs? W + eid * dim + k : nullptr;
        const DType val = Op::Call(lhs_off, rhs_off);
        if (Cmp::Call(accum, val)) {
          accum = val;
          if (Op::use_lhs)
            ax = cid;
          if (Op::use_rhs)
            aw = eid;
        }
      }
      out_off[k] = accum;
      if (Op::use_lhs)
        argx_off[k] = ax;
      if (Op::use_rhs)
        argw_off[k] = aw;
    }
  }
}

template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCoo(const aten::COOMatrix& coo,
                NDArray ufeat, NDArray efeat,
                NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !aten::IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
#pragma omp parallel for
  for (IdType i = 0; i < out.Numel(); ++i) {
    O[i] = Cmp::zero;
  }
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
      const DType* lhs_off = Op::use_lhs? X + rid * dim + k : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * dim + k : nullptr;
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

template <typename IdType, typename DType, typename Op>
void SpMMBcastSumCsr(
    const BcastInfo& info,
    const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !aten::IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = utils::Prod(info.out_shape),
          lhs_dim = utils::Prod(info.lhs_shape),
          rhs_dim = utils::Prod(info.rhs_shape);
  DType* O = static_cast<DType*>(out->data);
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      DType accum = 0;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx? edges[j] : j;
        const DType* lhs_off = Op::use_lhs? X + cid * lhs_dim + info.lhs_offset[k] : nullptr;
        const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + info.rhs_offset[k] : nullptr;
        CHECK_LT(cid * lhs_dim + info.lhs_offset[k], ufeat.Numel());

        accum += Op::Call(lhs_off, rhs_off);
      }
      out_off[k] = accum;
    }
  }
}

template <typename IdType, typename DType, typename Op>
void SpMMBcastSumCoo(
    const BcastInfo& info,
    const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !aten::IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = utils::Prod(info.out_shape),
          lhs_dim = utils::Prod(info.lhs_shape),
          rhs_dim = utils::Prod(info.rhs_shape);
  DType* O = static_cast<DType*>(out->data);
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
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + info.lhs_offset[k] : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + info.rhs_offset[k] : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
#pragma omp atomic
      out_off[k] += val;
    }
  }
}

template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMBcastCmpCsr(
    const BcastInfo& info,
    const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !aten::IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges = has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = utils::Prod(info.out_shape),
          lhs_dim = utils::Prod(info.lhs_shape),
          rhs_dim = utils::Prod(info.rhs_shape);
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    DType* out_off = O + rid * dim;
    IdType* argx_off = argX + rid * dim;
    IdType* argw_off = argW + rid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      DType accum = Cmp::zero;
      IdType ax = 0, aw = 0;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx? edges[j] : j;
        const DType* lhs_off = Op::use_lhs? X + cid * lhs_dim + info.lhs_offset[k] : nullptr;
        const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + info.rhs_offset[k] : nullptr;
        const DType val = Op::Call(lhs_off, rhs_off);
        if (Cmp::Call(accum, val)) {
          accum = val;
          if (Op::use_lhs)
            ax = cid;
          if (Op::use_rhs)
            aw = eid;
        }
      }
      out_off[k] = accum;
      if (Op::use_lhs)
        argx_off[k] = ax;
      if (Op::use_rhs)
        argw_off[k] = aw;
    }
  }
}

template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMBcastCmpCoo(
    const BcastInfo& info,
    const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !aten::IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges = has_idx? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs? static_cast<DType*>(efeat->data) : nullptr;
  int64_t dim = utils::Prod(info.out_shape),
          lhs_dim = utils::Prod(info.lhs_shape),
          rhs_dim = utils::Prod(info.rhs_shape);
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs? static_cast<IdType*>(arge->data) : nullptr;
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
#pragma omp parallel for
  for (IdType i = 0; i < out.Numel(); ++i) {
    O[i] = Cmp::zero;
  }
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
      const DType* lhs_off = Op::use_lhs? X + rid * lhs_dim + info.lhs_offset[k] : nullptr;
      const DType* rhs_off = Op::use_rhs? W + eid * rhs_dim + info.rhs_offset[k] : nullptr;
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
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off + *rhs_off;
  }
};

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off) {
    return *lhs_off * *rhs_off;
  }
};

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType* ) {
    return *lhs_off;
  }
};

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* , const DType* rhs_off) {
    return *rhs_off;
  }
};

template <typename DType>
struct Max {
  static constexpr DType zero = std::numeric_limits<DType>::lowest();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum < val;
  }
};

template <typename DType>
struct Min {
  static constexpr DType zero = std::numeric_limits<DType>::max();
  // return true if accum should be replaced
  inline static DType Call(DType accum, DType val) {
    return accum > val;
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
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM binary operator: " << op;     \
    }                                                               \
  } while (0)

}  // namespace op

}  // namespace cpu
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_SPMM_CUH_
