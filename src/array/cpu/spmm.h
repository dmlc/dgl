/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/spmm.h
 * @brief SPMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SPMM_H_
#define DGL_ARRAY_CPU_SPMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dgl/runtime/config.h>
#include <dgl/runtime/parallel_for.h>
#include <math.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "spmm_binary_ops.h"
#if !defined(_WIN32)
#ifdef USE_LIBXSMM
#include "spmm_blocking_libxsmm.h"
#endif  // USE_LIBXSMM
#endif  // _WIN32
namespace dgl {
namespace aten {
namespace cpu {

template <typename DType>
using AccType = typename std::conditional<
    std::is_same<DType, BFloat16>::value, float, DType>::type;

/**
 * @brief Naive CPU kernel of SpMM on Csr format.
 * @param cpu_spec JIT'ed kernel
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param X The feature on source nodes.
 * @param W The feature on edges.
 * @param O The result feature on destination nodes.
 * @note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <typename IdType, typename DType, typename Op>
typename std::enable_if<!std::is_same<DType, BFloat16>::value, void>::type
SpMMSumCsrNaive(
    const BcastOff& bcast, const CSRMatrix& csr, const DType* X, const DType* W,
    DType* O) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;
  runtime::parallel_for(0, csr.num_rows, [&](size_t b, size_t e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      DType* out_off = O + rid * dim;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx ? edges[j] : j;
        for (int64_t k = 0; k < dim; ++k) {
          const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* lhs_off =
              Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
          const DType* rhs_off =
              Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
          out_off[k] += Op::Call(lhs_off, rhs_off);
        }
      }
    }
  });
}

// Naive implementation with additional accumulator, which prevents accuracy
// degradation in less precise data types, like bfloat16.
template <typename IdType, typename DType, typename Op>
typename std::enable_if<std::is_same<DType, BFloat16>::value, void>::type
SpMMSumCsrNaive(
    const BcastOff& bcast, const CSRMatrix& csr, const DType* X, const DType* W,
    DType* O) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;
  runtime::parallel_for(0, csr.num_rows, [&](size_t b, size_t e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      DType* out_off = O + rid * dim;
      for (int64_t k = 0; k < dim; ++k) {
        AccType<DType> acc = 0.;
        for (IdType j = row_start; j < row_end; ++j) {
          const IdType cid = indices[j];
          const IdType eid = has_idx ? edges[j] : j;
          const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* lhs_off =
              Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
          const DType* rhs_off =
              Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
          acc += Op::Call(lhs_off, rhs_off);
        }
        out_off[k] += acc;
      }
    }
  });
}

/**
 * @brief CPU kernel of SpMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <typename IdType, typename DType, typename Op>
void SpMMSumCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  DType* O = out.Ptr<DType>();
  CHECK_NOTNULL(indptr);
  CHECK_NOTNULL(O);
  if (Op::use_lhs) {
    CHECK_NOTNULL(indices);
    CHECK_NOTNULL(X);
  }
  if (Op::use_rhs) {
    if (has_idx) CHECK_NOTNULL(edges);
    CHECK_NOTNULL(W);
  }
#if !defined(_WIN32)
#ifdef USE_LIBXSMM
  int cpu_id = libxsmm_cpuid_x86();
  const bool no_libxsmm =
      bcast.use_bcast || std::is_same<DType, double>::value ||
      (std::is_same<DType, BFloat16>::value && cpu_id < LIBXSMM_X86_AVX512) ||
      !dgl::runtime::Config::Global()->IsLibxsmmAvailable();
  if (!no_libxsmm) {
    SpMMSumCsrLibxsmm<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
  } else {
#endif  // USE_LIBXSMM
#endif  // _WIN32
    SpMMSumCsrNaive<IdType, DType, Op>(bcast, csr, X, W, O);
#if !defined(_WIN32)
#ifdef USE_LIBXSMM
  }
#endif  // USE_LIBXSMM
#endif  // _WIN32
}

/**
 * @brief CPU kernel of SpMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes. To avoid possible data hazard,
 *       we use atomic operators in the reduction phase.
 */
template <typename IdType, typename DType, typename Op>
typename std::enable_if<!std::is_same<DType, BFloat16>::value, void>::type
SpMMSumCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  const DType* X = ufeat.Ptr<DType>();
  const DType* W = efeat.Ptr<DType>();
  int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len, rhs_dim = bcast.rhs_len;
  DType* O = out.Ptr<DType>();
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  memset(O, 0, out.GetSize());
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx ? edges[i] : i;
    DType* out_off = O + cid * dim;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off =
          Op::use_lhs ? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off =
          Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
      if (val != 0) {
#pragma omp atomic
        out_off[k] += val;
      }
    }
  }
}

template <typename IdType, typename DType, typename Op>
typename std::enable_if<std::is_same<DType, BFloat16>::value, void>::type
SpMMSumCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out) {
  LOG(FATAL) << "Unsupported CPU kernel for SpMMSumCoo for BF16.";
}

/**
 * @brief CPU kernel of SpMM-Min/Max on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 *        reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
          correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 *        reducer.
 * @note It uses node parallel strategy, different threads are responsible for
 *       the computation of different nodes.
 * @note The result will contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsr(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges =
      has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs ? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs ? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs ? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs ? static_cast<IdType*>(arge->data) : nullptr;
  CHECK_NOTNULL(indptr);
  CHECK_NOTNULL(O);
  if (Op::use_lhs) {
    CHECK_NOTNULL(indices);
    CHECK_NOTNULL(X);
    CHECK_NOTNULL(argX);
  }
  if (Op::use_rhs) {
    if (has_idx) CHECK_NOTNULL(edges);
    CHECK_NOTNULL(W);
    CHECK_NOTNULL(argW);
  }
#if !defined(_WIN32)
#ifdef USE_LIBXSMM
  int cpu_id = libxsmm_cpuid_x86();
  const bool no_libxsmm = bcast.use_bcast ||
                          std::is_same<DType, double>::value ||
                          cpu_id < LIBXSMM_X86_AVX512 ||
                          !dgl::runtime::Config::Global()->IsLibxsmmAvailable();
  if (!no_libxsmm) {
    SpMMCmpCsrLibxsmm<IdType, DType, Op, Cmp>(
        bcast, csr, ufeat, efeat, out, argu, arge);
  } else {
#endif  // USE_LIBXSMM
#endif  // _WIN32

    runtime::parallel_for(0, csr.num_rows, [&](size_t b, size_t e) {
      for (auto rid = b; rid < e; ++rid) {
        const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
        DType* out_off = O + rid * dim;
        IdType* argx_off = argX + rid * dim;
        IdType* argw_off = argW + rid * dim;
        for (IdType j = row_start; j < row_end; ++j) {
          const IdType cid = indices[j];
          const IdType eid = has_idx ? edges[j] : j;
          for (int64_t k = 0; k < dim; ++k) {
            const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
            const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
            const DType* lhs_off =
                Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
            const DType* rhs_off =
                Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
            const DType val = Op::Call(lhs_off, rhs_off);
            if (Cmp::Call(out_off[k], val)) {
              out_off[k] = val;
              if (Op::use_lhs) argx_off[k] = cid;
              if (Op::use_rhs) argw_off[k] = eid;
            }
          }
        }
      }
    });
#if !defined(_WIN32)
#ifdef USE_LIBXSMM
  }
#endif  // USE_LIBXSMM
#endif  // _WIN32
}

/**
 * @brief CPU kernel of SpMM-Min/Max on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 *        reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 *        reducer.
 * @param argu_ntype Node type of the arg-Min/Max on source nodes, which refers
 *        the source node types correspond to the minimum/maximum values of
 *        reduction result on destination nodes. It's useful in computing
 *        gradients of Min/Max reducer.
 * @param arge_etype Edge-type of the arg-Min/Max on edges. which refers the
 *        source node indices correspond to the minimum/maximum values of
 *        reduction result on destination nodes. It's useful in computing
 *        gradients of Min/Max reducer.
 * @param src_type Node type of the source nodes of an etype
 * @param etype Edge type
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCsrHetero(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge, NDArray argu_ntype,
    NDArray arge_etype, const int ntype, const int etype) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices = static_cast<IdType*>(csr.indices->data);
  const IdType* edges =
      has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* X = Op::use_lhs ? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs ? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs ? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs ? static_cast<IdType*>(arge->data) : nullptr;
  IdType* argX_ntype =
      Op::use_lhs ? static_cast<IdType*>(argu_ntype->data) : nullptr;
  IdType* argW_etype =
      Op::use_rhs ? static_cast<IdType*>(arge_etype->data) : nullptr;
  CHECK_NOTNULL(indptr);
  CHECK_NOTNULL(O);
  if (Op::use_lhs) {
    CHECK_NOTNULL(indices);
    CHECK_NOTNULL(X);
    CHECK_NOTNULL(argX);
  }
  if (Op::use_rhs) {
    if (has_idx) CHECK_NOTNULL(edges);
    CHECK_NOTNULL(W);
    CHECK_NOTNULL(argW);
  }
  // TODO(Israt): Use LIBXSMM. Homogeneous graph uses LIBXMM when enabled.
  runtime::parallel_for(0, csr.num_rows, [&](size_t b, size_t e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      DType* out_off = O + rid * dim;
      IdType* argx_off = argX + rid * dim;
      IdType* argw_off = argW + rid * dim;
      IdType* argx_ntype = argX_ntype + rid * dim;
      IdType* argw_etype = argW_etype + rid * dim;
      for (IdType j = row_start; j < row_end; ++j) {
        const IdType cid = indices[j];
        const IdType eid = has_idx ? edges[j] : j;
        for (int64_t k = 0; k < dim; ++k) {
          const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* lhs_off =
              Op::use_lhs ? X + cid * lhs_dim + lhs_add : nullptr;
          const DType* rhs_off =
              Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
          const DType val = Op::Call(lhs_off, rhs_off);
          if (Cmp::Call(out_off[k], val)) {
            out_off[k] = val;
            if (Op::use_lhs) {
              argx_off[k] = cid;
              argx_ntype[k] = ntype;
            }
            if (Op::use_rhs) {
              argw_off[k] = eid;
              argw_etype[k] = etype;
            }
          }
        }
      }
    }
  });
}

/**
 * @brief CPU kernel of SpMM-Min/Max on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 *        reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 *        reducer.
 * @note it uses node parallel strategy, different threads are responsible for
 *       the computation of different nodes. To avoid possible data hazard, we
 *       use atomic operators in the reduction phase.
 * @note The result will contain infinity for zero-degree nodes.
 */
template <typename IdType, typename DType, typename Op, typename Cmp>
void SpMMCmpCoo(
    const BcastOff& bcast, const COOMatrix& coo, NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = static_cast<IdType*>(coo.row->data);
  const IdType* col = static_cast<IdType*>(coo.col->data);
  const IdType* edges =
      has_idx ? static_cast<IdType*>(coo.data->data) : nullptr;
  const DType* X = Op::use_lhs ? static_cast<DType*>(ufeat->data) : nullptr;
  const DType* W = Op::use_rhs ? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len;
  DType* O = static_cast<DType*>(out->data);
  IdType* argX = Op::use_lhs ? static_cast<IdType*>(argu->data) : nullptr;
  IdType* argW = Op::use_rhs ? static_cast<IdType*>(arge->data) : nullptr;
  const int64_t nnz = coo.row->shape[0];
  // fill zero elements
  std::fill(O, O + out.NumElements(), Cmp::zero);
  // spmm
#pragma omp parallel for
  for (IdType i = 0; i < nnz; ++i) {
    const IdType rid = row[i];
    const IdType cid = col[i];
    const IdType eid = has_idx ? edges[i] : i;
    DType* out_off = O + cid * dim;
    IdType* argx_off = Op::use_lhs ? argX + cid * dim : nullptr;
    IdType* argw_off = Op::use_rhs ? argW + cid * dim : nullptr;
    for (int64_t k = 0; k < dim; ++k) {
      const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
      const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
      const DType* lhs_off =
          Op::use_lhs ? X + rid * lhs_dim + lhs_add : nullptr;
      const DType* rhs_off =
          Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
      const DType val = Op::Call(lhs_off, rhs_off);
#pragma omp critical
      if (Cmp::Call(out_off[k], val)) {
        out_off[k] = val;
        if (Op::use_lhs) argx_off[k] = rid;
        if (Op::use_rhs) argw_off[k] = eid;
      }
    }
  }
}

/**
 * @brief CPU kernel of Edge_softmax_csr_forward on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result of edge_softmax_forward.
 */
template <typename IdType, typename DType, typename Op>
void Edge_softmax_csr_forward(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray ufeat, NDArray efeat,
    NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* edges =
      has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* W = Op::use_rhs ? static_cast<DType*>(efeat->data) : nullptr;
  const int64_t dim = bcast.out_len, rhs_dim = bcast.rhs_len;
  runtime::parallel_for(0, csr.num_rows, [&](size_t b, size_t e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      std::vector<AccType<DType>> data_e(row_end - row_start, 0);
      std::vector<IdType> num(row_end - row_start, 0);
      for (int64_t k = 0; k < dim; ++k) {
        DType max_v = -std::numeric_limits<DType>::infinity();
        for (IdType j = row_start; j < row_end; ++j) {
          const IdType eid = has_idx ? edges[j] : j;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* rhs_off =
              Op::use_rhs ? W + eid * rhs_dim + rhs_add : nullptr;
          data_e[j - row_start] = *rhs_off;
          num[j - row_start] = eid * rhs_dim + rhs_add;
          max_v = std::max<DType>(max_v, (*rhs_off));
        }
        DType exp_sum = 0;
        for (auto& element : data_e) {
          element -= max_v;
          element = std::exp(element);
          exp_sum += element;
        }
        for (int i = 0; i < row_end - row_start; i++) {
          out.Ptr<DType>()[num[i]] = data_e[i] / exp_sum;
        }
      }
    }
  });
}

/**
 * @brief CPU kernel of Edge_softmax_csr_backward on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param out The result of forward.
 * @param sds The result of gradiet * out.
 * @param back_out The result of edge_softmax_backward.
 */
template <typename IdType, typename DType, typename Op>
void Edge_softmax_csr_backward(
    const BcastOff& bcast, const CSRMatrix& csr, NDArray out, NDArray sds,
    NDArray back_out) {
  typedef typename std::conditional<
      std::is_same<DType, BFloat16>::value, float, DType>::type AccType;
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = static_cast<IdType*>(csr.indptr->data);
  const IdType* edges =
      has_idx ? static_cast<IdType*>(csr.data->data) : nullptr;
  const DType* W_out = Op::use_rhs ? static_cast<DType*>(out->data) : nullptr;
  const DType* W_sds = Op::use_rhs ? static_cast<DType*>(sds->data) : nullptr;
  const int64_t dim = bcast.out_len, rhs_dim = bcast.rhs_len;
  runtime::parallel_for(0, csr.num_rows, [&](size_t b, size_t e) {
    for (auto rid = b; rid < e; ++rid) {
      const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
      for (int64_t k = 0; k < dim; ++k) {
        AccType sum_sds = 0;
        for (IdType j = row_start; j < row_end; ++j) {
          const IdType eid = has_idx ? edges[j] : j;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* rhs_off_sds =
              Op::use_rhs ? W_sds + eid * rhs_dim + rhs_add : nullptr;
          sum_sds += (*rhs_off_sds);
        }
        for (IdType j = row_start; j < row_end; ++j) {
          const IdType eid = has_idx ? edges[j] : j;
          const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
          const DType* rhs_off_out =
              Op::use_rhs ? W_out + eid * rhs_dim + rhs_add : nullptr;
          const DType* rhs_off_sds =
              Op::use_rhs ? W_sds + eid * rhs_dim + rhs_add : nullptr;
          back_out.Ptr<DType>()[eid * rhs_dim + rhs_add] =
              (*rhs_off_sds) - sum_sds * (*rhs_off_out);
        }
      }
    }
  });
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPMM_H_
