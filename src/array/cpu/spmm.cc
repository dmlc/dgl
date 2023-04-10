/**
 *  Copyright (c) 2020 by Contributors
 * @file kernel/cpu/spmm.cc
 * @brief SPMM C APIs and definitions.
 */
#include "./spmm.h"

#include <dgl/array.h>

namespace dgl {
namespace aten {

/** @brief Generalized SpMM on Csr format. */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  const int64_t dim = bcast.out_len;
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_OP(op, Op, {
      DType* out_off = out.Ptr<DType>();
      if (reduce == "max") {
        std::fill(
            out_off, out_off + csr.num_rows * dim, cpu::op::Max<DType>::zero);
        cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
            bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      } else {
        std::fill(
            out_off, out_off + csr.num_rows * dim, cpu::op::Min<DType>::zero);
        cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
            bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      }
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

/** @brief Generalized SpMM on Csr format. */
template <int XPU, typename IdType, typename DType>
void SpMMCsrHetero(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& vec_ufeat,
    const std::vector<NDArray>& vec_efeat, std::vector<NDArray>* vec_out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids) {
  const int64_t dim = bcast.out_len;
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      /* Call  SpMM for each relation type */
      for (dgl_type_t etype = 0; etype < ufeat_node_tids.size(); ++etype) {
        const dgl_type_t src_id = ufeat_node_tids[etype];
        const dgl_type_t dst_id = out_node_tids[etype];
        CSRMatrix csr = vec_csr[etype];
        NDArray ufeat =
            (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
        NDArray efeat =
            (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
        NDArray out = (*vec_out)[dst_id];
        cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
      }
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_OP(op, Op, {
      std::vector<bool> updated((*vec_out).size(), false);
      // TODO(Israt): use vector updated to fill(out...) too
      for (dgl_type_t etype = 0; etype < ufeat_node_tids.size(); ++etype) {
        DType* out_off = (*vec_out)[out_node_tids[etype]].Ptr<DType>();
        if (reduce == "max")
          std::fill(
              out_off, out_off + vec_csr[etype].num_rows * dim,
              cpu::op::Max<DType>::zero);
        else
          std::fill(
              out_off, out_off + vec_csr[etype].num_rows * dim,
              cpu::op::Min<DType>::zero);
        const dgl_type_t dst_id = out_node_tids[etype];
        if (!updated[dst_id]) {
          updated[dst_id] = true;
          if (Op::use_lhs) {
            IdType* argu_ntype = (*out_aux)[2][dst_id].Ptr<IdType>();
            std::fill(
                argu_ntype, argu_ntype + vec_csr[etype].num_rows * dim, -1);
          }
          if (Op::use_rhs) {
            IdType* arge_etype = (*out_aux)[3][dst_id].Ptr<IdType>();
            std::fill(
                arge_etype, arge_etype + vec_csr[etype].num_rows * dim, -1);
          }
        }
      }
      /* Call  SpMM for each relation type */
      for (dgl_type_t etype = 0; etype < ufeat_node_tids.size(); ++etype) {
        const dgl_type_t src_id = ufeat_node_tids[etype];
        const dgl_type_t dst_id = out_node_tids[etype];
        CSRMatrix csr = vec_csr[etype];
        NDArray ufeat =
            (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
        NDArray efeat =
            (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
        NDArray out = (*vec_out)[dst_id];
        if (reduce == "max") {
          cpu::SpMMCmpCsrHetero<IdType, DType, Op, cpu::op::Max<DType>>(
              bcast, csr, ufeat, efeat, out, (*out_aux)[0][dst_id],
              (*out_aux)[1][dst_id], (*out_aux)[2][dst_id],
              (*out_aux)[3][dst_id], src_id, etype);
        } else {
          cpu::SpMMCmpCsrHetero<IdType, DType, Op, cpu::op::Min<DType>>(
              bcast, csr, ufeat, efeat, out, (*out_aux)[0][dst_id],
              (*out_aux)[1][dst_id], (*out_aux)[2][dst_id],
              (*out_aux)[3][dst_id], src_id, etype);
        }
      }
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCsr<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCsr<kDGLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

template void SpMMCsrHetero<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDGLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDGLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDGLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDGLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);

/** @brief Edge_softmax_csr forward op on Csr format. */
template <int XPU, typename IdType, typename DType>
void Edge_softmax_csr_forward(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out) {
  SWITCH_OP(op, Op, {
    cpu::Edge_softmax_csr_forward<IdType, DType, Op>(
        bcast, csr, ufeat, efeat, out);
  });
}

/** @brief Edge_softmax_csr backward op on Csr format. */
template <int XPU, typename IdType, typename DType>
void Edge_softmax_csr_backward(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray out, NDArray sds, NDArray back_out) {
  SWITCH_OP(op, Op, {
    cpu::Edge_softmax_csr_backward<IdType, DType, Op>(
        bcast, csr, out, sds, back_out);
  });
}
template void Edge_softmax_csr_forward<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_forward<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_forward<kDGLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_forward<kDGLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_forward<kDGLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_forward<kDGLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);

template void Edge_softmax_csr_backward<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_backward<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_backward<kDGLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_backward<kDGLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_backward<kDGLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void Edge_softmax_csr_backward<kDGLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);

/** @brief Generalized SpMM on Coo format. */
template <int XPU, typename IdType, typename DType>
void SpMMCoo(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_OP(op, Op, {
      cpu::SpMMSumCoo<IdType, DType, Op>(bcast, coo, ufeat, efeat, out);
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_OP(op, Op, {
      if (reduce == "max")
        cpu::SpMMCmpCoo<IdType, DType, Op, cpu::op::Max<DType>>(
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      else
        cpu::SpMMCmpCoo<IdType, DType, Op, cpu::op::Min<DType>>(
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCoo<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCPU, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCPU, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCPU, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);
template void SpMMCoo<kDGLCPU, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

}  // namespace aten
}  // namespace dgl
