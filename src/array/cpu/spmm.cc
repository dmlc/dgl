/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/spmm.cc
 * \brief SPMM C APIs and definitions.
 */
#include "./spmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

/*! \brief Generalized SpMM on Csr format. */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  const int64_t dim = bcast.out_len;
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
      });
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        DType *out_off = out.Ptr<DType>();
        IdType* argX = Op::use_lhs ? static_cast<IdType*>(out_aux[0]->data) : nullptr;
        IdType* argW = Op::use_rhs ? static_cast<IdType*>(out_aux[1]->data) : nullptr;
        if (reduce == "max") {
          std::fill(out_off, out_off + csr.num_rows * dim, cpu::op::Max<DType>::zero);
          cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
              bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
        } else {
          std::fill(out_off, out_off + csr.num_rows * dim, cpu::op::Min<DType>::zero);
          cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
              bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
        }
      });
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

/*! \brief Generalized SpMM on Csr format. */
template <int XPU, typename IdType, int bits>
void SpMMCsrHetero(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const std::vector<CSRMatrix>& vec_csr,
             const std::vector<NDArray>& vec_ufeat,
             const std::vector<NDArray>& vec_efeat,
             std::vector<NDArray> vec_out,
             const std::vector<NDArray>& out_aux,
             const std::vector<dgl_type_t>& ufeat_node_tids,
             const std::vector<dgl_type_t>& out_node_tids) {
  const int64_t dim = bcast.out_len;
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        /* Call  SpMM for each relation type */
        for (dgl_type_t etype = 0; etype < ufeat_node_tids.size(); ++etype) {
          const dgl_type_t src_id = ufeat_node_tids[etype];
          const dgl_type_t dst_id = out_node_tids[etype];
          CSRMatrix csr = vec_csr[etype];
          NDArray ufeat = (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
          NDArray efeat = (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
          NDArray out = vec_out[dst_id];
          cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
        }
      });
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        /* Call  SpMM for each relation type */
        for (dgl_type_t etype = 0; etype < ufeat_node_tids.size(); ++etype) {
          const dgl_type_t src_id = ufeat_node_tids[etype];
          const dgl_type_t dst_id = out_node_tids[etype];
          CSRMatrix csr = vec_csr[etype];
          DType *out_off = vec_out[out_node_tids[etype]].Ptr<DType>();
          NDArray ufeat = (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
          NDArray efeat = (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
          NDArray out = vec_out[dst_id];
          if (reduce == "max") {
            std::fill(out_off, out_off + csr.num_rows * dim, cpu::op::Max<DType>::zero);
            cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
                bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
          } else {
            std::fill(out_off, out_off + csr.num_rows * dim, cpu::op::Min<DType>::zero);
            cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
                bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
          }
        }
      });
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCsr<kDLCPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template void SpMMCsrHetero<kDLCPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    const std::vector<NDArray>& ufeat, const std::vector<NDArray>& efeat,
    std::vector<NDArray> out, const std::vector<NDArray>& out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDLCPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    const std::vector<NDArray>& ufeat, const std::vector<NDArray>& efeat,
    std::vector<NDArray> out, const std::vector<NDArray>& out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDLCPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    const std::vector<NDArray>& ufeat, const std::vector<NDArray>& efeat,
    std::vector<NDArray> out, const std::vector<NDArray>& out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDLCPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    const std::vector<NDArray>& ufeat, const std::vector<NDArray>& efeat,
    std::vector<NDArray> out, const std::vector<NDArray>& out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDLCPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    const std::vector<NDArray>& ufeat, const std::vector<NDArray>& efeat,
    std::vector<NDArray> out, const std::vector<NDArray>& out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);
template void SpMMCsrHetero<kDLCPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    const std::vector<NDArray>& ufeat, const std::vector<NDArray>& efeat,
    std::vector<NDArray> out, const std::vector<NDArray>& out_aux,
    const std::vector<dgl_type_t>& ufeat_node_tids,
    const std::vector<dgl_type_t>& out_node_tids);

/*! \brief Generalized SpMM on Coo format. */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cpu::SpMMSumCoo<IdType, DType, Op>(bcast, coo, ufeat, efeat, out);
      });
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        if (reduce == "max")
          cpu::SpMMCmpCoo<IdType, DType, Op, cpu::op::Max<DType>>(
              bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
        else
          cpu::SpMMCmpCoo<IdType, DType, Op, cpu::op::Min<DType>>(
              bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCoo<kDLCPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


}  // namespace aten
}  // namespace dgl
