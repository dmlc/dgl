/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./sddmm.cuh"

namespace dgl {
namespace aten {

/*!
 * \brief CUDA implementation of g-SDDMM on heterograph using 
    Csr format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCsrHetero(const std::string& op,
              const BcastOff& bcast,
              const std::vector<CSRMatrix>& vec_csr,
              const std::vector<NDArray>& vec_lhs,
              const std::vector<NDArray>& vec_rhs,
              std::vector<NDArray> vec_out,
              int lhs_target,
              int rhs_target,
              const std::vector<dgl_type_t>& lhs_eid,
              const std::vector<dgl_type_t>& rhs_eid) {
  SWITCH_BITS(bits, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        /* Call SDDMM CUDA kernel for each relation type sequentially */
        for (dgl_type_t etype = 0; etype < lhs_eid.size(); ++etype) {
          CSRMatrix csr = vec_csr[etype];
          NDArray lhs = vec_lhs[lhs_eid[etype]];
          NDArray rhs = vec_rhs[rhs_eid[etype]];
          NDArray out = vec_out[etype];
          cuda::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(
            bcast, csr, lhs, rhs, out);
        }
      });
    });
  });
}


/*!
 * \brief CUDA implementation of g-SDDMM on heterograph using
    Csr format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCooHetero(const std::string& op,
              const BcastOff& bcast,
              const std::vector<COOMatrix>& vec_coo,
              const std::vector<NDArray>& vec_lhs,
              const std::vector<NDArray>& vec_rhs,
              std::vector<NDArray> vec_out,
              int lhs_target,
              int rhs_target,
              const std::vector<dgl_type_t>& lhs_eid,
              const std::vector<dgl_type_t>& rhs_eid) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  SWITCH_BITS(bits, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        /* Call SDDMM CUDA kernel for each relation type sequentially */
        for (dgl_type_t etype = 0; etype < lhs_eid.size(); ++etype) {
          COOMatrix coo = vec_coo[etype];
          NDArray lhs = vec_lhs[lhs_eid[etype]];
          NDArray rhs = vec_rhs[rhs_eid[etype]];
          NDArray out = vec_out[etype];
          cuda::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(
            bcast, coo, lhs, rhs, out);
        }
      });
    });
  });
}


template void SDDMMCsrHetero<kDLGPU, int32_t, 16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDLGPU, int64_t, 16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDLGPU, int32_t, 32>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDLGPU, int64_t, 32>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDLGPU, int32_t, 64>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDLGPU, int64_t, 64>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);

template void SDDMMCooHetero<kDLGPU, int32_t, 16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDLGPU, int64_t, 16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDLGPU, int32_t, 32>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDLGPU, int64_t, 32>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDLGPU, int32_t, 64>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDLGPU, int64_t, 64>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo,
    const std::vector<NDArray>& lhs, const std::vector<NDArray>& rhs,
    std::vector<NDArray> out, int lhs_target, int rhs_target,
    const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);

}  // namespace aten
}  // namespace dgl
