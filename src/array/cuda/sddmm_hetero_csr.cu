/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/sddmm.cu
 * @brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>

#include "./sddmm.cuh"

namespace dgl {
namespace aten {

/**
 * @brief CUDA implementation of g-SDDMM on heterograph using Csr format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCsrHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& lhs_eid,
    const std::vector<dgl_type_t>& rhs_eid) {
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
}

template void SDDMMCsrHetero<kDGLCUDA, int32_t, __half>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int64_t, __half>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int32_t, float>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int64_t, float>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int32_t, double>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCUDA, int64_t, double>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);

}  // namespace aten
}  // namespace dgl
