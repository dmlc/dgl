/**
 *  Copyright (c) 2020 by Contributors
 * @file aten/cpu/sddmm.cc
 * @brief SDDMM C APIs and definitions.
 */
#include "./sddmm.h"

#include <dgl/array.h>

namespace dgl {
namespace aten {

#define SWITCH_RHS(rhs_target, RhsTarget, ...)             \
  do {                                                     \
    if ((rhs_target) == 0) {                               \
      constexpr int RhsTarget = 0;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 1) {                        \
      constexpr int RhsTarget = 1;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 2) {                        \
      constexpr int RhsTarget = 2;                         \
      { __VA_ARGS__ }                                      \
    } else {                                               \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target); \
    }                                                      \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...) \
  do {                                                                   \
    if ((lhs_target) == 0) {                                             \
      constexpr int LhsTarget = 0;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 1) {                                      \
      constexpr int LhsTarget = 1;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 2) {                                      \
      constexpr int LhsTarget = 2;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else {                                                             \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);               \
    }                                                                    \
  } while (0)

/** @brief Generalized SDDMM on Csr format. */
template <int XPU, typename IdType, typename DType>
void SDDMMCsr(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      cpu::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(
          bcast, csr, lhs, rhs, out);
    });
  });
}

/** @brief Generalized SDDMM on Csr format with Heterograph support. */
template <int XPU, typename IdType, typename DType>
void SDDMMCsrHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& lhs_nid,
    const std::vector<dgl_type_t>& rhs_nid) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      /* Call  SDDMM for each relation type */
      for (dgl_type_t etype = 0; etype < lhs_nid.size(); ++etype) {
        CSRMatrix csr = vec_csr[etype];
        NDArray lhs = vec_lhs[lhs_nid[etype]];
        NDArray rhs = vec_rhs[rhs_nid[etype]];
        NDArray out = vec_out[etype];
        cpu::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(
            bcast, csr, lhs, rhs, out);
      }
    });
  });
}

template void SDDMMCsr<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCsr<kDGLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);

template void SDDMMCsrHetero<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCsrHetero<kDGLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);

/** @brief Generalized SDDMM on Coo format. */
template <int XPU, typename IdType, typename DType>
void SDDMMCoo(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      cpu::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(
          bcast, coo, lhs, rhs, out);
    });
  });
}

/** @brief Generalized SDDMM on Coo format with Heterograph support. */
template <int XPU, typename IdType, typename DType>
void SDDMMCooHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& lhs_nid,
    const std::vector<dgl_type_t>& rhs_nid) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      /* Call  SDDMM for each relation type */
      for (dgl_type_t etype = 0; etype < lhs_nid.size(); ++etype) {
        COOMatrix coo = vec_coo[etype];
        NDArray lhs = vec_lhs[lhs_nid[etype]];
        NDArray rhs = vec_rhs[rhs_nid[etype]];
        NDArray out = vec_out[etype];
        cpu::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(
            bcast, coo, lhs, rhs, out);
      }
    });
  });
}

template void SDDMMCoo<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
template void SDDMMCoo<kDGLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);

template void SDDMMCooHetero<kDGLCPU, int32_t, BFloat16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDGLCPU, int64_t, BFloat16>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDGLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDGLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDGLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);
template void SDDMMCooHetero<kDGLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& lhs,
    const std::vector<NDArray>& rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target, const std::vector<dgl_type_t>& in_eid,
    const std::vector<dgl_type_t>& out_eid);

}  // namespace aten
}  // namespace dgl
