/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./sddmm.cuh"
#include "./functor.cuh"

namespace dgl {
namespace aten {

/*!
 * \brief CUDA implementation of g-SDDMM on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const CSRMatrix& csr,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target) {
  SWITCH_BITS(bits, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        cuda::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, csr, lhs, rhs, out);
      });
    });
  });
}


/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const COOMatrix& coo,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target) {
  SWITCH_BITS(bits, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        cuda::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, coo, lhs, rhs, out);
      });
    });
  });
}


template void SDDMMCsr<kDLGPU, int32_t, 16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int64_t, 16>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int32_t, 32>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int64_t, 32>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int32_t, 64>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int64_t, 64>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);

template void SDDMMCoo<kDLGPU, int32_t, 16>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int64_t, 16>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int32_t, 32>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int64_t, 32>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int32_t, 64>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int64_t, 64>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);

}  // namespace aten
}  // namespace dgl
