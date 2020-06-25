/*!
 *  Copyright (c) 2020 by Contributors
 * \file aten/cpu/sddmm.cc
 * \brief SDDMM C APIs and definitions.
 */
#include "./sddmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

/*! \brief Generalized SDDMM on Csr format. */
template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const CSRMatrix& csr,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out) {
  SWITCH_OP(op, Op, {
    cpu::SDDMMCsr<IdType, DType, Op>(bcast, csr, ufeat, vfeat, out);
  });
}

template void SDDMMCsr<kDLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void SDDMMCsr<kDLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void SDDMMCsr<kDLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
template void SDDMMCsr<kDLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);

/*! \brief Generalized SDDMM on Coo format. */
template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const COOMatrix& coo,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out) {
  SWITCH_OP(op, Op, {
    cpu::SDDMMCoo<IdType, DType, Op>(bcast, coo, ufeat, vfeat, out);
  });
}

template void SDDMMCoo<kDLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out);
template void SDDMMCoo<kDLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out);
template void SDDMMCoo<kDLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out);
template void SDDMMCoo<kDLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out);

}  // namespace aten
}  // namespace dgl
