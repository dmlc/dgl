/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/sddmm.cc
 * \brief SDDMM C APIs and definitions.
 */
#include "./sddmm.h"
#include <dgl/array.h>

namespace dgl {
namespace kernel {

template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
  if (!aten::IsNullArray(ufeat))
    CHECK_EQ(ufeat->shape[0], csr.num_rows);
  if (!aten::IsNullArray(vfeat))
    CHECK_EQ(vfeat->shape[0], csr.num_cols);
  CHECK_EQ(out->shape[0], csr.indices->shape[0]);
  SWITCH_OP(op, Op, {
    cpu::SDDMMCsr<IdType, DType, Op>(bcast, csr, ufeat, vfeat, out);
  });
}

template void SDDMMCsr<kDLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCsr<kDLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              std::vector<NDArray> out_aux) {
  if (!aten::IsNullArray(ufeat))
    CHECK_EQ(ufeat->shape[0], coo.num_rows);
  if (!aten::IsNullArray(vfeat))
    CHECK_EQ(vfeat->shape[0], coo.num_cols);
  CHECK_EQ(out->shape[0], coo.row->shape[0]);
  SWITCH_OP(op, Op, {
    cpu::SDDMMCoo<IdType, DType, Op>(bcast, coo, ufeat, vfeat, out);
  });
}

template void SDDMMCoo<kDLCPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLCPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLCPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SDDMMCoo<kDLCPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

}  // namespace kernel
}  // namespace dgl
