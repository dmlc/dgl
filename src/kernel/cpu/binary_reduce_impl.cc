#include <dgl/runtime/ndarray.h>
#include "../binary_reduce.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {
namespace cpu {

void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_mapping,
    NDArray out_data) {
  LOG(INFO) << "Not implemented (CPU)";
}

void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_mapping,
    NDArray out_data) {
  LOG(INFO) << "Not implemented (CPU)";
}

void BackwardBinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray out_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data,
    NDArray grad_rhs_data) {
  LOG(INFO) << "Not implemented (CPU)";
}

void BackwardBinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    binary_op::Target lhs_tgt,
    binary_op::Target rhs_tgt,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray out_mapping,
    NDArray lhs,
    NDArray rhs,
    NDArray out,
    NDArray grad_out,
    NDArray grad_lhs,
    NDArray grad_rhs) {
  LOG(INFO) << "Not implemented (CPU)";
}

}  // namespace cpu
}  // namespace kernel
}  // namespace dgl
