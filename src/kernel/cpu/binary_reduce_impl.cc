#include <dgl/runtime/ndarray.h>
#include "../binary_reduce.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {
namespace cpu {

void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr,
    NDArray indices,
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
    NDArray indptr,
    NDArray indices,
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

}  // namespace cpu
}  // namespace kernel
}  // namespace dgl
