#include <dgl/runtime/ndarray.h>
#include "../binary_reduce_common.h"

namespace dgl {
namespace kernel {
namespace cpu {

void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    runtime::NDArray indptr,
    runtime::NDArray indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    binary_op::Target out,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data) {
  LOG(INFO) << "Not implemented (CPU)";
}

}  // namespace cpu
}  // namespace kernel
}  // namespace dgl
