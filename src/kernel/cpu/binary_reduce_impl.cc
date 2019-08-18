/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/binary_reduce_impl.cc
 * \brief Binary reduce implementation on CPU.
 */
#include "../binary_reduce_impl.h"
#include "../csr_interface.h"

using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {

template void BinaryReduceImpl<kDLCPU>(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

template void BinaryReduceBcastImpl<kDLCPU>(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

template void BackwardBinaryReduceImpl<kDLCPU>(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs_data, NDArray rhs_data, NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data, NDArray grad_rhs_data);

template void BackwardBinaryReduceBcastImpl<kDLCPU>(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs_tgt, binary_op::Target rhs_tgt,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs);

}  // namespace kernel
}  // namespace dgl
