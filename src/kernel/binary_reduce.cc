/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/kernel_apis.cc
 * \brief kernel APIs for graph computation
 */
#include "../c_api_common.h"
#include "./binary_reduce.h"
#include "./common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;

namespace dgl {
namespace kernel {
namespace {

bool IsValidBinaryOpShape(NDArray lhs, NDArray rhs) {
  if (lhs->ndim != rhs->ndim) {
    return false;
  }
  for (int i = 1; i < lhs->ndim; ++i) {
    if (lhs->shape[i] != rhs->shape[i]) {
      return false;
    }
  }
  return true;
}

std::string ShapeString(NDArray nd) {
  std::ostringstream oss;
  oss << "(";
  for (int i = 1; i < nd->ndim; ++i) {
    oss << nd->shape[i];
    if (i != nd->ndim - 1) {
      oss << ",";
    }
  }
  oss << ")";
  return oss.str();
}

}  // namespace

std::vector<int64_t> BinaryElewiseInferShape(
    const std::string& reducer,
    NDArray indptr,
    NDArray indices,
    NDArray lhs,
    NDArray rhs) {
  // TODO(minjie): no broadcasting shape
  std::vector<int64_t> ret;
  if (reducer == binary_op::kReduceNone) {
    // Output is an edge feature tensor.
    ret.push_back(indices->shape[0]);
  } else {
    // Output is a node feature tensor.
    ret.push_back(indptr->shape[0] - 1);
  }
  for (int i = 1; i < lhs->ndim; ++i) {
    ret.push_back(lhs->shape[i]);
  }
  return ret;
}

NDArray SrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr,
    NDArray indices,
    NDArray src_mapping,
    NDArray edge_mapping,
    NDArray src_data,
    NDArray edge_data,
    NDArray out_mapping,
    const int64_t out_size) {
  // TODO(minjie): no broadcasting shape
  CHECK(IsValidBinaryOpShape(src_data, edge_data))
    << "Cannot compute binary operation between feature shapes "
    << ShapeString(src_data) << " and " << ShapeString(edge_data);
  const DLContext& ctx = indptr->ctx;
  const DLDataType& dtype = src_data->dtype;
  // Allocate output
  const auto& out_shape = BinaryElewiseInferShape(reducer, indptr,
      indices, src_data, edge_data);
  NDArray out_data = NDArray::Empty(out_shape, dtype, ctx);
  DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
      reducer, binary_op, indptr, indices,
      binary_op::kSrc, binary_op::kEdge,
      src_mapping, edge_mapping,
      src_data, edge_data,
      out_mapping, out_data);
  return out_data;
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelSrcMulEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string binary_op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray src_mapping = args[4];
    NDArray edge_mapping = args[5];
    NDArray src_data = args[6];
    NDArray edge_data = args[7];
    NDArray out_mapping = args[8];
    const int64_t out_size = args[9];
    *rv = SrcOpEdgeReduce(reducer, binary_op, indptr, indices,
        src_mapping, edge_mapping, src_data, edge_data, out_mapping, out_size);
  });

NDArray SrcOpDstReduce(
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr,
    NDArray indices,
    NDArray src_mapping,
    NDArray dst_mapping,
    NDArray src_data,
    NDArray dst_data,
    NDArray out_mapping,
    const int64_t out_size) {
  return NDArray();
}

DGL_REGISTER_GLOBAL("backend.kernel._CAPI_DGLKernelSrcMulDstReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string binary_op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray src_mapping = args[4];
    NDArray dst_mapping = args[5];
    NDArray src_data = args[6];
    NDArray dst_data = args[7];
    NDArray out_mapping = args[8];
    const int64_t out_size = args[9];
    *rv = SrcOpDstReduce(reducer, binary_op, indptr, indices,
        src_mapping, dst_mapping, src_data, dst_data, out_mapping, out_size);
  });

}  // namespace kernel
}  // namespace dgl
