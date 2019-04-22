/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/kernel_apis.cc
 * \brief kernel APIs for graph computation
 */
#include "../c_api_common.h"
#include "./binary_reduce.h"
#include "./common.h"
#include "./functor.h"

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
    NDArray edge_ids,
    NDArray src_data,
    NDArray edge_data) {
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
  NDArray dummy;  // dummy dst data
  DGL_XPU_SWITCH(ctx.device_type, XPU, {
    DGL_DTYPE_SWITCH(dtype, DType, {
      REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
        BINARY_OP_SWITCH(binary_op, DType, BinaryOp, {
          if (edge_ids->ndim == 0) {
            BinaryReduceExecutor<XPU, DType, DirectId<int64_t>,
                                 SelectDst, SelectSrc, SelectEdge,
                                 BinaryOp, Reducer>::Run(
              indptr, indices, edge_ids, src_data, edge_data, dummy, out_data);
          } else {
            BinaryReduceExecutor<XPU, DType, IndirectId<XPU, int64_t>,
                                 SelectDst, SelectSrc, SelectEdge,
                                 BinaryOp, Reducer>::Run(
              indptr, indices, edge_ids, src_data, edge_data, dummy, out_data);
          }
        });
      });
    });
  });
  return out_data;
}

DGL_REGISTER_GLOBAL("backend.kernel._CAPI_DGLKernelSrcMulEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string binary_op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray edge_ids = args[4];
    NDArray src_data = args[5];
    NDArray edge_data = args[6];
    *rv = SrcOpEdgeReduce(reducer, binary_op, indptr, indices,
        edge_ids, src_data, edge_data);
  });

NDArray SrcOpDstReduce(
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr,
    NDArray indices,
    NDArray edge_ids,
    NDArray src_data,
    NDArray dst_data) {
}

DGL_REGISTER_GLOBAL("backend.kernel._CAPI_DGLKernelSrcMulDstReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string binary_op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray edge_ids = args[4];
    NDArray src_data = args[5];
    NDArray dst_data = args[6];
    *rv = SrcOpDstReduce(reducer, binary_op, indptr, indices,
        edge_ids, src_data, dst_data);
  });

}  // namespace kernel
}  // namespace dgl
