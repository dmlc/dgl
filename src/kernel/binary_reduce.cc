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

NDArray BinaryOpReduce(
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
    const int64_t out_size) {
  CHECK(IsValidBinaryOpShape(lhs_data, rhs_data))
    << "Cannot compute binary operation between feature shapes "
    << ShapeString(lhs_data) << " and " << ShapeString(rhs_data);
  const DLContext& ctx = indptr->ctx;
  const DLDataType& dtype = lhs_data->dtype;
  // Allocate output
  std::vector<int64_t> out_shape = {out_size};
  out_shape.insert(out_shape.end(), lhs_data->shape + 1, lhs_data->shape + lhs_data->ndim);
  NDArray out_data = NDArray::Empty(out_shape, dtype, ctx);
  DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
      reducer, binary_op, indptr, indices,
      binary_op::kSrc, binary_op::kEdge,
      lhs_mapping, rhs_mapping,
      lhs_data, rhs_data,
      out_mapping, out_data);
  return out_data;
}

bool HasBcast(NDArray lhs, NDArray rhs) {
  if (lhs->ndim != rhs->ndim) {
    return true;
  }
  for (int i = 1; i < lhs->ndim; ++i) {
    if (lhs->shape[i] != rhs->shape[i]) {
      return true;
    }
  }
  return false;
}

BcastInfo CalcBcastInfo(NDArray lhs, NDArray rhs) {
  BcastInfo ret;
  const int max_ndim = std::max(lhs->ndim, rhs->ndim);
  for (int j = 0; j < max_ndim; ++j) {
    const int dl = (j > lhs->ndim)? 1 : lhs->shape[(lhs->ndim - j)];
    const int dr = (j > rhs->ndim)? 1 : rhs->shape[(rhs->ndim - j)];
    if (dl != dr) {
      if (dl != 1 && dr != 1) {
        LOG(FATAL) << "Invalid broadcasting between feature shapes "
          << ShapeString(lhs) << " and " << ShapeString(rhs);
      }
    }
    ret.real_out_shape.push_back(std::max(dl, dr));
  }
  return ret;
}

NDArray BinaryOpReduceBcast(
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
    const int64_t out_size) {
  BcastInfo info = CalcBcastInfo(lhs_data, rhs_data);
  const DLContext& ctx = indptr->ctx;
  const DLDataType& dtype = lhs_data->dtype;
  // Allocate output
  std::vector<int64_t> out_shape = {out_size};
  out_shape.insert(out_shape.end(),
      info.real_out_shape.begin(), info.real_out_shape.end());
  NDArray out_data = NDArray::Empty(out_shape, dtype, ctx);

  //DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
  //    reducer, binary_op, indptr, indices,
  //    binary_op::kSrc, binary_op::kEdge,
  //    lhs_mapping, rhs_mapping,
  //    lhs_data, rhs_data,
  //    out_mapping, out_data);
  return out_data;
}
}  // namespace


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
  if (HasBcast(src_data, edge_data)) {
    return BinaryOpReduceBcast(reducer, binary_op, indptr, indices,
        binary_op::kSrc, binary_op::kEdge,
        src_mapping, edge_mapping,
        src_data, edge_data,
        out_mapping, out_size);
  } else {
    return BinaryOpReduce(reducer, binary_op, indptr, indices,
        binary_op::kSrc, binary_op::kEdge,
        src_mapping, edge_mapping,
        src_data, edge_data,
        out_mapping, out_size);
  }
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
  return BinaryOpReduce(reducer, binary_op, indptr, indices,
      binary_op::kSrc, binary_op::kDst,
      src_mapping, dst_mapping,
      src_data, dst_data,
      out_mapping, out_size);
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
