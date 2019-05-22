/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce.cc
 * \brief Binary reduce C APIs and definitions.
 */
#include "./binary_reduce.h"
#include "./common.h"
#include "./binary_reduce_impl_decl.h"
#include "./utils.h"
#include "../c_api_common.h"

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

std::vector<int64_t> ComputeStride(const std::vector<int64_t>& shape) {
  std::vector<int64_t> ret(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) {
    ret[i] = ret[i+1] * shape[i+1];
  }
  return ret;
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

void BinaryOpReduce(
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
  CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  if (op != binary_op::kUseLhs) {
    CHECK(IsValidBinaryOpShape(lhs_data, rhs_data))
      << "Cannot compute binary operation between feature shapes "
      << ShapeString(lhs_data) << " and " << ShapeString(rhs_data);
  }
  const DLContext& ctx = indptr->ctx;
  // Allocate output
  DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
      reducer, op, indptr, indices, rev_indptr, rev_indices,
      lhs, rhs,
      lhs_mapping, rhs_mapping,
      lhs_data, rhs_data,
      out_mapping, out_data);
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
  const int max_ndim = std::max(lhs->ndim, rhs->ndim) - 1;
  int64_t accum = 0;
  for (int j = 0; j < max_ndim; ++j) {
    const int dl = (lhs->ndim - 1 - j < 1)? 1 : lhs->shape[lhs->ndim - 1 - j];
    const int dr = (rhs->ndim - 1 - j < 1)? 1 : rhs->shape[rhs->ndim - 1 - j];
    if (dl != dr) {
      if (dl != 1 && dr != 1) {
        LOG(FATAL) << "Invalid broadcasting between feature shapes "
          << ShapeString(lhs) << " and " << ShapeString(rhs);
      }
      if (accum != 0) {
        ret.lhs_shape.push_back(accum);
        ret.rhs_shape.push_back(accum);
        ret.out_shape.push_back(accum);
        accum = 0;
      }
      ret.lhs_shape.push_back(dl);
      ret.rhs_shape.push_back(dr);
      ret.out_shape.push_back(std::max(dl, dr));
    } else {
      if (accum == 0) {
        accum = dl;
      } else {
        accum *= dl;
      }
    }
    ret.real_out_shape.push_back(std::max(dl, dr));
  }
  if (accum != 0) {
    ret.lhs_shape.push_back(accum);
    ret.rhs_shape.push_back(accum);
    ret.out_shape.push_back(accum);
    accum = 0;
  }
  std::reverse(ret.real_out_shape.begin(), ret.real_out_shape.end());
  std::reverse(ret.lhs_shape.begin(), ret.lhs_shape.end());
  std::reverse(ret.rhs_shape.begin(), ret.rhs_shape.end());
  std::reverse(ret.out_shape.begin(), ret.out_shape.end());
  // stride
  ret.lhs_stride = ComputeStride(ret.lhs_shape);
  ret.rhs_stride = ComputeStride(ret.rhs_shape);
  ret.out_stride = ComputeStride(ret.out_shape);
  return ret;
}

void BinaryOpReduceBcast(
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
  CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  BcastInfo info = CalcBcastInfo(lhs_data, rhs_data);
  const DLContext& ctx = indptr->ctx;
  DGL_XPU_SWITCH(ctx.device_type, BinaryReduceBcastImpl,
      info, reducer, binary_op,
      indptr, indices, rev_indptr, rev_indices,
      lhs, rhs,
      lhs_mapping, rhs_mapping,
      lhs_data, rhs_data,
      out_mapping, out_data);
}
}  // namespace


std::vector<int64_t> InferBinaryFeatureShape(
    NDArray lhs,
    NDArray rhs) {
  return CalcBcastInfo(lhs, rhs).real_out_shape;
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelInferBinaryFeatureShape")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray lhs = args[0];
    NDArray rhs = args[1];
    const auto& shape = InferBinaryFeatureShape(lhs, rhs);
    const int64_t len = shape.size();
    NDArray ret = NDArray::Empty(
        {len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t* ret_data = static_cast<int64_t*>(ret->data);
    std::copy(shape.begin(), shape.end(), ret_data);
    *rv = ret;
  });

void BinaryOpReduce_v2(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_data, NDArray rhs_data,
    NDArray out_data,
    NDArray lhs_mapping, NDArray rhs_mapping,
    NDArray out_mapping) {
  // sanity check
  const auto& ctx = graph->Context();
  CHECK_EQ(ctx, lhs_data->ctx) << "Expected device context (type="
    << ctx.device_type << ", id=" << ctx.device_id << "). But got (type="
    << lhs_data->ctx.device_type << ", id=" << lhs_data->ctx.device_id << " for lhs_data.";
  CHECK_EQ(ctx, rhs_data->ctx) << "Expected device context (type="
    << ctx.device_type << ", id=" << ctx.device_id << "). But got (type="
    << rhs_data->ctx.device_type << ", id=" << rhs_data->ctx.device_id << " for rhs_data.";
  CHECK_EQ(ctx, out_data->ctx) << "Expected device context (type="
    << ctx.device_type << ", id=" << ctx.device_id << "). But got (type="
    << out_data->ctx.device_type << ", id=" << out_data->ctx.device_id << " for out_data.";
  if (!utils::IsNoneArray(lhs_mapping)) {
    CHECK_EQ(ctx, lhs_mapping->ctx) << "Expected device context (type="
      << ctx.device_type << ", id=" << ctx.device_id << "). But got (type="
      << lhs_mapping->ctx.device_type << ", id=" << lhs_mapping->ctx.device_id << " for lhs_mapping.";
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    CHECK_EQ(ctx, rhs_mapping->ctx) << "Expected device context (type="
      << ctx.device_type << ", id=" << ctx.device_id << "). But got (type="
      << rhs_mapping->ctx.device_type << ", id=" << rhs_mapping->ctx.device_id << " for rhs_mapping.";
  }
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context (type="
      << ctx.device_type << ", id=" << ctx.device_id << "). But got (type="
      << out_mapping->ctx.device_type << ", id=" << out_mapping->ctx.device_id << " for out_mapping.";
  }
  // Process mapping
  if (HasBcast(lhs_data, rhs_data)) {
    BcastInfo info = CalcBcastInfo(lhs_data, rhs_data);
    const DLContext& ctx = lhs_data->ctx;
    DGL_XPU_SWITCH(ctx.device_type, BinaryReduceBcastImpl_v2,
        info, reducer, op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping);
  } else {
    if (op != binary_op::kUseLhs) {
      CHECK(IsValidBinaryOpShape(lhs_data, rhs_data))
        << "Cannot compute binary operation between feature shapes "
        << ShapeString(lhs_data) << " and " << ShapeString(rhs_data);
    }
    const DLContext& ctx = lhs_data->ctx;
    DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl_v2,
        reducer, op, graph,
        lhs, rhs,
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBinaryOpReduce_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    GraphHandle ghdl = args[2];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_data = args[5];
    NDArray rhs_data = args[6];
    NDArray out_data = args[7];
    NDArray lhs_mapping = args[8];
    NDArray rhs_mapping = args[9];
    NDArray out_mapping = args[10];

    GraphInterface* gptr = static_cast<GraphInterface*>(ghdl);
    const ImmutableGraph* igptr = dynamic_cast<ImmutableGraph*>(gptr);
    CHECK(igptr) << "Invalid graph object argument. Must be an immutable graph.";
    BinaryOpReduce_v2(reducer, op, igptr,
        static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
        lhs_data, rhs_data, out_data,
        lhs_mapping, rhs_mapping, out_mapping);
  });

/*
void SrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray src_mapping,
    NDArray edge_mapping,
    NDArray src_data,
    NDArray edge_data,
    NDArray out_mapping,
    NDArray out_data) {
  if (HasBcast(src_data, edge_data)) {
    BinaryOpReduceBcast(reducer, binary_op,
        indptr, indices, rev_indptr, rev_indices,
        binary_op::kSrc, binary_op::kEdge,
        src_mapping, edge_mapping,
        src_data, edge_data,
        out_mapping, out_data);
  } else {
    BinaryOpReduce(reducer, binary_op,
        indptr, indices, rev_indptr, rev_indices,
        binary_op::kSrc, binary_op::kEdge,
        src_mapping, edge_mapping,
        src_data, edge_data,
        out_mapping, out_data);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelSrcOpEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string binary_op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray rev_indptr = args[4];
    NDArray rev_indices = args[5];
    NDArray src_mapping = args[6];
    NDArray edge_mapping = args[7];
    NDArray src_data = args[8];
    NDArray edge_data = args[9];
    NDArray out_mapping = args[10];
    NDArray out_data = args[11];
    SrcOpEdgeReduce(reducer, binary_op,
        indptr, indices, rev_indptr, rev_indices,
        src_mapping, edge_mapping, src_data, edge_data,
        out_mapping, out_data);
  });

void SrcOpDstReduce(
    const std::string& reducer,
    const std::string& binary_op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray src_mapping,
    NDArray dst_mapping,
    NDArray src_data,
    NDArray dst_data,
    NDArray out_mapping,
    NDArray out_data) {
  if (HasBcast(src_data, dst_data)) {
    BinaryOpReduceBcast(reducer, binary_op,
        indptr, indices, rev_indptr, rev_indices,
        binary_op::kSrc, binary_op::kDst,
        src_mapping, dst_mapping,
        src_data, dst_data,
        out_mapping, out_data);
  } else {
    BinaryOpReduce(reducer, binary_op,
        indptr, indices, rev_indptr, rev_indices,
        binary_op::kSrc, binary_op::kDst,
        src_mapping, dst_mapping,
        src_data, dst_data,
        out_mapping, out_data);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelSrcOpDstReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string binary_op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray rev_indptr = args[4];
    NDArray rev_indices = args[5];
    NDArray src_mapping = args[6];
    NDArray dst_mapping = args[7];
    NDArray src_data = args[8];
    NDArray dst_data = args[9];
    NDArray out_mapping = args[10];
    NDArray out_data = args[11];
    SrcOpDstReduce(reducer, binary_op,
        indptr, indices, rev_indptr, rev_indices,
        src_mapping, dst_mapping, src_data, dst_data,
        out_mapping, out_data);
  });


void CopySrcReduce(
    const std::string& reducer,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray src_mapping,
    NDArray src_data,
    NDArray out_mapping,
    NDArray out_data) {
  BinaryOpReduce(reducer, binary_op::kUseLhs,
      indptr, indices, rev_indptr, rev_indices,
      binary_op::kSrc, binary_op::kEdge,
      src_mapping, utils::NoneArray(),
      src_data, utils::NoneArray(),
      out_mapping, out_data);
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelCopySrcReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    NDArray indptr = args[1];
    NDArray indices = args[2];
    NDArray rev_indptr = args[3];
    NDArray rev_indices = args[4];
    NDArray src_mapping = args[5];
    NDArray src_data = args[6];
    NDArray out_mapping = args[7];
    NDArray out_data = args[8];
    CopySrcReduce(reducer, indptr, indices, rev_indptr, rev_indices,
        src_mapping, src_data, out_mapping, out_data);
  });

void CopyEdgeReduce(
    const std::string& reducer,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray edge_mapping,
    NDArray edge_data,
    NDArray out_mapping,
    NDArray out_data) {
  BinaryOpReduce(reducer, binary_op::kUseLhs,
      indptr, indices, rev_indptr, rev_indices,
      binary_op::kEdge, binary_op::kDst,
      edge_mapping, utils::NoneArray(),
      edge_data, utils::NoneArray(),
      out_mapping, out_data);
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelCopyEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    NDArray indptr = args[1];
    NDArray indices = args[2];
    NDArray rev_indptr = args[3];
    NDArray rev_indices = args[4];
    NDArray edge_mapping = args[5];
    NDArray edge_data = args[6];
    NDArray out_mapping = args[7];
    NDArray out_data = args[8];
    CopyEdgeReduce(reducer, indptr, indices, rev_indptr, rev_indices,
        edge_mapping, edge_data, out_mapping, out_data);
  });


void BackwardLhsSrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray src_mapping,
    NDArray edge_mapping,
    NDArray out_mapping,
    NDArray src_data,
    NDArray edge_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_src_data) {
  CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  CHECK(IsValidCsr(rev_indptr, rev_indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  if (HasBcast(src_data, edge_data)) {
    BcastInfo info = CalcBcastInfo(src_data, edge_data);
    DGL_XPU_SWITCH(grad_src_data->ctx.device_type, BackwardBinaryReduceBcastImpl,
        info, reducer, op, indptr, indices, rev_indptr, rev_indices,
        binary_op::kDst, binary_op::kEdge,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_src_data, utils::NoneArray());
  } else {
    DGL_XPU_SWITCH(grad_src_data->ctx.device_type, BackwardBinaryReduceImpl,
        reducer, op, indptr, indices, rev_indptr, rev_indices,
        binary_op::kDst, binary_op::kEdge,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_src_data, utils::NoneArray());
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardLhsSrcOpEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray rev_indptr = args[4];
    NDArray rev_indices = args[5];
    NDArray src_mapping = args[6];
    NDArray edge_mapping = args[7];
    NDArray out_mapping = args[8];
    NDArray src_data = args[9];
    NDArray edge_data = args[10];
    NDArray out_data = args[11];
    NDArray grad_out_data = args[12];
    NDArray grad_src_data = args[13];
    BackwardLhsSrcOpEdgeReduce(
        reducer, op, indptr, indices, rev_indptr, rev_indices,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_src_data);
  });

void BackwardRhsSrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray src_mapping,
    NDArray edge_mapping,
    NDArray out_mapping,
    NDArray src_data,
    NDArray edge_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_edge_data) {
  CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  CHECK(IsValidCsr(rev_indptr, rev_indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  if (HasBcast(src_data, edge_data)) {
    BcastInfo info = CalcBcastInfo(src_data, edge_data);
    DGL_XPU_SWITCH(grad_edge_data->ctx.device_type, BackwardBinaryReduceBcastImpl,
        info, reducer, op, indptr, indices, rev_indptr, rev_indices,
        binary_op::kDst, binary_op::kEdge,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        utils::NoneArray(), grad_edge_data);
  } else {
    DGL_XPU_SWITCH(grad_edge_data->ctx.device_type, BackwardBinaryReduceImpl,
        reducer, op, indptr, indices, rev_indptr, rev_indices,
        binary_op::kDst, binary_op::kEdge,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        utils::NoneArray(), grad_edge_data);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardRhsSrcOpEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray rev_indptr = args[4];
    NDArray rev_indices = args[5];
    NDArray src_mapping = args[6];
    NDArray edge_mapping = args[7];
    NDArray out_mapping = args[8];
    NDArray src_data = args[9];
    NDArray edge_data = args[10];
    NDArray out_data = args[11];
    NDArray grad_out_data = args[12];
    NDArray grad_edge_data = args[13];
    BackwardRhsSrcOpEdgeReduce(
        reducer, op, indptr, indices, rev_indptr, rev_indices,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_edge_data);
  });


void BackwardBothSrcOpEdgeReduce(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr, NDArray indices,
    NDArray rev_indptr, NDArray rev_indices,
    NDArray src_mapping,
    NDArray edge_mapping,
    NDArray out_mapping,
    NDArray src_data,
    NDArray edge_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_src_data,
    NDArray grad_edge_data) {
  CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  CHECK(IsValidCsr(rev_indptr, rev_indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
  if (HasBcast(src_data, edge_data)) {
    BcastInfo info = CalcBcastInfo(src_data, edge_data);
    DGL_XPU_SWITCH(grad_edge_data->ctx.device_type, BackwardBinaryReduceBcastImpl,
        info, reducer, op, indptr, indices, rev_indptr, rev_indices,
        binary_op::kDst, binary_op::kEdge,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_src_data, grad_edge_data);
  } else {
    DGL_XPU_SWITCH(grad_edge_data->ctx.device_type, BackwardBinaryReduceImpl,
        reducer, op, indptr, indices, rev_indptr, rev_indices,
        binary_op::kDst, binary_op::kEdge,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_src_data, grad_edge_data);
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardBothSrcOpEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    NDArray indptr = args[2];
    NDArray indices = args[3];
    NDArray rev_indptr = args[4];
    NDArray rev_indices = args[5];
    NDArray src_mapping = args[6];
    NDArray edge_mapping = args[7];
    NDArray out_mapping = args[8];
    NDArray src_data = args[9];
    NDArray edge_data = args[10];
    NDArray out_data = args[11];
    NDArray grad_out_data = args[12];
    NDArray grad_src_data = args[13];
    NDArray grad_edge_data = args[14];
    BackwardBothSrcOpEdgeReduce(
        reducer, op, indptr, indices, rev_indptr, rev_indices,
        src_mapping, edge_mapping, out_mapping,
        src_data, edge_data, out_data, grad_out_data,
        grad_src_data, grad_edge_data);
  });

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardCopySrcReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    NDArray indptr = args[1];
    NDArray indices = args[2];
    NDArray rev_indptr = args[3];
    NDArray rev_indices = args[4];
    NDArray src_mapping = args[5];
    NDArray out_mapping = args[6];
    NDArray src_data = args[7];
    NDArray out_data = args[8];
    NDArray grad_out_data = args[9];
    NDArray grad_src_data = args[10];
    CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
    CHECK(IsValidCsr(rev_indptr, rev_indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
    DGL_XPU_SWITCH(grad_src_data->ctx.device_type, BackwardBinaryReduceImpl,
      reducer, binary_op::kUseLhs, indptr, indices, rev_indptr, rev_indices,
      binary_op::kDst, binary_op::kEdge,
      src_mapping, utils::NoneArray(), out_mapping,
      src_data, utils::NoneArray(), out_data, grad_out_data,
      grad_src_data, utils::NoneArray());
  });


DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardCopyEdgeReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    NDArray indptr = args[1];
    NDArray indices = args[2];
    NDArray rev_indptr = args[3];
    NDArray rev_indices = args[4];
    NDArray edge_mapping = args[5];
    NDArray out_mapping = args[6];
    NDArray edge_data = args[7];
    NDArray out_data = args[8];
    NDArray grad_out_data = args[9];
    NDArray grad_edge_data = args[10];
    CHECK(IsValidCsr(indptr, indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
    CHECK(IsValidCsr(rev_indptr, rev_indices)) << "Invalid CSR arrays (e.g. shape, dtype)";
    DGL_XPU_SWITCH(grad_edge_data->ctx.device_type, BackwardBinaryReduceImpl,
      reducer, binary_op::kUseLhs, indptr, indices, rev_indptr, rev_indices,
      binary_op::kEdge, binary_op::kDst,
      edge_mapping, utils::NoneArray(), out_mapping,
      edge_data, utils::NoneArray(), out_data, grad_out_data,
      grad_edge_data, utils::NoneArray());
  });
*/

}  // namespace kernel
}  // namespace dgl
