/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce.cc
 * \brief Binary reduce C APIs and definitions.
 */
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include "./binary_reduce.h"
#include "./common.h"
#include "./binary_reduce_impl_decl.h"
#include "./utils.h"
#include "../c_api_common.h"
#include "../graph/unit_graph.h"
#include "./csr_interface.h"

using namespace dgl::runtime;

namespace dgl {
namespace kernel {
namespace {

// convert ndarray shape to string
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

// compute stride vector given shape; assume row-major storage
std::vector<int64_t> ComputeStride(const std::vector<int64_t>& shape) {
  std::vector<int64_t> ret(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; --i) {
    ret[i] = ret[i+1] * shape[i+1];
  }
  return ret;
}

// Return true if the feature shapes of the two ndarrays can be
// computed element-wisely *without* broadcasting.
// Examples:
//
// valid:
//  lhs.shape = (N, D1, D2)
//  rhs.shape = (M, D1, D2)  # the first dimension could be different
//
// invalid:
//  lhs.shape = (N, D1, D2)
//  rhs.shape = (M, D1)
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

// Return true if broadcasting might be required to compute the element-wise
// operation between the features of the two ndarrays.
// The broadcasting semantic strictly follows numpy.
// Note that the function could return true for invalid element-wise shapes
// (e.g. lhs.shape = (N, 3), rhs.shape = (N, 5)). This is fine since
// ``CalcBcastInfo`` will handle that.
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

// Compute auxiliary information of broadcasting dimensions.
// The function preprocesses the feature shapes so that:
//  - The first dimension (for graph) is removed.
//  - Feature dimensions are aligned.
//    e.g. (4,) and (3, 4) become (1, 4) and (3, 4)
//  - Continuous non-broadcasting dimenions are flattened to reduce number of
//    integers used to represent the feature shape.
//    e.g. (4, 1, 3, 3) and (4, 5, 3, 3) become (4, 1, 9) and (4, 5, 9)
//
// See also: BcastInfo (kernel/binary_reduce.h)
BcastInfo CalcBcastInfo(const std::string& op, NDArray lhs, NDArray rhs) {
  BcastInfo ret;
  const int max_ndim = std::max(lhs->ndim, rhs->ndim) - 1;
  int64_t accum = 0;
  int j = 0;
  // for dot operation: vector [dot] vector
  // lhs_shape[ndim-1] == rhs_shape[ndim-1] = sizeof(vector)
  // out_shape[ndim-1] = 1
  if (op == binary_op::kDot) {
    // get size of vector
    ret.data_len = lhs->shape[lhs->ndim - 1];
    // skip vector size dim
    ++j;
    ret.real_out_shape.push_back(ret.data_len);
  } else {  // op != binary_op::kDot
    ret.data_len = 1;
  }

  for (; j < max_ndim; ++j) {
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

// Function to convert an idarray to string
std::string IdArrayToStr(IdArray arr) {
  arr = arr.CopyTo(DLContext{kDLCPU, 0});
  int64_t len = arr->shape[0];
  std::ostringstream oss;
  oss << "(" << len << ")[";
  if (arr->dtype.bits == 32) {
    int32_t* data = static_cast<int32_t*>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  } else {
    int64_t* data = static_cast<int64_t*>(arr->data);
    for (int64_t i = 0; i < len; ++i) {
      oss << data[i] << " ";
    }
  }
  oss << "]";
  return oss.str();
}

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DLContext& ctx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (utils::IsNoneArray(arrays[i]))
      continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
      << "Expected device context " << ctx << ". But got "
      << arrays[i]->ctx << " for " << names[i] << ".";
  }
}

// Check whether the given arguments use the same number of bits.
inline void CheckIdArray(
    const uint8_t bits,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (utils::IsNoneArray(arrays[i]))
      continue;
    CHECK(arrays[i]->dtype.code == kDLInt);
    CHECK_EQ(arrays[i]->ndim, 1);
    CHECK_EQ(bits, arrays[i]->dtype.bits)
      << "Expected " << bits << " integer array. But got "
      << arrays[i]->dtype.bits << " for " << names[i] << ".";
  }
}

// Return true if the operator is commutative and lhs and rhs need
// to be switched. For example, Add(kDst, kSrc) needs to be changed
// to Add(kSrc, kDst).
// This is because we only generate kernels for
//  Add(kSrc, kDst), Add(kDst, kEdge), Add(kSrc, kDst)
// to save compilation time.
inline bool NeedSwitchOrder(const std::string& op,
    binary_op::Target lhs, binary_op::Target rhs) {
  CHECK_NE(lhs, rhs);
  return (op == binary_op::kAdd || op == binary_op::kMul)
    && lhs > rhs;
}

class ImmutableGraphCSRWrapper : public CSRWrapper {
 public:
  explicit ImmutableGraphCSRWrapper(const ImmutableGraph* graph) :
    gptr_(graph) { }

  aten::CSRMatrix GetInCSRMatrix() const override {
    return gptr_->GetInCSR()->ToCSRMatrix();
  }

  aten::CSRMatrix GetOutCSRMatrix() const override {
    return gptr_->GetOutCSR()->ToCSRMatrix();
  }

  DGLContext Context() const override {
    return gptr_->Context();
  }

  int NumBits() const override {
    return gptr_->NumBits();
  }

 private:
  const ImmutableGraph* gptr_;
};

class UnitGraphCSRWrapper : public CSRWrapper {
 public:
  explicit UnitGraphCSRWrapper(const UnitGraph* graph) :
    gptr_(graph) { }

  aten::CSRMatrix GetInCSRMatrix() const override {
    return gptr_->GetInCSRMatrix();
  }

  aten::CSRMatrix GetOutCSRMatrix() const override {
    return gptr_->GetOutCSRMatrix();
  }

  DGLContext Context() const override {
    return gptr_->Context();
  }

  int NumBits() const override {
    return gptr_->NumBits();
  }

 private:
  const UnitGraph* gptr_;
};

}  // namespace


std::vector<int64_t> InferBinaryFeatureShape(
    const std::string& op,
    NDArray lhs,
    NDArray rhs) {
  return CalcBcastInfo(op, lhs, rhs).real_out_shape;
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelInferBinaryFeatureShape")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string op = args[0];
    NDArray lhs = args[1];
    NDArray rhs = args[2];
    const auto& shape = InferBinaryFeatureShape(op, lhs, rhs);
    const int64_t len = shape.size();
    NDArray ret = NDArray::Empty(
        {len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t* ret_data = static_cast<int64_t*>(ret->data);
    std::copy(shape.begin(), shape.end(), ret_data);
    *rv = ret;
  });

void BinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_data, NDArray rhs_data,
    NDArray out_data,
    NDArray lhs_mapping, NDArray rhs_mapping,
    NDArray out_mapping) {
  const auto& ctx = graph.Context();
  // sanity check
  CheckCtx(ctx,
      {lhs_data, rhs_data, out_data, lhs_mapping, rhs_mapping, out_mapping},
      {"lhs_data", "rhs_data", "out_data", "lhs_mapping", "rhs_mapping", "out_mapping"});
  CheckIdArray(graph.NumBits(),
      {lhs_mapping, rhs_mapping, out_mapping},
      {"lhs_mapping", "rhs_mapping", "out_mapping"});
  // Switch order for commutative operation
  if (NeedSwitchOrder(op, lhs, rhs)) {
    BinaryOpReduce(reducer, op, graph,
        rhs, lhs, rhs_data, lhs_data, out_data,
        rhs_mapping, lhs_mapping, out_mapping);
  } else {
    if (HasBcast(lhs_data, rhs_data)) {
      BcastInfo info = CalcBcastInfo(op, lhs_data, rhs_data);
      DGL_XPU_SWITCH(ctx.device_type, BinaryReduceBcastImpl,
          info, reducer, op, graph,
          lhs, rhs,
          lhs_data, rhs_data, out_data,
          lhs_mapping, rhs_mapping, out_mapping);
    } else {
      CHECK(IsValidBinaryOpShape(lhs_data, rhs_data))
        << "Cannot compute binary operation between feature shapes "
        << ShapeString(lhs_data) << " and " << ShapeString(rhs_data);
      DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
          reducer, op, graph,
          lhs, rhs,
          lhs_data, rhs_data, out_data,
          lhs_mapping, rhs_mapping, out_mapping);
    }
  }
}

// Comes from DGLArgValue::AsObjectRef() that allows argvalue to be either a GraphRef
// or a HeteroGraphRef
#define CSRWRAPPER_SWITCH(argvalue, wrapper, ...) do {            \
  DGLArgValue argval = (argvalue);                                \
  DGL_CHECK_TYPE_CODE(argval.type_code(), kObjectHandle);         \
  std::shared_ptr<Object>& sptr =                                 \
      *argval.ptr<std::shared_ptr<Object>>();                     \
  if (ObjectTypeChecker<GraphRef>::Check(sptr.get())) {           \
    GraphRef g = argval;                                          \
    auto igptr = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr()); \
    CHECK_NOTNULL(igptr);                                         \
    ImmutableGraphCSRWrapper wrapper(igptr.get());                \
    {__VA_ARGS__}                                                 \
  } else if (ObjectTypeChecker<HeteroGraphRef>::Check(sptr.get())) { \
    HeteroGraphRef g = argval;                                    \
    auto bgptr = std::dynamic_pointer_cast<UnitGraph>(g.sptr());  \
    CHECK_NOTNULL(bgptr);                                         \
    UnitGraphCSRWrapper wrapper(bgptr.get());                     \
    {__VA_ARGS__}                                                 \
  }                                                               \
} while (0)

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBinaryOpReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_data = args[5];
    NDArray rhs_data = args[6];
    NDArray out_data = args[7];
    NDArray lhs_mapping = args[8];
    NDArray rhs_mapping = args[9];
    NDArray out_mapping = args[10];

    CSRWRAPPER_SWITCH(args[2], wrapper, {
      BinaryOpReduce(reducer, op, wrapper,
          static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
          lhs_data, rhs_data, out_data,
          lhs_mapping, rhs_mapping, out_mapping);
      });
  });

void BackwardLhsBinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray out_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data) {
  const auto& ctx = graph.Context();
  // sanity check
  CheckCtx(ctx,
      {lhs_data, rhs_data, out_data, grad_out_data, grad_lhs_data,
       lhs_mapping, rhs_mapping, out_mapping},
      {"lhs_data", "rhs_data", "out_data", "grad_out_data", "grad_lhs_data",
       "lhs_mapping", "rhs_mapping", "out_mapping"});
  CheckIdArray(graph.NumBits(),
      {lhs_mapping, rhs_mapping, out_mapping},
      {"lhs_mapping", "rhs_mapping", "out_mapping"});
  // Switch order for commutative operation
  if (NeedSwitchOrder(op, lhs, rhs)) {
    BackwardRhsBinaryOpReduce(reducer, op, graph,
        rhs, lhs,
        rhs_mapping, lhs_mapping, out_mapping,
        rhs_data, lhs_data, out_data,
        grad_out_data, grad_lhs_data);
  } else {
    if (HasBcast(lhs_data, rhs_data)) {
      BcastInfo info = CalcBcastInfo(op, lhs_data, rhs_data);
      DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceBcastImpl,
          info, reducer, op, graph,
          lhs, rhs,
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          grad_lhs_data, utils::NoneArray());
    } else {
      DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceImpl,
          reducer, op, graph,
          lhs, rhs,
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          grad_lhs_data, utils::NoneArray());
    }
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardLhsBinaryOpReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_mapping = args[5];
    NDArray rhs_mapping = args[6];
    NDArray out_mapping = args[7];
    NDArray lhs_data = args[8];
    NDArray rhs_data = args[9];
    NDArray out_data = args[10];
    NDArray grad_out_data = args[11];
    NDArray grad_lhs_data = args[12];

    CSRWRAPPER_SWITCH(args[2], wrapper, {
      BackwardLhsBinaryOpReduce(
          reducer, op, wrapper,
          static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          grad_lhs_data);
    });
  });

void BackwardRhsBinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray out_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_rhs_data) {
  const auto& ctx = graph.Context();
  // sanity check
  CheckCtx(ctx,
      {lhs_data, rhs_data, out_data, grad_out_data, grad_rhs_data,
       lhs_mapping, rhs_mapping, out_mapping},
      {"lhs_data", "rhs_data", "out_data", "grad_out_data", "grad_rhs_data",
       "lhs_mapping", "rhs_mapping", "out_mapping"});
  CheckIdArray(graph.NumBits(),
      {lhs_mapping, rhs_mapping, out_mapping},
      {"lhs_mapping", "rhs_mapping", "out_mapping"});
  if (NeedSwitchOrder(op, lhs, rhs)) {
    BackwardLhsBinaryOpReduce(reducer, op, graph,
        rhs, lhs,
        rhs_mapping, lhs_mapping, out_mapping,
        rhs_data, lhs_data, out_data,
        grad_out_data, grad_rhs_data);
  } else {
    if (HasBcast(lhs_data, rhs_data)) {
      BcastInfo info = CalcBcastInfo(op, lhs_data, rhs_data);
      DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceBcastImpl,
          info, reducer, op, graph,
          lhs, rhs,
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          utils::NoneArray(), grad_rhs_data);
    } else {
      DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceImpl,
          reducer, op, graph,
          lhs, rhs,
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          utils::NoneArray(), grad_rhs_data);
    }
  }
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardRhsBinaryOpReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    std::string op = args[1];
    int lhs = args[3];
    int rhs = args[4];
    NDArray lhs_mapping = args[5];
    NDArray rhs_mapping = args[6];
    NDArray out_mapping = args[7];
    NDArray lhs_data = args[8];
    NDArray rhs_data = args[9];
    NDArray out_data = args[10];
    NDArray grad_out_data = args[11];
    NDArray grad_rhs_data = args[12];

    CSRWRAPPER_SWITCH(args[2], wrapper, {
      BackwardRhsBinaryOpReduce(
          reducer, op, wrapper,
          static_cast<binary_op::Target>(lhs), static_cast<binary_op::Target>(rhs),
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          grad_rhs_data);
    });
  });

void CopyReduce(
    const std::string& reducer,
    const CSRWrapper& graph,
    binary_op::Target target,
    NDArray in_data, NDArray out_data,
    NDArray in_mapping, NDArray out_mapping) {
  const auto& ctx = graph.Context();
  // sanity check
  CheckCtx(ctx,
      {in_data, out_data, in_mapping, out_mapping},
      {"in_data", "out_data", "in_mapping", "out_mapping"});
  CheckIdArray(graph.NumBits(),
      {in_mapping, out_mapping},
      {"in_mapping", "out_mapping"});
  DGL_XPU_SWITCH(ctx.device_type, BinaryReduceImpl,
      reducer, binary_op::kUseLhs, graph,
      target, binary_op::kNone,
      in_data, utils::NoneArray(), out_data,
      in_mapping, utils::NoneArray(), out_mapping);
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelCopyReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    int target = args[2];
    NDArray in_data = args[3];
    NDArray out_data = args[4];
    NDArray in_mapping = args[5];
    NDArray out_mapping = args[6];

    CSRWRAPPER_SWITCH(args[1], wrapper, {
      CopyReduce(reducer, wrapper,
          static_cast<binary_op::Target>(target),
          in_data, out_data,
          in_mapping, out_mapping);
    });
  });

void BackwardCopyReduce(
    const std::string& reducer,
    const CSRWrapper& graph,
    binary_op::Target target,
    NDArray in_mapping,
    NDArray out_mapping,
    NDArray in_data,
    NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_in_data) {
  const auto& ctx = graph.Context();
  // sanity check
  CheckCtx(ctx,
      {in_data, out_data, grad_out_data, grad_in_data, in_mapping, out_mapping},
      {"in_data", "out_data", "grad_out_data", "grad_in_data", "in_mapping", "out_mapping"});
  CheckIdArray(graph.NumBits(),
      {in_mapping, out_mapping},
      {"in_mapping", "out_mapping"});
  if (!utils::IsNoneArray(out_mapping)) {
    CHECK_EQ(ctx, out_mapping->ctx) << "Expected device context " << ctx
      << ". But got " << out_mapping->ctx << " for rhs_data.";
  }
  DGL_XPU_SWITCH(ctx.device_type, BackwardBinaryReduceImpl,
      reducer, binary_op::kUseLhs, graph,
      target, binary_op::kNone,
      in_mapping, utils::NoneArray(), out_mapping,
      in_data, utils::NoneArray(), out_data, grad_out_data,
      grad_in_data, utils::NoneArray());
}

DGL_REGISTER_GLOBAL("kernel._CAPI_DGLKernelBackwardCopyReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string reducer = args[0];
    int target = args[2];
    NDArray in_data = args[3];
    NDArray out_data = args[4];
    NDArray grad_out_data = args[5];
    NDArray grad_in_data = args[6];
    NDArray in_mapping = args[7];
    NDArray out_mapping = args[8];

    CSRWRAPPER_SWITCH(args[1], wrapper, {
      BackwardCopyReduce(
          reducer, wrapper, static_cast<binary_op::Target>(target),
          in_mapping, out_mapping,
          in_data, out_data, grad_out_data,
          grad_in_data);
    });
  });

}  // namespace kernel
}  // namespace dgl
