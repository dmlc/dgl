/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel.cc
 * \brief New kernels
 */
#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>

#include "kernel_decl.h"
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace {

// Check whether the given arguments have the same context.
inline void CheckCtx(
    const DLContext& ctx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (IsNullArray(arrays[i]))
      continue;
    CHECK_EQ(ctx, arrays[i]->ctx)
      << "Expected device context " << ctx << ". But got "
      << arrays[i]->ctx << " for " << names[i] << ".";
  }
}

// Check whether input tensors are contiguous.
inline void CheckContiguous(
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (IsNullArray(arrays[i]))
      continue;
    CHECK(arrays[i].IsContiguous())
      << "Expect " << names[i] << " to be a contiguous tensor";
  }
}

// Check whether input tensors have valid shape.
inline void CheckShape(
    const std::vector<uint64_t>& gdim,
    const std::vector<int>& uev_idx,
    const std::vector<NDArray>& arrays,
    const std::vector<std::string>& names) {
  for (size_t i = 0; i < arrays.size(); ++i) {
    if (IsNullArray(arrays[i]))
      continue;
    CHECK_GE(arrays[i]->ndim, 2)
      << "Expect " << names[i] << " to have ndim >= 2, "
      << "Note that for scalar feature we expand its "
      << "dimension with an additional dimension of "
      << "length one.";
    CHECK_EQ(gdim[uev_idx[i]], arrays[i]->shape[0])
      << "Expect " << names[i] << " to have size "
      << gdim[uev_idx[i]] << " on the first dimension, "
      << "but got " << arrays[i]->shape[0];
  }
}

}  // namespace

/*! \brief Generalized Sparse Matrix-Matrix Multiplication. */
void SpMM(const std::string& op, const std::string& reduce,
          HeteroGraphPtr graph,
          NDArray ufeat,
          NDArray efeat,
          NDArray out,
          std::vector<NDArray> out_aux) {
  // TODO(zihao): format tuning
  SparseFormat format = graph->SelectFormat(0, csc_code);
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(out->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSC) {
          SpMMCsr<XPU, IdType, DType>(
              op, reduce, bcast, graph->GetCSCMatrix(0),
              ufeat, efeat, out, out_aux);
        } else if (format == SparseFormat::kCOO) {
          SpMMCoo<XPU, IdType, DType>(
              op, reduce, bcast, graph->GetCOOMatrix(0),
              ufeat, efeat, out, out_aux);
        } else {
          LOG(FATAL) << "SpMM only supports CSC and COO foramts";
        }
      });
    });
  });
}

/*! \brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void SDDMM(const std::string& op,
           HeteroGraphPtr graph,
           NDArray lhs,
           NDArray rhs,
           NDArray out,
           int lhs_target,
           int rhs_target) {
  // TODO(zihao): format tuning
  SparseFormat format = graph->SelectFormat(0, coo_code);
  const auto &bcast = CalcBcastOff(op, lhs, rhs);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH(out->dtype, DType, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsr<XPU, IdType, DType>(
              op, bcast, graph->GetCSRMatrix(0),
              lhs, rhs, out, lhs_target, rhs_target);
        } else if (format == SparseFormat::kCOO) {
          SDDMMCoo<XPU, IdType, DType>(
              op, bcast, graph->GetCOOMatrix(0),
              lhs, rhs, out, lhs_target, rhs_target);
        } else {
          LOG(FATAL) << "SDDMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSpMM")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    const std::string op = args[1];
    const std::string reduce_op = args[2];
    NDArray U = args[3];
    NDArray E = args[4];
    NDArray V = args[5];
    NDArray ArgU = args[6];
    NDArray ArgE = args[7];
    CheckCtx(graph->Context(), {U, E, V, ArgU, ArgE},
        {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
    CheckContiguous({U, E, V, ArgU, ArgE},
        {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
    auto pair = graph->meta_graph()->FindEdge(0);  // only one etype in the graph.
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    CheckShape(
        {graph->NumVertices(src_vtype), graph->NumEdges(0), graph->NumVertices(dst_vtype)},
        {0, 1, 2, 2, 2},
        {U, E, V, ArgU, ArgE},
        {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
    SpMM(op, reduce_op, graph.sptr(), U, E, V, {ArgU, ArgE});
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSDDMM")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    const std::string op = args[1];
    NDArray lhs = args[2];
    NDArray rhs = args[3];
    NDArray out = args[4];
    int lhs_target = args[5];
    int rhs_target = args[6];
    CheckCtx(graph->Context(), {lhs, rhs, out}, {"lhs", "rhs", "out"});
    CheckContiguous({lhs, rhs, out}, {"lhs", "rhs", "out"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
    auto pair = graph->meta_graph()->FindEdge(0);  // only one etype in the graph.
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    CheckShape(
        {graph->NumVertices(src_vtype), graph->NumEdges(0), graph->NumVertices(dst_vtype)},
        {lhs_target, rhs_target, 1},
        {lhs, rhs, out},
        {"U_data", "E_data", "V_data"});
    SDDMM(op, graph.sptr(), lhs, rhs, out, lhs_target, rhs_target);
  });

}  // namespace aten
}  // namespace dgl
