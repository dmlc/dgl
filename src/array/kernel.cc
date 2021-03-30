/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel.cc
 * \brief New kernels
 */
#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>

#ifdef USE_TVM
#include <featgraph.h>
#endif  // USE_TVM

#include "kernel_decl.h"
#include "../c_api_common.h"
#include "./check.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace {

}  // namespace

/*! \brief Generalized Sparse Matrix-Matrix Multiplication. */
void SpMM(const std::string& op, const std::string& reduce,
          HeteroGraphPtr graph,
          NDArray ufeat,
          NDArray efeat,
          NDArray out,
          std::vector<NDArray> out_aux) {
  // TODO(zihao): format tuning
  SparseFormat format = graph->SelectFormat(0, CSC_CODE);
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out->dtype, bits, "Feature data", {
        if (format == SparseFormat::kCSC) {
          SpMMCsr<XPU, IdType, bits>(
              op, reduce, bcast, graph->GetCSCMatrix(0),
              ufeat, efeat, out, out_aux);
        } else if (format == SparseFormat::kCOO) {
          SpMMCoo<XPU, IdType, bits>(
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
  SparseFormat format = graph->SelectFormat(0, COO_CODE);
  const auto &bcast = CalcBcastOff(op, lhs, rhs);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out->dtype, bits, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsr<XPU, IdType, bits>(
              op, bcast, graph->GetCSRMatrix(0),
              lhs, rhs, out, lhs_target, rhs_target);
        } else if (format == SparseFormat::kCOO) {
          SDDMMCoo<XPU, IdType, bits>(
              op, bcast, graph->GetCOOMatrix(0),
              lhs, rhs, out, lhs_target, rhs_target);
        } else {
          LOG(FATAL) << "SDDMM only supports CSR and COO foramts";
        }
      });
    });
  });
}

NDArray GetEdgeMapping(HeteroGraphRef graph) {
  SparseFormat format = graph->SelectFormat(0, CSC_CODE);
  if (format == SparseFormat::kCSC) {
    return graph.sptr()->GetCSCMatrix(0).data;
  } else {
    return NullArray();
  }
}

/*! \brief Segment reduce dispatch function. */
void SegmentReduceDispatch(const std::string& op,
                           NDArray feat,
                           NDArray offsets,
                           NDArray out,
                           NDArray arg) {
  ATEN_XPU_SWITCH_CUDA(feat->ctx.device_type, XPU, "SegmentReduce", {
    ATEN_ID_TYPE_SWITCH(offsets->dtype, IdType, {
      ATEN_FLOAT_BITS_SWITCH(feat->dtype, bits, "Feature data", {
          SegmentReduce<XPU, IdType, bits>(op, feat, offsets, out, arg);
      });
    });
  });
}

/*! \brief Scatter Add (on first dimension) dispatch function. */
void ScatterAddDispatch(NDArray feat, NDArray idx, NDArray out) {
  ATEN_XPU_SWITCH_CUDA(feat->ctx.device_type, XPU, "ScatterAdd", {
    ATEN_ID_TYPE_SWITCH(idx->dtype, IdType, {
      ATEN_FLOAT_BITS_SWITCH(feat->dtype, bits, "Feature data", {
        ScatterAdd<XPU, IdType, bits>(feat, idx, out);
      });
    });
  });
}

/*! \brief Backward segment cmp dispatch function.*/
void BackwardSegmentCmpDispatch(NDArray feat, NDArray arg, NDArray out) {
  ATEN_XPU_SWITCH_CUDA(feat->ctx.device_type, XPU, "BackwardSegmentCmp", {
    ATEN_ID_TYPE_SWITCH(arg->dtype, IdType, {
      ATEN_FLOAT_BITS_SWITCH(feat->dtype, bits, "Feature data", {
        BackwardSegmentCmp<XPU, IdType, bits>(feat, arg, out);
      });
    });
  });
}

std::pair<CSRMatrix, NDArray> CSRMM(
    CSRMatrix A,
    NDArray A_weights,
    CSRMatrix B,
    NDArray B_weights) {
  CheckCtx(
      A.indptr->ctx,
      {A_weights, B_weights},
      {"A's edge weights", "B's edge weights"});
  CHECK_EQ(A.indptr->ctx, B.indptr->ctx) << "Device of two graphs must match.";
  CHECK_EQ(A.indptr->dtype, B.indptr->dtype) << "ID types of two graphs must match.";
  CHECK_EQ(A_weights->dtype, B_weights->dtype) << "Data types of two edge weights must match.";

  std::pair<CSRMatrix, NDArray> ret;
  // TODO(BarclayII): change to ATEN_XPU_SWITCH_CUDA once the GPU kernels are implemented
  ATEN_XPU_SWITCH(A.indptr->ctx.device_type, XPU, "CSRMM", {
    ATEN_ID_TYPE_SWITCH(A.indptr->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH(A_weights->dtype, DType, "Edge weights", {
        ret = CSRMM<XPU, IdType, DType>(A, A_weights, B, B_weights);
      });
    });
  });
  return ret;
}

std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A,
    const std::vector<NDArray>& A_weights) {
  CHECK(A.size() > 0) << "The list of graphs must not be empty.";
  CHECK_EQ(A.size(), A_weights.size()) <<
    "The list of edge weights must have the same length as the list of graphs.";
  auto ctx = A[0].indptr->ctx;
  auto idtype = A[0].indptr->dtype;
  auto dtype = A_weights[0]->dtype;
  for (size_t i = 0; i < A.size(); ++i) {
    CHECK_EQ(A[i].indptr->ctx, ctx) << "The devices of all graphs must be equal.";
    CHECK_EQ(A[i].indptr->dtype, idtype) << "The ID types of all graphs must be equal.";
    CHECK_EQ(A[i].indices->shape[0], A_weights[i]->shape[0]) <<
      "Shape of edge weights does not match the number of edges.";
    CHECK_EQ(A_weights[i]->ctx, ctx) <<
      "The devices of edge weights must be the same as that of the graphs.";
    CHECK_EQ(A_weights[i]->dtype, dtype) <<
      "The data types of all edge weights must be equal.";
  }

  std::pair<CSRMatrix, NDArray> ret;
  // TODO(BarclayII): change to ATEN_XPU_SWITCH_CUDA once the GPU kernels are implemented
  ATEN_XPU_SWITCH(ctx.device_type, XPU, "CSRSum", {
    ATEN_ID_TYPE_SWITCH(idtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH(dtype, DType, "Edge weights", {
        ret = CSRSum<XPU, IdType, DType>(A, A_weights);
      });
    });
  });
  return ret;
}

NDArray CSRMask(const CSRMatrix& A, NDArray A_weights, const CSRMatrix& B) {
  CHECK_EQ(A.indptr->ctx, A_weights->ctx) <<
    "Device of the graph and the edge weights must match.";
  CHECK_EQ(A.indptr->ctx, B.indptr->ctx) << "Device of two graphs must match.";
  CHECK_EQ(A.indptr->dtype, B.indptr->dtype) << "ID types of two graphs must match.";
  CHECK_EQ(A_weights->shape[0], A.indices->shape[0]) <<
    "Shape of edge weights does not match the number of edges.";
  auto ctx = A.indptr->ctx;
  auto idtype = A.indptr->dtype;
  auto dtype = A_weights->dtype;

  NDArray ret;
  // TODO(BarclayII): change to ATEN_XPU_SWITCH_CUDA once the GPU kernels are implemented
  ATEN_XPU_SWITCH(ctx.device_type, XPU, "CSRMask", {
    ATEN_ID_TYPE_SWITCH(idtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH(dtype, DType, "Edge weights", {
        ret = CSRMask<XPU, IdType, DType>(A, A_weights, B);
      });
    });
  });
  return ret;
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

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSegmentReduce")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string op = args[0];
    NDArray feat = args[1];
    NDArray offsets = args[2];
    NDArray out = args[3];
    NDArray arg = args[4];
    CheckCtx(feat->ctx, {feat, offsets, out}, {"feat", "offsets", "out"});
    CheckContiguous({feat, offsets, out}, {"feat", "offsets", "out"});
    SegmentReduceDispatch(op, feat, offsets, out, arg);
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelScatterAdd")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    NDArray feat = args[0];
    NDArray idx = args[1];
    NDArray out = args[2];
    CheckCtx(feat->ctx, {feat, idx, out}, {"feat", "idx", "out"});
    CheckContiguous({feat, idx, out}, {"feat", "idx", "out"});
    ScatterAddDispatch(feat, idx, out);
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelBwdSegmentCmp")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    NDArray feat = args[0];
    NDArray arg = args[1];
    NDArray out = args[2];
    CheckCtx(feat->ctx, {feat, arg, out}, {"feat", "arg", "out"});
    CheckContiguous({feat, arg, out}, {"feat", "arg", "out"});
    BackwardSegmentCmpDispatch(feat, arg, out);
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelGetEdgeMapping")
.set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef graph = args[0];
    *rv = GetEdgeMapping(graph);
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRMM")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int M = args[0];
    int N = args[1];
    int P = args[2];
    NDArray A_indptr = args[3];
    NDArray A_indices = args[4];
    NDArray A_data = args[5];
    NDArray B_indptr = args[6];
    NDArray B_indices = args[7];
    NDArray B_data = args[8];
    auto result = CSRMM(
        CSRMatrix(M, N, A_indptr, A_indices),
        A_data,
        CSRMatrix(N, P, B_indptr, B_indices),
        B_data);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first.indptr)));
    ret.push_back(Value(MakeValue(result.first.indices)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRSum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int M = args[0];
    int N = args[1];
    List<Value> A_indptr = args[2];
    List<Value> A_indices = args[3];
    List<Value> A_data = args[4];
    std::vector<NDArray> weights = ListValueToVector<NDArray>(A_data);
    std::vector<CSRMatrix> mats(A_indptr.size());
    for (int i = 0; i < A_indptr.size(); ++i)
      mats[i] = CSRMatrix(M, N, A_indptr[i]->data, A_indices[i]->data);
    auto result = CSRSum(mats, weights);
    List<Value> ret;
    ret.push_back(Value(MakeValue(result.first.indptr)));
    ret.push_back(Value(MakeValue(result.first.indices)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRMask")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int M = args[0];
    int N = args[1];
    NDArray A_indptr = args[2];
    NDArray A_indices = args[3];
    NDArray A_data = args[4];
    NDArray B_indptr = args[5];
    NDArray B_indices = args[6];
    auto result = CSRMask(
        CSRMatrix(M, N, A_indptr, A_indices),
        A_data,
        CSRMatrix(M, N, B_indptr, B_indices));
    *rv = result;
  });

#ifdef USE_TVM
DGL_REGISTER_GLOBAL("sparse._CAPI_FG_LoadModule")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string path = args[0];
    dgl::featgraph::LoadFeatGraphModule(path);
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_FG_SDDMMTreeReduction")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    NDArray lhs = args[1];
    NDArray rhs = args[2];
    NDArray out = args[3];
    CheckCtx(graph->Context(), {lhs, rhs, out}, {"lhs", "rhs", "out"});
    CheckContiguous({lhs, rhs, out}, {"lhs", "rhs", "out"});
    CHECK_EQ(graph->NumEdgeTypes(), 1);
    // auto pair = graph->meta_graph()->FindEdge(0);  // only one etype in the graph.
    // const dgl_type_t src_vtype = pair.first;
    // const dgl_type_t dst_vtype = pair.second;
    // CheckShape(
    //     {graph->NumVertices(src_vtype), graph->NumEdges(0), graph->NumVertices(dst_vtype)},
    //     {lhs_target, rhs_target, 1},
    //     {lhs, rhs, out},
    //     {"U_data", "E_data", "V_data"});
    COOMatrix coo = graph.sptr()->GetCOOMatrix(0);
    dgl::featgraph::SDDMMTreeReduction(coo.row.ToDLPack(), coo.col.ToDLPack(),
                                       lhs.ToDLPack(), rhs.ToDLPack(), out.ToDLPack());
  });
#endif  // USE_TVM


}  // namespace aten
}  // namespace dgl
