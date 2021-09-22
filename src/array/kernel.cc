/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel.cc
 * \brief New kernels
 */
#include <dgl/packed_func_ext.h>
#include <dgl/base_heterograph.h>
#include <vector>
#include <stdint.h>
#include <x86intrin.h>
#include <omp.h>

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
          LOG(FATAL) << "SpMM only supports CSC and COO formats";
        }
      });
    });
  });
}

/*! \brief Generalized Sparse Matrix-Matrix Multiplication with hetero-graph support. */
void SpMMHetero(const std::string& op, const std::string& reduce,
          HeteroGraphPtr graph,
          std::vector<NDArray> ufeat_vec,
          std::vector<NDArray> efeat_vec,
          std::vector<NDArray> out,
          std::vector<NDArray> out_aux) {
  SparseFormat format = graph->SelectFormat(0, CSC_CODE);

  std::vector<CSRMatrix> vec_graph;
  std::vector<dgl_type_t> ufeat_eid;
  std::vector<dgl_type_t> efeat_eid;
  std::vector<dgl_type_t> out_eid;
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    vec_graph.push_back(graph->GetCSCMatrix(etype));
    auto pair = graph->meta_graph()->FindEdge(etype);
    ufeat_eid.push_back(pair.first);
    efeat_eid.push_back(etype);
    out_eid.push_back(pair.second);
  }
  NDArray efeat = (efeat_vec.size() == 0) ? NullArray() : efeat_vec[efeat_eid[0]];
  NDArray ufeat = (ufeat_vec.size() == 0) ? NullArray() : ufeat_vec[ufeat_eid[0]];
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out[out_eid[0]]->dtype, bits, "Feature data", {
        if (format == SparseFormat::kCSC) {
          SpMMCsrHetero<XPU, IdType, bits>(
              op, reduce, bcast, vec_graph,
              ufeat_vec, efeat_vec, out, out_aux,
              ufeat_eid, out_eid);
        } else {
          // TODO(Israt): Add support for COO format
          LOG(FATAL) << "SpMM only supports CSC format for graphs with number "
                     << "of relation types > 1";
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
          LOG(FATAL) << "SDDMM only supports CSR and COO formats";
        }
      });
    });
  });
}


/*!
 * \brief Find the src/dst/etype id based on the target 'u', 'v' or 'e'.
 *
 * \param graph The input graph.
 * \param target 'u', 'v' or 'e'. The target of the lhs or rhs data of an etype.
 * \param etype Relation type of the input graph.
 */
int get_typeid_by_target(HeteroGraphPtr graph, int target, dgl_type_t etype) {
  auto pair = graph->meta_graph()->FindEdge(etype);
  if (target == 0)
    return pair.first;
  if (target == 2)
    return pair.second;
  return etype;
}


/*! \brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void SDDMMHetero(const std::string& op,
           HeteroGraphPtr graph,
           std::vector<NDArray> lhs,
           std::vector<NDArray> rhs,
           std::vector<NDArray> out,
           int lhs_target,
           int rhs_target) {
  // TODO(Israt): change it to COO_CODE
  SparseFormat format = graph->SelectFormat(0, CSR_CODE);

  std::vector<CSRMatrix> vec_csr;
  std::vector<dgl_type_t> lhs_eid;
  std::vector<dgl_type_t> rhs_eid;
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    vec_csr.push_back(graph->GetCSRMatrix(etype));
    lhs_eid.push_back(get_typeid_by_target(graph, lhs_target, etype));
    rhs_eid.push_back(get_typeid_by_target(graph, rhs_target, etype));
  }
  const auto &bcast = CalcBcastOff(op, lhs[lhs_eid[0]], rhs[rhs_eid[0]]);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_BITS_SWITCH(out[rhs_eid[0]]->dtype, bits, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsrHetero<XPU, IdType, bits>(
              op, bcast, vec_csr,
              lhs, rhs, out, lhs_target, rhs_target,
              lhs_eid, rhs_eid);
        } else {
          // TODO(Israt): Add support for COO format
          LOG(FATAL) << "SDDMM only supports CSC format for graphs with number "
                     << "of relation types > 1";
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
  CHECK_EQ(A.num_cols, B.num_rows) <<
    "The number of nodes of destination node type of the first graph must be the "
    "same as the number of nodes of source node type of the second graph.";
  CheckCtx(
      A.indptr->ctx,
      {A_weights, B_weights},
      {"A's edge weights", "B's edge weights"});
  CHECK_EQ(A.indptr->ctx, B.indptr->ctx) << "Device of two graphs must match.";
  CHECK_EQ(A.indptr->dtype, B.indptr->dtype) << "ID types of two graphs must match.";
  CHECK_EQ(A_weights->dtype, B_weights->dtype) << "Data types of two edge weights must match.";

  std::pair<CSRMatrix, NDArray> ret;
  ATEN_XPU_SWITCH_CUDA(A.indptr->ctx.device_type, XPU, "CSRMM", {
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
  const auto ctx = A[0].indptr->ctx;
  const auto idtype = A[0].indptr->dtype;
  const auto dtype = A_weights[0]->dtype;
  const auto num_rows = A[0].num_rows;
  const auto num_cols = A[0].num_cols;
  for (size_t i = 0; i < A.size(); ++i) {
    CHECK_EQ(A[i].indptr->ctx, ctx) << "The devices of all graphs must be equal.";
    CHECK_EQ(A[i].indptr->dtype, idtype) << "The ID types of all graphs must be equal.";
    CHECK_EQ(A[i].indices->shape[0], A_weights[i]->shape[0]) <<
      "Shape of edge weights does not match the number of edges.";
    CHECK_EQ(A_weights[i]->ctx, ctx) <<
      "The devices of edge weights must be the same as that of the graphs.";
    CHECK_EQ(A_weights[i]->dtype, dtype) <<
      "The data types of all edge weights must be equal.";
    CHECK_EQ(A[i].num_rows, num_rows) << "Graphs must have the same number of nodes.";
    CHECK_EQ(A[i].num_cols, num_cols) << "Graphs must have the same number of nodes.";
  }

  std::pair<CSRMatrix, NDArray> ret;
  ATEN_XPU_SWITCH_CUDA(ctx.device_type, XPU, "CSRSum", {
    ATEN_ID_TYPE_SWITCH(idtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH(dtype, DType, "Edge weights", {
        ret = CSRSum<XPU, IdType, DType>(A, A_weights);
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

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSpMMHetero")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    const std::string op = args[1];
    const std::string reduce_op = args[2];
    List<Value> list_U = args[3];
    List<Value> list_E = args[4];
    List<Value> list_V = args[5];
    NDArray ArgU = args[6];
    NDArray ArgE = args[7];
    std::vector<NDArray> U_vec;
    std::vector<NDArray> V_vec;
    std::vector<NDArray> E_vec;
    U_vec.reserve(list_U.size());
    V_vec.reserve(list_V.size());
    E_vec.reserve(list_E.size());
    for (Value val : list_U) {
      U_vec.push_back(val->data);
    }
    for (Value val : list_V) {
      V_vec.push_back(val->data);
    }
    for (Value val : list_E) {
      E_vec.push_back(val->data);
    }
    for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
      auto pair = graph->meta_graph()->FindEdge(etype);
      const dgl_id_t src_id = pair.first;
      const dgl_id_t dst_id = pair.second;
      NDArray U = (U_vec.size() == 0) ? NullArray() : U_vec[src_id];
      NDArray E = (E_vec.size() == 0) ? NullArray() : E_vec[etype];
      CheckCtx(graph->Context(), {U, E, V_vec[dst_id], ArgU, ArgE},
          {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
      CheckContiguous({U, E, V_vec[dst_id], ArgU, ArgE},
          {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
    }
    SpMMHetero(op, reduce_op, graph.sptr(), U_vec, E_vec, V_vec, {ArgU, ArgE});
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


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSDDMMHetero")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef graph = args[0];
    const std::string op = args[1];
    List<Value> list_lhs = args[2];
    List<Value> list_rhs = args[3];
    List<Value> list_out = args[4];
    int lhs_target = args[5];
    int rhs_target = args[6];
    std::vector<NDArray> vec_lhs;
    std::vector<NDArray> vec_rhs;
    std::vector<NDArray> vec_out;

    vec_lhs.reserve(list_lhs.size());
    vec_rhs.reserve(list_rhs.size());
    vec_out.reserve(list_out.size());

    for (Value val : list_lhs) {
      vec_lhs.push_back(val->data);
    }
    for (Value val : list_rhs) {
      vec_rhs.push_back(val->data);
    }
    for (Value val : list_out) {
      vec_out.push_back(val->data);
    }
    SDDMMHetero(op, graph.sptr(), vec_lhs, vec_rhs, vec_out, lhs_target, rhs_target);
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

/*!
 * \brief Sparse matrix multiplication with graph interface.
 *
 * \param A_ref The left operand.
 * \param A_weights The edge weights of graph A.
 * \param B_ref The right operand.
 * \param B_weights The edge weights of graph B.
 * \param num_vtypes The number of vertex types of the graph to be returned.
 * \return A pair consisting of the new graph as well as its edge weights.
 */
DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRMM")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const HeteroGraphRef A_ref = args[0];
    NDArray A_weights = args[1];
    const HeteroGraphRef B_ref = args[2];
    NDArray B_weights = args[3];
    int num_vtypes = args[4];

    const HeteroGraphPtr A = A_ref.sptr();
    const HeteroGraphPtr B = B_ref.sptr();
    CHECK_EQ(A->NumEdgeTypes(), 1) << "The first graph must have only one edge type.";
    CHECK_EQ(B->NumEdgeTypes(), 1) << "The second graph must have only one edge type.";
    const auto A_csr = A->GetCSRMatrix(0);
    const auto B_csr = B->GetCSRMatrix(0);
    auto result = CSRMM(A_csr, A_weights, B_csr, B_weights);

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(CreateFromCSR(num_vtypes, result.first, ALL_CODE)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRSum")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    List<HeteroGraphRef> A_refs = args[0];
    List<Value> A_weights = args[1];

    std::vector<NDArray> weights = ListValueToVector<NDArray>(A_weights);
    std::vector<CSRMatrix> mats;
    mats.reserve(A_refs.size());
    int num_vtypes = 0;
    for (auto A_ref : A_refs) {
      const HeteroGraphPtr A = A_ref.sptr();
      CHECK_EQ(A->NumEdgeTypes(), 1) << "Graphs must have only one edge type.";
      mats.push_back(A->GetCSRMatrix(0));
      if (num_vtypes == 0)
        num_vtypes = A->NumVertexTypes();
    }
    auto result = CSRSum(mats, weights);

    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(CreateFromCSR(num_vtypes, result.first, ALL_CODE)));
    ret.push_back(Value(MakeValue(result.second)));
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRMask")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const HeteroGraphRef A_ref = args[0];
    NDArray A_weights = args[1];
    const HeteroGraphRef B_ref = args[2];

    const HeteroGraphPtr A = A_ref.sptr();
    const HeteroGraphPtr B = B_ref.sptr();
    CHECK_EQ(A->NumEdgeTypes(), 1) << "Both graphs must have only one edge type.";
    CHECK_EQ(B->NumEdgeTypes(), 1) << "Both graphs must have only one edge type.";
    const CSRMatrix& A_csr = A->GetCSRMatrix(0);
    const COOMatrix& B_coo = B->GetCOOMatrix(0);
    CHECK_EQ(A_csr.num_rows, B_coo.num_rows) <<
      "Both graphs must have the same number of nodes.";
    CHECK_EQ(A_csr.num_cols, B_coo.num_cols) <<
      "Both graphs must have the same number of nodes.";

    NDArray result;
    ATEN_FLOAT_TYPE_SWITCH(A_weights->dtype, DType, "Edge weights", {
      result = aten::CSRGetData<DType>(A_csr, B_coo.row, B_coo.col, A_weights, 0.);
    });
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

/**************************************************************************************/
/* DistGNN functions */

template<typename IdType>
int32_t Ver2partition(IdType in_val, int32_t* node_map, int32_t num_parts) {
  int32_t pos = 0;
  for (int32_t p=0; p<num_parts; p++) {
    if (in_val < node_map[p])
      return pos;
    pos = pos + 1;
  }
  LOG(FATAL) << "Error: Unexpected output in Ver2partition!";
  //printf("Error: func Ver2partition\n");
  // exit(EXIT_FAILURE);
}


int32_t Map2localIndex(int32_t otf, int32_t *node_map,
                       int32_t num_parts, int32_t &pid,
                       int32_t cur_part) {
  int32_t last_p = 0;
  if (cur_part != 0)
    last_p = node_map[cur_part - 1];
  for (int32_t i=cur_part; i<num_parts; i++) {
    int32_t nnodes = node_map[i];
    if (otf < nnodes) {
      pid = i;
      return otf - last_p;
    }
    last_p = nnodes;
  }
  LOG(FATAL) << "Error: Unexpected output in Map2localIndex!";
  // printf("Error: func Map2localIndex\n");
  // exit(EXIT_FAILURE);
}


template<typename IdType, typename DType>
void FdrpaGatherEmbLR(
  NDArray feat_a,
  int64_t feat_shape,
  NDArray adj_a,
  NDArray send_feat_list_a,
  int64_t offset,
  NDArray send_node_list_a,
  NDArray send_to_node_list_a,    
  NDArray selected_nodes_a,
  NDArray in_degs_a,
  NDArray ver2part_a,
  NDArray ver2part_index_a,
  int32_t width,
  int32_t feat_size,
  int32_t cur_part,
  int64_t soffset_base,
  int64_t soffset_cur,
  NDArray node_map_a,
  int32_t num_parts) {
  
  if (soffset_cur == 0) return;
        
  DType*   feat              = feat_a.Ptr<DType>();
  IdType*  adj               = adj_a.Ptr<IdType>();
  DType*   send_feat_list    = send_feat_list_a.Ptr<DType>() + offset * (feat_size + 1);
  int32_t* send_node_list    = send_node_list_a.Ptr<int32_t>();
  int32_t* send_to_node_list = send_to_node_list_a.Ptr<int32_t>() + offset;
  int32_t* selected_nodes    = selected_nodes_a.Ptr<int32_t>();
  DType*   in_degs           = in_degs_a.Ptr<DType>();
  int32_t* ver2part          = ver2part_a.Ptr<int32_t>();
  int32_t* ver2part_index    = ver2part_index_a.Ptr<int32_t>();
  int32_t* node_map          = node_map_a.Ptr<int32_t>();
    
  int32_t pos = 0, pos_out = 0;
  int32_t len = selected_nodes_a.GetSize() >> 2;
    
  int32_t *gindex = (int32_t*)calloc(sizeof(int32_t), len);
  int32_t lim = send_node_list_a.GetSize() >> 2;
  int32_t counter = 0;
  pos = 0;
  
  for (int32_t i=0; i<len; i++) {
    if (ver2part[i] == cur_part) {
      if (counter >= soffset_base && pos < soffset_cur) {
        gindex[pos++] = i;
      }
      counter ++;
    }
  }    
  CHECK(pos == soffset_cur);
  
  #pragma omp parallel for
  for (int64_t i=0; i<pos; i++)
  {
    int32_t id = gindex[i];
    int64_t index = selected_nodes[id];
    
    #if DEBUG  //checks
    if (index >= feat_shape) {
      printf("Error: at index: %d, feat_shape: %d\n", index, feat_shape);
      fflush(0);
      CHECK(index < feat_shape);
    }       
    CHECK(index >= 0);
    #endif
        
    DType *iptr = feat + index * feat_size;        
    send_node_list[i] = index;        
    DType *optr = send_feat_list + i * (feat_size + 1);
    optr[0] = in_degs[index];
    send_to_node_list[i] = ver2part_index[id];
    
    #if DEBUG   // checks
    int32_t  p = Ver2partition<int32_t>(ver2part_index[id], node_map, num_parts);
    if (cur_part != p)
      LOG(FATAL) << "Error: Cur_part not matching !";
      // printf("Error: at not matching: %d %d\n", cur_part, p);
    CHECK(cur_part == p);
    #endif
        
    #pragma omp simd
    for (int64_t k=0; k<feat_size; k++)
      optr[1 + k] = iptr[k];
  }
  free(gindex);
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaGatherEmbLR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray feat              = args[0];
  int64_t feat_shape        = args[1];
  NDArray adj               = args[2];
  NDArray send_feat_list    = args[3];
  int64_t offset            = args[4];
  NDArray send_node_list    = args[5];
  NDArray send_to_node_list = args[6];
  NDArray selected_nodes    = args[7];
  NDArray in_degs           = args[8];
  NDArray ver2part          = args[9];
  NDArray ver2part_index    = args[10];
  int32_t width             = args[11];
  int32_t feat_size         = args[12];
  int32_t cur_part          = args[13];
  int64_t soffset_base      = args[14];
  int64_t soffset_cur       = args[15];
  NDArray node_map          = args[16];
  int32_t num_parts         = args[17];

  ATEN_ID_TYPE_SWITCH(adj->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Feature data", {
          FdrpaGatherEmbLR<IdType, DType>(feat, feat_shape,
                                          adj, send_feat_list,
                                          offset, send_node_list,
                                          send_to_node_list, selected_nodes,
                                          in_degs, ver2part,
                                          ver2part_index, width,
                                          feat_size, cur_part,
                                          soffset_base, soffset_cur,
                                          node_map, num_parts);          
        });
    });
});


template<typename IdType, typename DType>
void ScatterReduceLR(
  NDArray otf_a,
  int64_t offsetf,
  NDArray otn_a,
  int64_t offsetn,    
  NDArray feat_a,    
  NDArray in_degs_a,
  NDArray node_map_a,
  int64_t dim,
  int64_t feat_size,
  int64_t num_parts,
  NDArray recv_list_nodes_a,
  NDArray pos_a,
  int64_t lim,
  int32_t cur_part) {
  
  DType*   otf             = otf_a.Ptr<DType>() + offsetf;
  int32_t* otn             = otn_a.Ptr<int32_t>() + offsetn;
  DType*   feat            = feat_a.Ptr<DType>();
  DType*  in_degs          = in_degs_a.Ptr<DType>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  int32_t* recv_list_nodes = recv_list_nodes_a.Ptr<int32_t>();
  int64_t* pos             = pos_a.Ptr<int64_t>();

  int64_t size = in_degs_a.GetSize() >> 2;
  int64_t fsize = feat_a.GetSize() >> 2;

  #pragma omp parallel for
  for (int64_t i=0; i<lim; i++) {
    DType *iptr = otf + i * (feat_size + 1);
    CHECK(iptr[0] >= 0) << "Error: ScatterReduceLR, -ve index!"; 
    // if(iptr[0] < 0) {
    //   printf("Error: ScatterReduceLR, at -ve index, i: %d, iptr: %f\n", i, iptr[0]);
    //   fflush(0);
    //   exit(EXIT_FAILURE);
    // }

    int32_t pid = -1;
    int64_t index = Map2localIndex(otn[i], node_map, num_parts, pid, cur_part);
    CHECK(pid == cur_part);
    CHECK(index < size && index >= 0) << "Error: ScatterReudce, index out of range!";
    // if (index >= size || index < 0) {
    //   printf("Error: ScatterReduceLR, at index out of range: %d\n", index);
    //   fflush(0);
    //   exit(EXIT_FAILURE);      
    // }
    
    in_degs[index] += iptr[0];
    recv_list_nodes[i] = index;
    DType *optr = feat + index * feat_size;

    CHECK((i+1)*(feat_size+1) <= dim) << "Error: ScatterReudce, feat_size out of range!";
    // if ((i+1)*(feat_size+1) > dim) {
    //   printf("Error: lim: %d %d\n", i+1+feat_size, dim);
    //   fflush(0);exit(EXIT_FAILURE);
    // }
    CHECK((index+1)*feat_size <= fsize) << "Error: ScatterReudce, feat_size (fsize) out of range!";
    // if( (index+1)*feat_size > fsize) {
    //   printf("Error: fsize, %d %d %d\n", (index+1)*feat_size, fsize, feat_size);
    //   fflush(0);exit(EXIT_FAILURE);
    // }
        
    #pragma omp simd
    for (int32_t j=0; j<feat_size; j++)
      optr[j] += iptr[1 + j];
  }
  pos[0] += lim;
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelScatterReduceLR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray otf             = args[0];
  int64_t offsetf         = args[1];
  NDArray otn             = args[2];
  int64_t offsetn         = args[3];    
  NDArray feat            = args[4];
  NDArray in_degs         = args[5];
  NDArray node_map        = args[6];
  int64_t dim             = args[7];
  int64_t feat_size       = args[8];
  int64_t num_parts       = args[9];
  NDArray recv_list_nodes = args[10];
  NDArray pos             = args[11];
  int64_t lim             = args[12];
  int64_t cur_part        = args[13];

  ATEN_FLOAT_TYPE_SWITCH(in_degs->dtype, IdType, "Id data", {
      ATEN_FLOAT_TYPE_SWITCH(otf->dtype, DType, "Feature data", {
          ScatterReduceLR<IdType, DType>(otf, offsetf, otn, offsetn,
                                         feat, in_degs, node_map,
                                         dim, feat_size, num_parts,
                                         recv_list_nodes, pos, lim,
                                         cur_part);
        });
    });
});


template<typename DType>
void FdrpaGatherEmbRL(
  NDArray feat_a,
  int64_t feat_shape,
  NDArray send_feat_list_a,
  int64_t offset,
  NDArray recv_list_nodes_a,
  int64_t lim,
  NDArray in_degs_a,
  int32_t feat_size,
  int32_t cur_part,
  NDArray node_map_a,
  int32_t num_parts) {
  
  DType*   feat            = feat_a.Ptr<DType>();
  DType*   send_feat_list  = send_feat_list_a.Ptr<DType>() + offset;
  int32_t* recv_list_nodes  = recv_list_nodes_a.Ptr<int32_t>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  DType*   in_degs         = in_degs_a.Ptr<DType>();
    
  int32_t pos = 0, pos_out = 0;

  #pragma omp parallel for
  for (int64_t i=0; i<lim; i++) {
    int64_t index = recv_list_nodes[i];
    CHECK(index < feat_shape) << "Error: FdrpaGartherRL, index of range of feat_shape!";
    //if (index >= feat_shape) {
    //  printf("Error:at index: %d, feat_shape: %d\n", index, feat_shape);
    //  fflush(0);
    //  exit(EXIT_FAILURE);
    //}

    DType *iptr = feat + index * feat_size;        
    DType *optr = send_feat_list + i * (feat_size + 1);
    optr[0] = in_degs[index];
        
    #pragma omp simd
    for (int32_t k=0; k<feat_size; k++)
      optr[1 + k] = iptr[k];
  }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaGatherEmbRL")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray feat              = args[0];
  int64_t feat_shape        = args[1];
  NDArray send_feat_list    = args[2];
  int64_t offset            = args[3];
  NDArray recv_list_nodes   = args[4];
  int64_t lim               = args[5];
  NDArray in_degs           = args[6];
  int32_t feat_size         = args[7];
  int32_t cur_part          = args[8];
  NDArray node_map          = args[9];
  int32_t num_parts         = args[10];

  ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Feature data", {
      FdrpaGatherEmbRL<DType>(feat,
                              feat_shape,
                              send_feat_list,
                              offset,
                              recv_list_nodes,
                              lim,
                              in_degs,
                              feat_size,
                              cur_part,
                              node_map,
                              num_parts);
    });    
});

template<typename DType>
void ScatterReduceRL(
  NDArray otf_a,
  int64_t offset,
  NDArray stn_a,
  int64_t lim,
  NDArray in_degs_a,
  NDArray feat_a,    
  NDArray node_map_a,
  int64_t dim,
  int64_t feat_size,
  int64_t num_parts,
  int32_t cur_part) {
  
  DType*   otf             = otf_a.Ptr<DType>() + offset;
  int32_t* stn             = stn_a.Ptr<int32_t>();
  DType*   feat            = feat_a.Ptr<DType>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  DType*   in_degs         = in_degs_a.Ptr<DType>();

  int64_t fsize = feat_a.GetSize() >> 2;

  #pragma omp parallel for
  for (int64_t i=0; i<lim; i++) {
    DType   *iptr  = otf + i * (feat_size + 1); 
    int64_t  index = stn[i];
    DType   *optr  = feat + index * feat_size;
    in_degs[index] = iptr[0];

    CHECK((i+1)*(1+feat_size) <= dim) << "Error: ScatterReduceRL, index out of range!";
    // if ((i+1)*(1+feat_size) > dim) {
    //   printf("Error: at lim: %d %d\n", i+1+feat_size, dim);
    //   fflush(0);exit(EXIT_FAILURE);
    // }
    CHECK((index+1)*feat_size <= fsize) << "Error: ScatterReduceRL, index (fsize) out of range!";
    // if( (index+1)*feat_size > fsize) {
    //   printf("Error: at fsize, %d %d %d\n", (index+1)*feat_size, fsize, feat_size);
    //   fflush(0);exit(EXIT_FAILURE);
    // }

    #pragma simd
    for (int32_t j=0; j<feat_size; j++)
      optr[j] = iptr[1+j];    
  }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelScatterReduceRL")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray otf             = args[0];
  int64_t offset          = args[1];
  NDArray stn             = args[2];
  int64_t lim             = args[3];
  NDArray in_degs         = args[4];
  NDArray feat            = args[5];
  NDArray node_map        = args[6];
  int64_t dim             = args[7];
  int64_t feat_size       = args[8];
  int64_t num_parts       = args[9];
  int64_t cur_part        = args[10];

  ATEN_FLOAT_TYPE_SWITCH(otf->dtype, DType, "Feature data", {
      ScatterReduceRL<DType>(otf, offset, stn, lim,
                             in_degs, feat, node_map,
                             dim, feat_size, num_parts,
                             cur_part);
    });
});

template<typename IdType>
void FdrpaCommBuckets(
  NDArray adj_a,
  NDArray selected_nodes_a,
  NDArray ver2part_a,
  NDArray ver2part_index_a,
  NDArray node_map_a,
  NDArray buckets_a,
  NDArray lf_a,
  int32_t width,
  int32_t num_parts,
  int32_t cur_part) {
  
  IdType*  adj             = adj_a.Ptr<IdType>();
  int32_t* selected_nodes  = selected_nodes_a.Ptr<int32_t>();
  int32_t* ver2part       = ver2part_a.Ptr<int32_t>();
  int32_t* ver2part_index = ver2part_index_a.Ptr<int32_t>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  int32_t* buckets         = buckets_a.Ptr<int32_t>();
  int32_t* lf              = lf_a.Ptr<int32_t>();
    
  int32_t nthreads = omp_get_max_threads();
  int32_t *gbucket = (int32_t*) calloc (sizeof(int32_t), nthreads * num_parts);
    
  int32_t tnodes = 0;
  int32_t len = selected_nodes_a.GetSize() >> 2;
    
  #pragma omp parallel
  {
    int32_t tid = omp_get_thread_num();
    std::vector<std::pair<int32_t, int32_t> > cache(num_parts);
    #pragma omp for
    for(int64_t i=0; i<len; i++) {
      int64_t index = selected_nodes[i];
      int32_t lf_val = lf[index];
      CHECK(lf_val != -200);
      int32_t p = Ver2partition<IdType>(lf_val, node_map, num_parts);
            
      IdType *ptr = adj + width * index;
      int32_t min_part = p;
      int32_t min_index = lf_val;
            
      if (min_part != cur_part) {
        // debug code
        bool flg = 1;
        for (int32_t j=0; j<width; j++) {
          if (ptr[j] >= 0) {
            if(ptr[j] == lf_val)
              flg = 0;
          }                
          else                         
            break;
        }                
        CHECK(flg == 0);
        // debug code ends
        gbucket[tid*num_parts + min_part] ++;
        ver2part[i] = min_part;
        ver2part_index[i] = min_index;
      }
      else {
        ver2part[i] = -100;
      }                
    }
  }
  // aggregate
  for (int32_t t=0; t<nthreads; t++) {
    for (int32_t p=0; p<num_parts; p++)
      buckets[p] += gbucket[t*num_parts + p];
  }
  for (int32_t p=0; p<num_parts; p++) tnodes += buckets[p];    

  free(gbucket);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaCommBuckets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray adj            = args[0];
  NDArray selected_nodes = args[1];
  NDArray ver2part       = args[2];
  NDArray ver2part_index = args[3];
  NDArray node_map       = args[4];
  NDArray buckets        = args[5];
  NDArray lf             = args[6];
  int32_t width          = args[7];
  int32_t num_parts      = args[8];
  int32_t cur_part       = args[9];

  ATEN_ID_TYPE_SWITCH(adj->dtype, IdType, {
      FdrpaCommBuckets<IdType>(adj,
                               selected_nodes,
                               ver2part,
                               ver2part_index,
                               node_map,
                               buckets,
                               lf,
                               width,
                               num_parts,
                               cur_part);
    });
});


template<typename IdType>
void FdrpaInitBuckets(
  NDArray adj_a,
  NDArray selected_nodes_a,
  NDArray node_map_a,
  NDArray buckets_a,
  NDArray lf_a,
  int32_t width,
  int32_t num_parts,
  int32_t cur_part) {
  
  IdType*  adj             = adj_a.Ptr<IdType>();
  int32_t* selected_nodes  = selected_nodes_a.Ptr<int32_t>();
  int32_t* node_map        = node_map_a.Ptr<int32_t>();
  int32_t* buckets         = buckets_a.Ptr<int32_t>();
  int32_t* lf              = lf_a.Ptr<int32_t>();
    
  int32_t nthreads = omp_get_max_threads();
  int32_t *gbucket = (int32_t*) calloc (sizeof(int32_t), nthreads * num_parts);    
  int32_t len = selected_nodes_a.GetSize() >> 2;
    
  #pragma omp parallel
  {
    int32_t tid = omp_get_thread_num();
    #pragma omp for
    for(int64_t i=0; i<len; i++) {
      int64_t index = selected_nodes[i];
      int32_t lf_val = lf[index];
      CHECK(lf_val != -200);
      int32_t p = Ver2partition<IdType>(lf_val, node_map, num_parts);
            
      IdType *ptr = adj + width * index;
      int32_t min_part = p;
      int32_t min_index = lf_val;
            
      if (min_part != cur_part) {
        // check-debug code
        bool flg = 1;
        for (int32_t j=0; j<width; j++) {
          if (ptr[j] >= 0) {
            if(ptr[j] == lf_val)
              flg = 0;
          }                
          else                         
            break;
        }                
        CHECK(flg == 0);
        // assert(flg == 0);
        // check-debug code ends
        gbucket[tid*num_parts + min_part] ++;
      }
    }
  }

  // aggregate
  for (int32_t t=0; t<nthreads; t++) {
    for (int32_t p=0; p<num_parts; p++)
      buckets[p] += gbucket[t*num_parts + p];
  }
  free(gbucket);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelFdrpaInitBuckets")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray adj            = args[0];
  NDArray selected_nodes = args[1];
  NDArray node_map       = args[2];
  NDArray buckets        = args[3];
  NDArray lf             = args[4];
  int32_t width          = args[5];
  int32_t num_parts      = args[6];
  int32_t cur_part       = args[7];

  ATEN_ID_TYPE_SWITCH(adj->dtype, IdType, {
      FdrpaInitBuckets<IdType>(adj,
                               selected_nodes,
                               node_map,
                               buckets,
                               lf,
                               width,
                               num_parts,
                               cur_part);
    });
});


template<typename DType>
void DegDiv(
  NDArray neigh_,    
  NDArray h_,        
  int64_t feat_size,
  NDArray in_degs_,  
  int64_t lim) {
  
  DType*  neigh = neigh_.Ptr<DType>();
  DType*  h     = h_.Ptr<DType>();
  DType* degs  = in_degs_.Ptr<DType>();

  #pragma omp parallel for
  for (int32_t i=0; i<lim; i++) {
    DType deg = degs[i];
    DType *ptrn = neigh + i* feat_size;
    DType *ptrh = h     + i* feat_size;
    for (int32_t j=0; j<feat_size; j++) {

      ptrn[j] = (ptrn[j] +  ptrh[j]) / (deg + 1);
    }

  }
}
    
DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelDegDiv")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray neigh_    = args[0];
  NDArray h_        = args[1];
  int64_t feat_size_= args[2];
  NDArray in_degs_  = args[3];
  int64_t lim_      = args[4];

  ATEN_FLOAT_TYPE_SWITCH(neigh_->dtype, DType, "Feature data", {
      DegDiv<DType>(neigh_, h_, feat_size_, in_degs_, lim_);
    });
});

template<typename DType>
void DegDivBack(
  NDArray neigh_grad_,    
  int64_t feat_size,
  NDArray in_degs_,  
  int64_t lim) {
  
  DType*  neigh_grad = neigh_grad_.Ptr<DType>();
  DType* degs  = in_degs_.Ptr<DType>();

  #pragma omp parallel for
  for (int32_t i=0; i<lim; i++) {
    DType deg = degs[i];
    DType *ptrn = neigh_grad + i* feat_size;
    for (int32_t j=0; j<feat_size; j++) {

      ptrn[j] = ptrn[j]  / (deg + 1);
    }
  }
}
    
DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelDegDivBack")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray neigh_    = args[0];
  int64_t feat_size_= args[1];
  NDArray in_degs_  = args[2];
  int64_t lim_      = args[3];

  ATEN_FLOAT_TYPE_SWITCH(neigh_->dtype, DType, "Feature data", {
      DegDivBack<DType>(neigh_, feat_size_, in_degs_, lim_);
    });
});




/**************************************************************************************/
/* Libra partitioning codebase */

/*! \brief Identifies the lead loaded partition/community for a given edge assignment.*/
int32_t LeastLoad(int64_t* community_edges, int32_t nc) {
  // initialize random seed
  srand (time(NULL));    
  std::vector<int> score, loc;
  int32_t min = 1e9;
  for (int32_t i=0; i<nc; i++) {
    if (community_edges[i] < min) {
      min = community_edges[i];
    }
  }
  for (int32_t i=0; i<nc; i++) {
    if (community_edges[i] == min) {
      loc.push_back(i);
    }
  }
    
  int32_t r = rand() % loc.size();
  CHECK(loc[r] < nc);
  return loc[r];    
}

/*! \brief Libra - vertexcut based graph partitioning.
  \param nc Number of partitions/communities
  \param node_degree_a Per node degree
  \param edgenum_unassigned_a intermediate memmory as tensor
  \param community_weights_a weight of the created partitions
  \param u_a src nodes
  \param v_a dst nodes
  \param w_a weight per edge
  \param out_a partition assignment of the edges
  \param N_n number of nodes in the input graph
  \param N_e number of edges in the input graph
  \param prefix output/partition storage location 
*/
template<typename IdType, typename IdType2, typename DType>
int32_t LibraVertexCut(
  int32_t nc,
  NDArray node_degree_a,
  NDArray edgenum_unassigned_a,
  NDArray community_weights_a,
  NDArray u_a,
  NDArray v_a,
  NDArray w_a,
  NDArray out_a,
  int64_t N_n,
  int64_t N_e,
  std::string prefix) {
  
  int32_t *out                = out_a.Ptr<int32_t>();
  IdType  *node_degree        = node_degree_a.Ptr<IdType>();
  IdType  *edgenum_unassigned = edgenum_unassigned_a.Ptr<IdType>();
  IdType2 *uptr               = u_a.Ptr<IdType2>();
  IdType2 *vptr               = v_a.Ptr<IdType2>();
  float   *wptr               = w_a.Ptr<float>();
  float   *community_weights  = community_weights_a.Ptr<float>();

  std::vector<std::vector<int32_t> > node_assignments(N_n);
  std::vector<IdType2> replication_list;
    
  // local allocations
  int64_t *community_edges = (int64_t*)calloc(sizeof(int64_t), nc);
  int64_t *cache = (int64_t*)calloc(sizeof(int64_t), nc);
        
  CHECK ((out_a.GetSize() >> 2) == N_e);
  CHECK ((node_degree_a.GetSize() >> 3) == N_n);
  CHECK ((u_a.GetSize() >> 3) == N_e);
  CHECK ((w_a.GetSize() >> 2) == N_e);
    
  for (int64_t i=0; i<N_e; i++) {
    IdType u = uptr[i];
    IdType v = vptr[i];
    float  w = wptr[i];
    CHECK (u < N_n);
    CHECK (v < N_n);
        
    if (i%10000000 == 0) {
      printf("."); fflush(0);
    }

    if (node_assignments[u].size() == 0 && node_assignments[v].size() == 0) {
      int32_t c = LeastLoad(community_edges, nc);
      out[i] = c;
      CHECK_LT(c, nc);
        
      community_edges[c] ++;
      community_weights[c] = community_weights[c] + w;
      node_assignments[u].push_back(c);
      if (u != v)
        node_assignments[v].push_back(c);

      CHECK(node_assignments[u].size() <= nc) <<
        "Error: 1. generated splits (u) are greater than nc!";
      CHECK(node_assignments[v].size() <= nc) <<
        "Error: 1. generated splits (v) are greater than nc!";
      edgenum_unassigned[u] --;
      edgenum_unassigned[v] --;
    }
    else if (node_assignments[u].size() != 0 && node_assignments[v].size() == 0) {
      for (uint32_t j=0; j<node_assignments[u].size(); j++) {
        int32_t cind = node_assignments[u][j];
        cache[j] = community_edges[cind];
      }
      int32_t cindex = LeastLoad(cache, node_assignments[u].size());
      int32_t c = node_assignments[u][cindex];
      out[i] = c;
      community_edges[c] ++;
      community_weights[c] = community_weights[c] + w;
            
      node_assignments[v].push_back(c);
      CHECK(node_assignments[v].size() <= nc) <<
        "Error: 2. generated splits (v) are greater than nc!";
      edgenum_unassigned[u] --;
      edgenum_unassigned[v] --;
    }
    else if (node_assignments[v].size() != 0 && node_assignments[u].size() == 0) {
      for (uint32_t j=0; j<node_assignments[v].size(); j++) {
        int32_t cind = node_assignments[v][j];
        cache[j] = community_edges[cind];
      }
      int32_t cindex = LeastLoad(cache, node_assignments[v].size());
      int32_t c = node_assignments[v][cindex];
      CHECK(c < nc) << "Error: 2. partition greater than nc !!";
      out[i] = c;

      community_edges[c] ++;
      community_weights[c] = community_weights[c] + w;
            
      node_assignments[u].push_back(c);
      CHECK(node_assignments[u].size() <= nc) <<
        "3. Error: generated splits (u) are greater than nc!";
      edgenum_unassigned[u] --;
      edgenum_unassigned[v] --;
    }
    else {
      std::vector<int> setv(nc), intersetv;
      for (int32_t j=0; j<nc; j++) setv[j] = 0;
      int32_t interset = 0;

      CHECK(node_assignments[u].size() <= nc) <<
        "4. Error: generated splits (u) are greater than nc!";
      CHECK(node_assignments[v].size() <= nc) <<
        "4. Error: generated splits (v) are greater than nc!";
      for (int32_t j=0; j<node_assignments[v].size(); j++) {
        CHECK(node_assignments[v][j] < nc) << "Error: 4. Part assigned (v) greater than nc!";
        setv[node_assignments[v][j]] ++;
      }
      
      for (int32_t j=0; j<node_assignments[u].size(); j++) {
        CHECK(node_assignments[u][j] < nc) << "Error: 4. Part assigned (u) greater than nc!";
        setv[node_assignments[u][j]] ++;
      }
            
      for (int32_t j=0; j<nc; j++) {        
        CHECK(setv[j] <= 2) << "Error: 4. unexpected computed value !!!";
        if (setv[j] == 2) {
          interset++;
          intersetv.push_back(j);
        }
      }
      if (interset) {
        for (int32_t j=0; j<intersetv.size(); j++) {
          int32_t cind = intersetv[j];
          cache[j] = community_edges[cind];                    
        }
        int32_t cindex = LeastLoad(cache, intersetv.size());
        int32_t c = intersetv[cindex];
        
        CHECK(c < nc) << "Error: 4. partition greater than nc !!";
        out[i] = c;
        community_edges[c] ++;
        community_weights[c] = community_weights[c] + w;
        edgenum_unassigned[u] --;
        edgenum_unassigned[v] --;                                
      }
      else {
        if (node_degree[u] < node_degree[v]) {
          for (uint32_t j=0; j<node_assignments[u].size(); j++) {
            int32_t cind = node_assignments[u][j];
            cache[j] = community_edges[cind];
          }
          int32_t cindex = LeastLoad(cache, node_assignments[u].size());
          int32_t c = node_assignments[u][cindex];
          CHECK(c < nc) << "Error: 5. partition greater than nc !!";
          out[i] = c;
          community_edges[c] ++;
          community_weights[c] = community_weights[c] + w;

          for (uint32_t j=0; j<node_assignments[v].size(); j++) {
            CHECK(node_assignments[v][j] != c) <<
              "Error: 5. duplicate partition (v) assignment !!";
          }

          node_assignments[v].push_back(c);
          CHECK(node_assignments[v].size() <= nc) <<
            "Error: 5. generated splits (v) greater than nc!!";
          replication_list.push_back(v);
          edgenum_unassigned[u] --;
          edgenum_unassigned[v] --;
        }
        else {
          for (uint32_t j=0; j<node_assignments[v].size(); j++) {
            int32_t cind = node_assignments[v][j];
            cache[j] = community_edges[cind];
          }
          int32_t cindex = LeastLoad(cache, node_assignments[v].size());
          int32_t c = node_assignments[v][cindex];
          CHECK(c < nc)  << "Error: 6. partition greater than nc !!";
          out[i] = c;
          community_edges[c] ++;
          community_weights[c] = community_weights[c] + w;
          for (uint32_t j=0; j<node_assignments[u].size(); j++) {
            CHECK(node_assignments[u][j] != c) <<
              "Error: 6. duplicate partition (u) assignment !!";
          }
          if (u != v)
            node_assignments[u].push_back(c);

          CHECK(node_assignments[u].size() <= nc) <<
            "Error: 6. generated splits (u) greater than nc!!";
          replication_list.push_back(u);
          edgenum_unassigned[u] --;
          edgenum_unassigned[v] --;                                                        
        }
      }
    }        
  }
  free(cache);

  std::string lprefix = prefix;
  for (int64_t c=0; c < nc; c++) {
    std::string str = lprefix + "/" + std::to_string(nc) + "Communities/community" +
      std::to_string(c) +".txt";
        
    FILE *fp = fopen(str.c_str(), "w");
    CHECK_NOTNULL(fp);
        
    for (int64_t i=0; i<N_e; i++) {
      if (out[i] == c) 
        fprintf(fp, "%ld,%ld,%f\n", uptr[i], vptr[i], wptr[i]);
    }
    fclose(fp);
  }

  std::string str = lprefix + "/"  + std::to_string(nc) + "Communities/replicationlist.csv";
  FILE *fp = fopen(str.c_str(), "w");
  CHECK_NOTNULL(fp);
    
  std::string str_ = "# The Indices of Nodes that are replicated :: Header";
  fprintf(fp, "## The Indices of Nodes that are replicated :: Header");
  printf("Total replication: %ld\n", replication_list.size());

  for (uint64_t i=0; i<replication_list.size(); i++)
    fprintf(fp, "%ld\n", replication_list[i]);

  printf("Community weights:\n");
  for (int64_t c=0; c < nc; c++)
    printf("%f ", community_weights[c]);
  printf("\n");
    
  printf("Community edges:\n");
  for (int64_t c=0; c < nc; c++)
    printf("%ld ", community_edges[c]);
  printf("\n");

  free(community_edges);    
  fclose(fp);
    
  return 0;
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibraVertexCut")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int32_t nc                 = args[0];
  NDArray node_degree        = args[1];
  NDArray edgenum_unassigned = args[2];
  NDArray community_weights  = args[3];
  NDArray u                  = args[4];
  NDArray v                  = args[5];
  NDArray w                  = args[6];
  NDArray out                = args[7];
  int64_t N                  = args[8];
  int64_t N_e                = args[9];
  std::string prefix        = args[10];
  
  ATEN_ID_TYPE_SWITCH(node_degree->dtype, IdType2, {
      ATEN_ID_TYPE_SWITCH(u->dtype, IdType, {
          ATEN_FLOAT_TYPE_SWITCH(w->dtype, DType, "Feature data", {
              LibraVertexCut<IdType, IdType2, DType>(nc,
                                                     node_degree,
                                                     edgenum_unassigned,
                                                     community_weights,
                                                     u,
                                                     v,
                                                     w,
                                                     out,
                                                     N,
                                                     N_e,
                                                     prefix);
            });
        });
    });
});


/*! \brief Builds dictionaries for assigning local node IDs to nodes in the paritions
  and prepares indices to gather the graph from input graph.
  \param a_a local src node ID of an edge in a partition
  \param b_a local dst node ID of an edge in a partition
  \param indices_a keep track of global node ID to local node ID
  \param ldt_key_a per partition dict for tacking gloal to local node IDs
  \param gdt_key_a global dict for assigning consecutive node IDs to nodes across all the
  partitions
  \param gdt_value_a global dict for assigning consecutive node IDs to nodes across all the
  partition
  \param node_map_a keeps track of range of local node IDs (consecutive) given to the nodes in
  the partitions
  \param nc  number of partitions/communities
  \param c current partition numbers
  \param fsize size of pre-allocated memory tensor
  \param hash_nodes_a returns the actual number of nodes and edges in the partition
  \param prefix output file location
 */
void Libra2dglBuildDict(
  NDArray a_a,
  NDArray b_a,
  NDArray indices_a,
  NDArray ldt_key_a,
  NDArray gdt_key_a,
  NDArray gdt_value_a,
  NDArray node_map_a,
  NDArray offset_a,
  int32_t nc,
  int32_t c,
  int64_t fsize,
  NDArray hash_nodes_a,
  std::string prefix) {

  int32_t *indices    = indices_a.Ptr<int32_t>();
  int64_t *ldt_key    = ldt_key_a.Ptr<int64_t>();
  int32_t *gdt_key    = gdt_key_a.Ptr<int32_t>();
  int32_t *gdt_value  = gdt_value_a.Ptr<int32_t>(); // 2D tensor
  int32_t *node_map   = node_map_a.Ptr<int32_t>();
  int32_t *offset     = offset_a.Ptr<int32_t>();
  int32_t *hash_nodes = hash_nodes_a.Ptr<int32_t>();    
  int32_t width = nc;
        
  int64_t *a = a_a.Ptr<int64_t>();
  int64_t *b = b_a.Ptr<int64_t>();

  int32_t N_n = indices_a.GetSize() >> 2;
  int32_t num_nodes = ldt_key_a.GetSize() >> 2;

  #pragma omp simd
  for (int32_t i=0; i<N_n; i++) {
    indices[i] = -100;
  }
    
  int32_t pos = 0;
  int64_t edge = 0;
  std::string lprefix = prefix;
  std::string str = lprefix + "/community" + std::to_string(c) + ".txt";    
  FILE *fp = fopen(str.c_str(), "r");
  CHECK_NOTNULL(fp);

  while(!feof(fp) && edge < fsize) {
    int64_t u, v;
    float w;
    fscanf(fp, "%ld,%ld,%f\n", &u, &v, &w);
            
    if (indices[u] == -100) {
      ldt_key[pos] = u;
      CHECK(pos < num_nodes);
      indices[u] = pos++;
    }
    if (indices[v] == -100) {
      ldt_key[pos] = v;
      CHECK(pos < num_nodes);
      indices[v] = pos++;
    }
    a[edge] = indices[u];
    b[edge++] = indices[v];
  }
  CHECK(edge <= fsize);
  fclose(fp);
  hash_nodes[0] = pos;
  hash_nodes[1] = edge;

  for (int32_t i=0; i<pos; i++) {
    int64_t  u   = ldt_key[i];
    int32_t  v   = indices[u];            
    int32_t *ind = &gdt_key[u];
    int32_t *ptr = gdt_value + u*width;
    ptr[*ind] = offset[0] + v;
    (*ind)++;
    CHECK(v != -100);
    CHECK(*ind <= nc);
  }
  node_map[c] = offset[0] +  pos;
  offset[0] += pos;
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglBuildDict")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray     a          = args[0];
  NDArray     b          = args[1];
  NDArray     indices    = args[2];
  NDArray     ldt_key    = args[3];
  NDArray     gdt_key    = args[4];
  NDArray     gdt_value  = args[5];
  NDArray     node_map   = args[6];
  NDArray     offset     = args[7];
  int32_t     nc         = args[8];
  int32_t     c          = args[9];
  int64_t     fsize      = args[10];
  NDArray     hash_nodes = args[11];
  std::string prefix     = args[12];
  
  Libra2dglBuildDict(a, b, indices, ldt_key, gdt_key,
                     gdt_value, node_map, offset,
                     nc, c, fsize, hash_nodes, prefix);
});


/*! \brief sets up the 1-level tree aomng the clones of the split-nodes.
  \param gdt_key_a global dict for assigning consecutive node IDs to nodes across all the
  partitions
  \param gdt_value_a global dict for assigning consecutive node IDs to nodes across all the
  partition
  \param lftensor_a keeps the root node ID of 1-level tree
  \param nc number of partitions/communities
  \param Nn number of nodes in the input graph
 */
void Libra2dglSetLF(
  NDArray gdt_key_a,
  NDArray gdt_value_a,
  NDArray lftensor_a,
  int32_t nc,
  int64_t Nn) {
    
  srand (time(NULL));
  int32_t *gdt_key    = gdt_key_a.Ptr<int32_t>();
  int32_t *gdt_value  = gdt_value_a.Ptr<int32_t>(); // 2D tensor
  int32_t *lftensor   = lftensor_a.Ptr<int32_t>();
    
  int32_t width = nc;
  int32_t cnt = 0;
  int32_t avg_split_copy = 0, scnt = 0;
  
  for (int64_t i=0; i<Nn; i++) {        
    if (gdt_key[i] <= 0) {
      cnt ++;
    }
    else {
      int32_t val = rand() % gdt_key[i];
      CHECK(val >= 0 && val < gdt_key[i]);
      CHECK(gdt_key[i] <= nc);
      
      int32_t *ptr = gdt_value + i*width;
      lftensor[i] = ptr[val];
    }
    if (gdt_key[i] > 1) {
      avg_split_copy += gdt_key[i];
      scnt ++;
    }
  }
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglSetLF")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray gdt_key   = args[0];
  NDArray gdt_value = args[1];
  NDArray lftensor  = args[2];
  int32_t nc        = args[3];
  int64_t Nn        = args[4];

  Libra2dglSetLF(gdt_key, gdt_value, lftensor, nc, Nn);
});


/*!
  \brief For each node in a partition, it creates a list of remote clone IDs;
  also, for each node in a partition, it gathers the data (feats, label, trian, test)
  from input graph.
  \param feat_a node features in current partition c
  \param gfeat_a input graph node features
  \param adj_a list of node IDs of remote clones
  \param inner_nodes_a marks whether a node is split or not
  \param ldt_key_a per partition dict for tacking gloal to local node IDs
  \param gdt_key_a global dict for assigning consecutive node IDs to nodes across all the
  partitions
  \param gdt_value_a global dict for assigning consecutive node IDs to nodes across all the
  partition
  \param node_map_a keeps track of range of local node IDs (consecutive) given to the nodes in
  the partitions
  \param lf_a 1-level tree marking for local split nodes
  \param lftensor_a gloabl (all the partitions) 1-level tree 
  \param num_nodes number of nodes in current partition
  \param nc number of partitions/communities
  \param c current partition/community
  \param feat_size node feature vector size
  \param labels_a local (for this partition) labels
  \param trainm_a local (for this partition) training nodes
  \param testm_a local (for this partition) testing nodes
  \param valm_a local (for this partition) validation nodes
  \param glabels_a gloabl (input graph) labels
  \param gtrainm_a gloabl (input graph) training nodes
  \param gtestm_a gloabl (input graph) testing nodes
  \param gvalm_a gloabl (input graph) validation nodes
  \param Nn number of nodes in the input graph
 */
template<typename IdType,typename DType>
void Libra2dglBuildAdjlist(
  NDArray feat_a,
  NDArray gfeat_a,
  NDArray adj_a,
  NDArray inner_node_a,
  NDArray ldt_key_a,
  NDArray gdt_key_a,
  NDArray gdt_value_a,
  NDArray node_map_a,
  NDArray lf_a,
  NDArray lftensor_a,
  int32_t num_nodes,
  int32_t nc,
  int32_t c,
  int32_t feat_size,
  NDArray labels_a , 
  NDArray trainm_a ,
  NDArray testm_a  ,
  NDArray valm_a   ,
  NDArray glabels_a,
  NDArray gtrainm_a,
  NDArray gtestm_a ,
  NDArray gvalm_a,
  int64_t Nn) {
  
  DType   *feat       = feat_a.Ptr<DType>(); // 2D tensor
  DType   *gfeat      = gfeat_a.Ptr<DType>(); // 2D tensor
  int32_t *adj        = adj_a.Ptr<int32_t>(); // 2D tensor
  int32_t *inner_node = inner_node_a.Ptr<int32_t>();
  int64_t *ldt_key    = ldt_key_a.Ptr<int64_t>();
  int32_t *gdt_key    = gdt_key_a.Ptr<int32_t>();
  int32_t *gdt_value  = gdt_value_a.Ptr<int32_t>(); // 2D tensor
  int32_t *node_map   = node_map_a.Ptr<int32_t>();
  int32_t *lf         = lf_a.Ptr<int32_t>();
  int32_t *lftensor   = lftensor_a.Ptr<int32_t>();
  int32_t width = nc - 1;

  #pragma omp parallel for
  for (int32_t i=0; i<num_nodes; i++) {
    int64_t k = ldt_key[i];
    int64_t v = i;
    int32_t ind = gdt_key[k];
        
    int32_t *adj_ptr = adj + v*width;
    if(ind == 1) {
      for (int32_t j=0; j<width; j++) adj_ptr[j] = -1;
      inner_node[i] = 1;
      lf[i] = -200;
    }
    else {
      lf[i] = lftensor[k];
      int32_t *ptr = gdt_value + k*nc;
      int32_t pos = 0;
      CHECK(ind <= nc);
      int32_t flg = 0;
      for (int32_t j=0; j<ind; j++) {
        if (ptr[j] == lf[i]) flg = 1;
        if(c != Ver2partition<int32_t>(ptr[j], node_map, nc) )
          adj_ptr[pos++] = ptr[j];
      }
      CHECK(flg == 1);
      CHECK(pos == ind - 1);
      for (; pos < width; pos++) adj_ptr[pos] = -1;
      inner_node[i] = 0;
    }        
  }    
  // gathering 
  #pragma omp parallel for
  for (int64_t i=0; i<num_nodes; i++) {
    int64_t k = ldt_key[i];
    int64_t ind = i*feat_size;
    DType *optr = gfeat + ind;
    DType *iptr = feat + k*feat_size;        

    for (int32_t j=0; j<feat_size; j++)
      optr[j] = iptr[j];
  }

  IdType *labels = labels_a.Ptr<IdType>();
  IdType *glabels = glabels_a.Ptr<IdType>();
  bool *trainm = trainm_a.Ptr<bool>();
  bool *gtrainm = gtrainm_a.Ptr<bool>();
  bool *testm = testm_a.Ptr<bool>();
  bool *gtestm = gtestm_a.Ptr<bool>();
  bool *valm = valm_a.Ptr<bool>();
  bool *gvalm = gvalm_a.Ptr<bool>();  
  #pragma omp parallel for
  for (int64_t i=0; i<num_nodes; i++) {
    int64_t k = ldt_key[i];
    CHECK(k >=0 && k < Nn);
    glabels[i] = labels[k];
    gtrainm[i] = trainm[k];
    gtestm[i] = testm[k];
    gvalm[i] = valm[k];        
  }    
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra2dglBuildAdjlist")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NDArray feat       = args[0];
  NDArray gfeat      = args[1];
  NDArray adj        = args[2];
  NDArray inner_node = args[3];
  NDArray ldt_key    = args[4];
  NDArray gdt_key    = args[5];
  NDArray gdt_value  = args[6];
  NDArray node_map   = args[7];
  NDArray lf         = args[8];
  NDArray lftensor   = args[9];
  int32_t num_nodes  = args[10];
  int32_t nc         = args[11];
  int32_t c          = args[12];
  int32_t feat_size  = args[13];
  NDArray labels     = args[14];
  NDArray trainm     = args[15];
  NDArray testm      = args[16];
  NDArray valm       = args[17];
  NDArray glabels    = args[18];
  NDArray gtrainm    = args[19];
  NDArray gtestm     = args[20];
  NDArray gvalm      = args[21];
  int64_t Nn         = args[22];

  ATEN_FLOAT_TYPE_SWITCH(feat->dtype, DType, "Features", {
      ATEN_ID_BITS_SWITCH((glabels->dtype).bits, IdType, {
          Libra2dglBuildAdjlist<IdType, DType>(feat, gfeat,
                                               adj, inner_node,
                                               ldt_key, gdt_key,
                                               gdt_value,
                                               node_map, lf,
                                               lftensor, num_nodes,
                                               nc, c,
                                               feat_size, labels,
                                               trainm, testm,
                                               valm, glabels,
                                               gtrainm, gtestm,
                                               gvalm, Nn);
        });
    });
});


}  // namespace aten
}  // namespace dgl
