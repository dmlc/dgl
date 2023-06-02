/**
 *  Copyright (c) 2020 by Contributors
 * @file array/kernel.cc
 * @brief New kernels
 */
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"
#include "./check.h"
#include "kernel_decl.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace {}  // namespace

/** @brief Generalized Sparse Matrix-Matrix Multiplication. */
void SpMM(
    const std::string& op, const std::string& reduce, HeteroGraphPtr graph,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux) {
  // TODO(zihao): format tuning
  SparseFormat format = graph->SelectFormat(0, CSC_CODE);
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(out->dtype, Dtype, XPU, "Feature data", {
        if (format == SparseFormat::kCSC) {
          SpMMCsr<XPU, IdType, Dtype>(
              op, reduce, bcast, graph->GetCSCMatrix(0), ufeat, efeat, out,
              out_aux);
        } else if (format == SparseFormat::kCOO) {
          SpMMCoo<XPU, IdType, Dtype>(
              op, reduce, bcast, graph->GetCOOMatrix(0), ufeat, efeat, out,
              out_aux);
        } else {
          LOG(FATAL) << "SpMM only supports CSC and COO formats";
        }
      });
    });
  });
}

/** @brief Generalized segmented dense Matrix-Matrix Multiplication. */
void SegmentMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool A_trans, bool B_trans) {
  CHECK_EQ(A->ndim, 2) << "segment_mm expects a 2D tensor for the first input.";
  CHECK_EQ(B->ndim, 3)
      << "segment_mm expects a 3D tensor for the second input.";
  CHECK(!A_trans);
  if (B_trans) {
    CHECK_EQ(A->shape[1], B->shape[2])
        << "segment_mm expects A.shape[1] == B.shape[2] when B_trans=True";
  } else {
    CHECK_EQ(A->shape[1], B->shape[1])
        << "segment_mm expects A.shape[1] == B.shape[1]";
  }
  CHECK_EQ(B->shape[0], seglen_A.NumElements())
      << "segment_mm expects len(seglen_A) == B.shape[0]";
  CHECK_EQ(seglen_A->ctx.device_type, kDGLCPU)
      << "segment_mm expects seglen_A to be on CPU.";
  CHECK(A->ctx == B->ctx)
      << "segment_mm expects A and B to be of the same device";
  ATEN_XPU_SWITCH_CUDA(A->ctx.device_type, XPU, "SegmentMM", {
    ATEN_ID_TYPE_SWITCH(seglen_A->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(A->dtype, Dtype, XPU, "Feature data", {
        SegmentMM<XPU, IdType, Dtype>(A, B, C, seglen_A, A_trans, B_trans);
      });
    });
  });
}

void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  CHECK_EQ(A->ndim, 2) << "segment_mm_backward operator expects a 2D tensor "
                          "for the first input.";
  CHECK_EQ(dC->ndim, 2) << "segment_mm_backward operator expects a 2D tensor "
                           "for the second input.";
  CHECK_EQ(seglen->ctx.device_type, kDGLCPU)
      << "segment_mm expects seglen to be on CPU.";
  ATEN_XPU_SWITCH_CUDA(A->ctx.device_type, XPU, "SegmentMMBackwardB", {
    ATEN_ID_TYPE_SWITCH(seglen->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(A->dtype, Dtype, XPU, "Feature data", {
        SegmentMMBackwardB<XPU, IdType, Dtype>(A, dC, dB, seglen);
      });
    });
  });
}

/** @brief Generalized Dense Matrix-Matrix Multiplication according to relation
 * types. */
void GatherMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b) {
  CHECK_EQ(A->ndim, 2)
      << "gather_mm operator expects a 2D tensor for the first input.";
  CHECK_EQ(B->ndim, 3)
      << "gather_mm operator expects a 3D tensor for the second input.";
  CHECK(A->ctx == B->ctx)
      << "gather_mm expects all arguments to be on the same device.";
  if (aten::IsNullArray(idx_a)) {
    CHECK_EQ(A->shape[0], idx_b->shape[0])
        << "gather_mm expects len(idx_b) == A.shape[0] when idx_a is None.";
    CHECK(A->ctx == idx_b->ctx)
        << "gather_mm expects all arguments to be on the same device.";
  } else if (aten::IsNullArray(idx_b)) {
    CHECK_EQ(B->shape[0], idx_a->shape[0])
        << "gather_mm expects len(idx_a) == B.shape[0] when idx_b is None.";
    CHECK(A->ctx == idx_a->ctx)
        << "gather_mm expects all arguments to be on the same device.";
  } else {
    CHECK_EQ(idx_a->shape[0], idx_b->shape[0])
        << "gather_mm expects len(idx_a) == len(idx_b) when both idx_a and "
           "idx_b are given.";
    CHECK(A->ctx == idx_a->ctx && A->ctx == idx_b->ctx)
        << "gather_mm expects all arguments to be on the same device.";
  }
  const auto idtype = aten::IsNullArray(idx_a) ? idx_b->dtype : idx_a->dtype;
  ATEN_XPU_SWITCH_CUDA(A->ctx.device_type, XPU, "GatherMM", {
    ATEN_ID_TYPE_SWITCH(idtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(A->dtype, Dtype, XPU, "Feature data", {
        GatherMM<XPU, IdType, Dtype>(A, B, C, idx_a, idx_b);
      });
    });
  });
}

/** @brief Generalized Dense Matrix-Matrix Multiplication according to relation
 * types. */
void GatherMMScatter(
    const NDArray A, const NDArray B, NDArray C, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c) {
  CHECK_EQ(A->ndim, 2)
      << "gather_mm_scatter expects a 2D tensor for the first input.";
  CHECK(A->ctx == B->ctx)
      << "gather_mm_scatter expects all arguments to be on the same device.";
  if (!aten::IsNullArray(idx_c))
    CHECK(A->ctx == idx_c->ctx)
        << "gather_mm_scatter expects all arguments to be on the same device.";
  if (aten::IsNullArray(idx_a) && !aten::IsNullArray(idx_b)) {
    CHECK_EQ(A->shape[0], idx_b->shape[0])
        << "gather_mm_scatter expects len(idx_b) == A.shape[0] when idx_a is "
           "None.";
    CHECK(A->ctx == idx_b->ctx)
        << "gather_mm_scatter expects all arguments to be on the same device.";
  } else if (aten::IsNullArray(idx_b) && !aten::IsNullArray(idx_a)) {
    CHECK_EQ(B->shape[0], idx_a->shape[0])
        << "gather_mm_scatter expects len(idx_a) == B.shape[0] when idx_b is "
           "None.";
    CHECK(A->ctx == idx_a->ctx)
        << "gather_mm_scatter expects all arguments to be on the same device.";
  } else if (!aten::IsNullArray(idx_b) && !aten::IsNullArray(idx_a)) {
    CHECK_EQ(idx_a->shape[0], idx_b->shape[0])
        << "gather_mm_scatter expects len(idx_a) == len(idx_b) "
        << "when both idx_a and idx_b are given.";
    CHECK(A->ctx == idx_a->ctx && A->ctx == idx_b->ctx)
        << "gather_mm_scatter expects all arguments to be on the same device.";
  }
  ATEN_XPU_SWITCH_CUDA(A->ctx.device_type, XPU, "GatherMM", {
    ATEN_ID_TYPE_SWITCH(idx_c->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(A->dtype, Dtype, XPU, "Feature data", {
        GatherMMScatter<XPU, IdType, Dtype>(A, B, C, idx_a, idx_b, idx_c);
      });
    });
  });
}

/** @brief Generalized Sparse Matrix-Matrix Multiplication with hetero-graph
 * support. */
void SpMMHetero(
    const std::string& op, const std::string& reduce, HeteroGraphPtr graph,
    const std::vector<NDArray>& ufeat_vec,
    const std::vector<NDArray>& efeat_vec, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux) {
  SparseFormat format = graph->SelectFormat(0, CSC_CODE);

  std::vector<CSRMatrix> vec_graph;
  std::vector<dgl_type_t> ufeat_eid;
  std::vector<dgl_type_t> efeat_eid;
  std::vector<dgl_type_t> out_eid;
  auto pair = graph->meta_graph()->FindEdge(0);  // first etype
  NDArray ufeat_etype0 =
      (ufeat_vec.size() == 0) ? NullArray() : ufeat_vec[pair.first];
  NDArray efeat_etype0 = (efeat_vec.size() == 0) ? NullArray() : efeat_vec[0];
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    vec_graph.push_back(graph->GetCSCMatrix(etype));
    auto pair = graph->meta_graph()->FindEdge(etype);
    ufeat_eid.push_back(pair.first);
    efeat_eid.push_back(etype);
    out_eid.push_back(pair.second);
    if (ufeat_etype0->shape[1] != ufeat_vec[pair.first]->shape[1])
      LOG(FATAL) << "Column width of the input node features of all etypes "
                    "must be same.";
    if (efeat_etype0->shape[1] != efeat_vec[etype]->shape[1])
      LOG(FATAL) << "Column width of the input edge features of all etypes "
                    "must be same.";
  }
  const auto& bcast = CalcBcastOff(op, ufeat_etype0, efeat_etype0);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SpMM", {
    ATEN_ID_TYPE_SWITCH(
        graph->DataType(), IdType, {
          ATEN_FLOAT_TYPE_SWITCH_16BITS(
              (*out)[out_eid[0]]->dtype, Dtype, XPU, "Feature data", {
                if (format == SparseFormat::kCSC) {
                  SpMMCsrHetero<XPU, IdType, Dtype>(
                      op, reduce, bcast, vec_graph, ufeat_vec, efeat_vec, out,
                      out_aux, ufeat_eid, out_eid);
                } else {
                  // TODO(Israt): Add support for COO format
                  LOG(FATAL)
                      << "SpMM only supports CSC format for graphs with number "
                      << "of relation types > 1";
                }
              });
        });
  });
}

/** @brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void SDDMM(
    const std::string& op, HeteroGraphPtr graph, NDArray lhs, NDArray rhs,
    NDArray out, int lhs_target, int rhs_target) {
  // TODO(zihao): format tuning
  SparseFormat format = graph->SelectFormat(0, COO_CODE);
  const auto& bcast = CalcBcastOff(op, lhs, rhs);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(out->dtype, Dtype, XPU, "Feature data", {
        if (format == SparseFormat::kCSR) {
          SDDMMCsr<XPU, IdType, Dtype>(
              op, bcast, graph->GetCSRMatrix(0), lhs, rhs, out, lhs_target,
              rhs_target);
        } else if (format == SparseFormat::kCOO) {
          SDDMMCoo<XPU, IdType, Dtype>(
              op, bcast, graph->GetCOOMatrix(0), lhs, rhs, out, lhs_target,
              rhs_target);
        } else {
          LOG(FATAL) << "SDDMM only supports CSR and COO formats";
        }
      });
    });
  });
}

/**
 * @brief Find the src/dst/etype id based on the target 'u', 'v' or 'e'.
 *
 * @param graph The input graph.
 * @param target 'u', 'v' or 'e'. The target of the lhs or rhs data of an etype.
 * @param etype Relation type of the input graph.
 */
int get_typeid_by_target(HeteroGraphPtr graph, int target, dgl_type_t etype) {
  auto pair = graph->meta_graph()->FindEdge(etype);
  if (target == 0) return pair.first;
  if (target == 2) return pair.second;
  return etype;
}

/** @brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void SDDMMHetero(
    const std::string& op, HeteroGraphPtr graph, std::vector<NDArray> lhs,
    std::vector<NDArray> rhs, std::vector<NDArray> out, int lhs_target,
    int rhs_target) {
  SparseFormat format = graph->SelectFormat(0, COO_CODE);

  std::vector<dgl_type_t> lhs_eid;
  std::vector<dgl_type_t> rhs_eid;
  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    lhs_eid.push_back(get_typeid_by_target(graph, lhs_target, etype));
    rhs_eid.push_back(get_typeid_by_target(graph, rhs_target, etype));
  }
  const auto& bcast = CalcBcastOff(op, lhs[lhs_eid[0]], rhs[rhs_eid[0]]);

  ATEN_XPU_SWITCH_CUDA(graph->Context().device_type, XPU, "SDDMM", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(
          out[rhs_eid[0]]->dtype, Dtype, XPU, "Feature data", {
            if (format == SparseFormat::kCSR) {
              std::vector<CSRMatrix> vec_csr;
              for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes();
                   ++etype) {
                vec_csr.push_back(graph->GetCSRMatrix(etype));
              }
              SDDMMCsrHetero<XPU, IdType, Dtype>(
                  op, bcast, vec_csr, lhs, rhs, out, lhs_target, rhs_target,
                  lhs_eid, rhs_eid);
            } else if (format == SparseFormat::kCOO) {
              std::vector<COOMatrix> vec_coo;
              for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes();
                   ++etype) {
                vec_coo.push_back(graph->GetCOOMatrix(etype));
              }
              SDDMMCooHetero<XPU, IdType, Dtype>(
                  op, bcast, vec_coo, lhs, rhs, out, lhs_target, rhs_target,
                  lhs_eid, rhs_eid);
            } else {
              LOG(FATAL) << "SDDMM only supports CSR and COO formats";
            }
          });
    });
  });
}

/** @brief Generalized Edge_softmax op for forward */
void Edge_softmax_forward(
    const std::string& op, HeteroGraphPtr graph, NDArray ufeat, NDArray efeat,
    NDArray out) {
  // TODO(zhejiang): add gpu op for edge_softmax
  const auto& bcast = CalcBcastOff(op, ufeat, efeat);

  ATEN_XPU_SWITCH(graph->Context().device_type, XPU, "edge_softmax", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(
          out->dtype, Dtype, XPU, "edge_softmax out data", {
            Edge_softmax_csr_forward<XPU, IdType, Dtype>(
                op, bcast, graph->GetCSCMatrix(0), ufeat, efeat, out);
          });
    });
  });
}

/** @brief Generalized Edge_softmax op for backward */
void Edge_softmax_backward(
    const std::string& op, HeteroGraphPtr graph, NDArray out, NDArray sds,
    NDArray back_out, NDArray ufeat) {
  // TODO(zhejiang): add gpu op for edge_softmax
  const auto& bcast = CalcBcastOff(op, ufeat, sds);

  ATEN_XPU_SWITCH(graph->Context().device_type, XPU, "edge_softmax_back", {
    ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(
          out->dtype, Dtype, XPU, "edge_softmax out data_back", {
            Edge_softmax_csr_backward<XPU, IdType, Dtype>(
                op, bcast, graph->GetCSCMatrix(0), out, sds, back_out);
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

/** @brief Segment reduce dispatch function. */
void SegmentReduceDispatch(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg) {
  ATEN_XPU_SWITCH_CUDA(feat->ctx.device_type, XPU, "SegmentReduce", {
    ATEN_ID_TYPE_SWITCH(offsets->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(feat->dtype, Dtype, XPU, "Feature data", {
        SegmentReduce<XPU, IdType, Dtype>(op, feat, offsets, out, arg);
      });
    });
  });
}

/** @brief Scatter Add (on first dimension) dispatch function. */
void ScatterAddDispatch(NDArray feat, NDArray idx, NDArray out) {
  ATEN_XPU_SWITCH_CUDA(feat->ctx.device_type, XPU, "ScatterAdd", {
    ATEN_ID_TYPE_SWITCH(idx->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(feat->dtype, Dtype, XPU, "Feature data", {
        ScatterAdd<XPU, IdType, Dtype>(feat, idx, out);
      });
    });
  });
}

/** @brief Update gradients (reduce op max/min) dispatch function on
 * heterogeneous graph. */
void UpdateGradMinMaxDispatchHetero(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  auto pair = graph->meta_graph()->FindEdge(0);  // checking the first etype
  auto src_id = pair.first;
  ATEN_XPU_SWITCH_CUDA(feat[src_id]->ctx.device_type, XPU, "ScatterAdd", {
    ATEN_ID_TYPE_SWITCH(idx[src_id]->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(
          feat[src_id]->dtype, Dtype, XPU, "Feature data", {
            UpdateGradMinMax_hetero<XPU, IdType, Dtype>(
                graph, op, feat, idx, idx_etype, out);
          });
    });
  });
}

/** @brief Backward segment cmp dispatch function.*/
void BackwardSegmentCmpDispatch(NDArray feat, NDArray arg, NDArray out) {
  ATEN_XPU_SWITCH_CUDA(feat->ctx.device_type, XPU, "BackwardSegmentCmp", {
    ATEN_ID_TYPE_SWITCH(arg->dtype, IdType, {
      ATEN_FLOAT_TYPE_SWITCH_16BITS(feat->dtype, Dtype, XPU, "Feature data", {
        BackwardSegmentCmp<XPU, IdType, Dtype>(feat, arg, out);
      });
    });
  });
}

std::pair<CSRMatrix, NDArray> CSRMM(
    CSRMatrix A, NDArray A_weights, CSRMatrix B, NDArray B_weights) {
  CHECK_EQ(A.num_cols, B.num_rows)
      << "The number of nodes of destination node type of the first graph must "
         "be the "
         "same as the number of nodes of source node type of the second graph.";
  CheckCtx(
      A.indptr->ctx, {A_weights, B_weights},
      {"A's edge weights", "B's edge weights"});
  CHECK_EQ(A.indptr->ctx, B.indptr->ctx) << "Device of two graphs must match.";
  CHECK_EQ(A.indptr->dtype, B.indptr->dtype)
      << "ID types of two graphs must match.";
  CHECK_EQ(A_weights->dtype, B_weights->dtype)
      << "Data types of two edge weights must match.";

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
    const std::vector<CSRMatrix>& A, const std::vector<NDArray>& A_weights) {
  CHECK(A.size() > 0) << "The list of graphs must not be empty.";
  CHECK_EQ(A.size(), A_weights.size())
      << "The list of edge weights must have the same length as the list of "
         "graphs.";
  const auto ctx = A[0].indptr->ctx;
  const auto idtype = A[0].indptr->dtype;
  const auto dtype = A_weights[0]->dtype;
  const auto num_rows = A[0].num_rows;
  const auto num_cols = A[0].num_cols;
  for (size_t i = 0; i < A.size(); ++i) {
    CHECK_EQ(A[i].indptr->ctx, ctx)
        << "The devices of all graphs must be equal.";
    CHECK_EQ(A[i].indptr->dtype, idtype)
        << "The ID types of all graphs must be equal.";
    CHECK_EQ(A[i].indices->shape[0], A_weights[i]->shape[0])
        << "Shape of edge weights does not match the number of edges.";
    CHECK_EQ(A_weights[i]->ctx, ctx) << "The devices of edge weights must be "
                                        "the same as that of the graphs.";
    CHECK_EQ(A_weights[i]->dtype, dtype)
        << "The data types of all edge weights must be equal.";
    CHECK_EQ(A[i].num_rows, num_rows)
        << "Graphs must have the same number of nodes.";
    CHECK_EQ(A[i].num_cols, num_cols)
        << "Graphs must have the same number of nodes.";
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
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      const std::string op = args[1];
      const std::string reduce_op = args[2];
      NDArray U = args[3];
      NDArray E = args[4];
      NDArray V = args[5];
      NDArray ArgU = args[6];
      NDArray ArgE = args[7];
      CheckCtx(
          graph->Context(), {U, E, V, ArgU, ArgE},
          {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
      CheckContiguous(
          {U, E, V, ArgU, ArgE}, {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
      CHECK_EQ(graph->NumEdgeTypes(), 1);
      auto pair =
          graph->meta_graph()->FindEdge(0);  // only one etype in the graph.
      const dgl_type_t src_vtype = pair.first;
      const dgl_type_t dst_vtype = pair.second;
      CheckShape(
          {graph->NumVertices(src_vtype), graph->NumEdges(0),
           graph->NumVertices(dst_vtype)},
          {0, 1, 2, 2, 2}, {U, E, V, ArgU, ArgE},
          {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
      SpMM(op, reduce_op, graph.sptr(), U, E, V, {ArgU, ArgE});
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelGATHERMM")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray A = args[0];
      NDArray B = args[1];
      NDArray C = args[2];
      NDArray idx_a = args[3];
      NDArray idx_b = args[4];
      GatherMM(A, B, C, idx_a, idx_b);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelGATHERMMSCATTER")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray A = args[0];
      NDArray B = args[1];
      NDArray C = args[2];
      NDArray idx_a = args[3];
      NDArray idx_b = args[4];
      NDArray idx_c = args[5];
      GatherMMScatter(A, B, C, idx_a, idx_b, idx_c);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSEGMENTMM")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray A = args[0];
      NDArray B = args[1];
      NDArray C = args[2];
      NDArray seglen_A = args[3];
      bool A_trans = args[4];
      bool B_trans = args[5];
      SegmentMM(A, B, C, seglen_A, A_trans, B_trans);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSEGMENTMMBackwardB")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray A = args[0];
      NDArray dC = args[1];
      NDArray dB = args[2];
      NDArray seglen = args[3];
      SegmentMMBackwardB(A, dC, dB, seglen);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelEdge_softmax_forward")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      const std::string op = args[1];
      NDArray U = args[2];
      NDArray E = args[3];
      NDArray V = args[4];
      Edge_softmax_forward(op, graph.sptr(), U, E, V);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelEdge_softmax_backward")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      const std::string op = args[1];
      NDArray out = args[2];
      NDArray sds = args[3];
      NDArray back_out = args[4];
      NDArray ufeat = args[5];
      Edge_softmax_backward(op, graph.sptr(), out, sds, back_out, ufeat);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSpMMHetero")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      const std::string op = args[1];
      const std::string reduce_op = args[2];
      List<Value> list_U = args[3];
      List<Value> list_E = args[4];
      List<Value> list_V = args[5];
      List<Value> list_ArgU = args[6];
      List<Value> list_ArgE = args[7];
      List<Value> list_ArgU_ntype = args[8];
      List<Value> list_ArgE_etype = args[9];
      std::vector<std::vector<NDArray>> Arg_vec;  // ArgU + ArgE
      for (int i = 0; i < 4; ++i) {  // ArgU + ArgE + ArgU_ntype + ArgE_etype
        Arg_vec.push_back(std::vector<NDArray>());
      }
      std::vector<NDArray> U_vec = ListValueToVector<NDArray>(list_U);
      std::vector<NDArray> V_vec = ListValueToVector<NDArray>(list_V);
      std::vector<NDArray> E_vec = ListValueToVector<NDArray>(list_E);
      Arg_vec[0] = ListValueToVector<NDArray>(list_ArgU);
      Arg_vec[1] = ListValueToVector<NDArray>(list_ArgE);
      Arg_vec[2] = ListValueToVector<NDArray>(list_ArgU_ntype);
      Arg_vec[3] = ListValueToVector<NDArray>(list_ArgE_etype);
      for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
        auto pair = graph->meta_graph()->FindEdge(etype);
        const dgl_id_t src_id = pair.first;
        const dgl_id_t dst_id = pair.second;
        NDArray U = (U_vec.size() == 0) ? NullArray() : U_vec[src_id];
        NDArray E = (E_vec.size() == 0) ? NullArray() : E_vec[etype];
        CheckCtx(
            graph->Context(),
            {U, E, V_vec[dst_id], Arg_vec[0][dst_id], Arg_vec[1][dst_id]},
            {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
        CheckContiguous(
            {U, E, V_vec[dst_id], Arg_vec[0][dst_id], Arg_vec[1][dst_id]},
            {"U_data", "E_data", "out", "Arg_U", "Arg_E"});
      }
      SpMMHetero(op, reduce_op, graph.sptr(), U_vec, E_vec, &V_vec, &Arg_vec);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSDDMM")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
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
      auto pair =
          graph->meta_graph()->FindEdge(0);  // only one etype in the graph.
      const dgl_type_t src_vtype = pair.first;
      const dgl_type_t dst_vtype = pair.second;

      CheckShape(
          {graph->NumVertices(src_vtype), graph->NumEdges(0),
           graph->NumVertices(dst_vtype)},
          {lhs_target, rhs_target, 1}, {lhs, rhs, out},
          {"U_data", "E_data", "V_data"});
      SDDMM(op, graph.sptr(), lhs, rhs, out, lhs_target, rhs_target);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSDDMMHetero")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
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
      SDDMMHetero(
          op, graph.sptr(), vec_lhs, vec_rhs, vec_out, lhs_target, rhs_target);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelSegmentReduce")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
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
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray feat = args[0];
      NDArray idx = args[1];
      NDArray out = args[2];
      CheckCtx(feat->ctx, {feat, idx, out}, {"feat", "idx", "out"});
      CheckContiguous({feat, idx, out}, {"feat", "idx", "out"});
      ScatterAddDispatch(feat, idx, out);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelUpdateGradMinMaxHetero")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      const std::string op = args[1];
      List<Value> list_feat = args[2];
      List<Value> list_idx = args[3];
      List<Value> list_idx_etype = args[4];
      List<Value> list_out = args[5];
      std::vector<NDArray> vec_feat = ListValueToVector<NDArray>(list_feat);
      std::vector<NDArray> vec_idx = ListValueToVector<NDArray>(list_idx);
      std::vector<NDArray> vec_idx_etype =
          ListValueToVector<NDArray>(list_idx_etype);
      std::vector<NDArray> vec_out = ListValueToVector<NDArray>(list_out);
      // CheckCtx(feat->ctx, {feat, idx, out}, {"feat", "idx", "out"});
      // CheckContiguous({feat, idx, out}, {"feat", "idx", "out"});
      UpdateGradMinMaxDispatchHetero(
          graph.sptr(), op, vec_feat, vec_idx, vec_idx_etype, &vec_out);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelBwdSegmentCmp")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      NDArray feat = args[0];
      NDArray arg = args[1];
      NDArray out = args[2];
      CheckCtx(feat->ctx, {feat, arg, out}, {"feat", "arg", "out"});
      CheckContiguous({feat, arg, out}, {"feat", "arg", "out"});
      BackwardSegmentCmpDispatch(feat, arg, out);
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelGetEdgeMapping")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef graph = args[0];
      *rv = GetEdgeMapping(graph);
    });

/**
 * @brief Sparse matrix multiplication with graph interface.
 *
 * @param A_ref The left operand.
 * @param A_weights The edge weights of graph A.
 * @param B_ref The right operand.
 * @param B_weights The edge weights of graph B.
 * @param num_vtypes The number of vertex types of the graph to be returned.
 * @return A pair consisting of the new graph as well as its edge weights.
 */
DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRMM")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const HeteroGraphRef A_ref = args[0];
      NDArray A_weights = args[1];
      const HeteroGraphRef B_ref = args[2];
      NDArray B_weights = args[3];
      int num_vtypes = args[4];

      const HeteroGraphPtr A = A_ref.sptr();
      const HeteroGraphPtr B = B_ref.sptr();
      CHECK_EQ(A->NumEdgeTypes(), 1)
          << "The first graph must have only one edge type.";
      CHECK_EQ(B->NumEdgeTypes(), 1)
          << "The second graph must have only one edge type.";
      const auto A_csr = A->GetCSRMatrix(0);
      const auto B_csr = B->GetCSRMatrix(0);
      auto result = CSRMM(A_csr, A_weights, B_csr, B_weights);

      List<ObjectRef> ret;
      ret.push_back(
          HeteroGraphRef(CreateFromCSR(num_vtypes, result.first, ALL_CODE)));
      ret.push_back(Value(MakeValue(result.second)));
      *rv = ret;
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRSum")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      List<HeteroGraphRef> A_refs = args[0];
      List<Value> A_weights = args[1];

      std::vector<NDArray> weights = ListValueToVector<NDArray>(A_weights);
      std::vector<CSRMatrix> mats;
      mats.reserve(A_refs.size());
      int num_vtypes = 0;
      for (auto A_ref : A_refs) {
        const HeteroGraphPtr A = A_ref.sptr();
        CHECK_EQ(A->NumEdgeTypes(), 1)
            << "Graphs must have only one edge type.";
        mats.push_back(A->GetCSRMatrix(0));
        if (num_vtypes == 0) num_vtypes = A->NumVertexTypes();
      }
      auto result = CSRSum(mats, weights);

      List<ObjectRef> ret;
      ret.push_back(
          HeteroGraphRef(CreateFromCSR(num_vtypes, result.first, ALL_CODE)));
      ret.push_back(Value(MakeValue(result.second)));
      *rv = ret;
    });

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLCSRMask")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      const HeteroGraphRef A_ref = args[0];
      NDArray A_weights = args[1];
      const HeteroGraphRef B_ref = args[2];

      const HeteroGraphPtr A = A_ref.sptr();
      const HeteroGraphPtr B = B_ref.sptr();
      CHECK_EQ(A->NumEdgeTypes(), 1)
          << "Both graphs must have only one edge type.";
      CHECK_EQ(B->NumEdgeTypes(), 1)
          << "Both graphs must have only one edge type.";
      const CSRMatrix& A_csr = A->GetCSRMatrix(0);
      const COOMatrix& B_coo = B->GetCOOMatrix(0);
      CHECK_EQ(A_csr.num_rows, B_coo.num_rows)
          << "Both graphs must have the same number of nodes.";
      CHECK_EQ(A_csr.num_cols, B_coo.num_cols)
          << "Both graphs must have the same number of nodes.";

      NDArray result;
      ATEN_FLOAT_TYPE_SWITCH(A_weights->dtype, DType, "Edge weights", {
        result =
            aten::CSRGetData<DType>(A_csr, B_coo.row, B_coo.col, A_weights, 0.);
      });
      *rv = result;
    });

}  // namespace aten
}  // namespace dgl
