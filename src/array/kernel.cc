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


//**************************************************************************************
// DistGNN functions

template<typename IdType>
int32_t node2partition(IdType in_val, int32_t* node_map, int32_t num_parts) {

    int32_t pos = 0;
    for (int p=0; p<num_parts; p++) {
        if (in_val < node_map[p])
            return pos;
        pos = pos + 1;
    }
    printf("Error: func node2partition\n");
    exit(EXIT_FAILURE);
}


int32_t map2local_index_(int32_t otf, int32_t *node_map,
                         int32_t num_parts,
                         int32_t &pid, int32_t cur_part)
{
    int last_p = 0;
    if (cur_part != 0)
        last_p = node_map[cur_part - 1];
    for (int i=cur_part; i<num_parts; i++) {
        int nnodes = node_map[i];
        if (otf < nnodes) {
            pid = i;
            return otf - last_p;
        }
        last_p = nnodes;
    }
    printf("Error: func map2local_index\n");
    exit(EXIT_FAILURE);
}


template<typename IdType, typename DType>
void fdrpa_gather_emb_v41(
    NDArray feat_,
    int64_t feat_shape,
    NDArray adj_,
    NDArray send_feat_list_,
    int64_t offset,
    NDArray send_node_list_,
    NDArray send_to_node_list_,    
    NDArray selected_nodes_,
    NDArray in_degs_,
    NDArray node2part_,
    NDArray node2part_index_,
    int32_t width,
    int32_t feat_size,
    int32_t cur_part,
    int64_t soffset_base,
    int64_t soffset_cur,
    NDArray node_map_,
    int32_t num_parts    
    )
{
    if (soffset_cur == 0)
        return;
        
    DType*   feat            = feat_.Ptr<DType>();
    IdType*  adj             = adj_.Ptr<IdType>();
    DType*   send_feat_list  = send_feat_list_.Ptr<DType>() + offset * (feat_size + 1);
    int32_t* send_node_list  = send_node_list_.Ptr<int32_t>();
    int32_t* send_to_node_list  = send_to_node_list_.Ptr<int32_t>() + offset;
    int32_t* selected_nodes  = selected_nodes_.Ptr<int32_t>();
    DType*   in_degs         = in_degs_.Ptr<DType>();
    int32_t* node2part       = node2part_.Ptr<int32_t>();
    int32_t* node2part_index = node2part_index_.Ptr<int32_t>();
    int32_t* node_map        = node_map_.Ptr<int32_t>();
    
    int32_t pos = 0, pos_out = 0;
    int32_t len = selected_nodes_.GetSize() >> 2;
    // int64_t fs = feat_.GetSize() >> 2;
    
    int32_t *gindex = (int32_t*)calloc(sizeof(int32_t), len);
    int lim = send_node_list_.GetSize() >> 2;
    int counter = 0;
    pos = 0;
    int max_val = 0, max_val2 = 0, max_val2_i = 0;
    for (int i=0; i<len; i++)
    {
        if (node2part[i] == cur_part) {
            if (counter >= soffset_base && pos < soffset_cur) {
                gindex[pos++] = i;
                // if (max_val < selected_nodes[i]) max_val = selected_nodes[i];
            }
            counter ++;
        }
    }
    
    assert(pos == soffset_cur);
    #pragma omp parallel for
    for (int64_t i=0; i<pos; i++)
    {
        int id = gindex[i];
        int64_t index = selected_nodes[id];
    
        #if CHECKS
        if (index >= feat_shape) {
            printf("index: %d, feat_shape: %d\n", index, feat_shape);
            fflush(0);
            assert(index < feat_shape);
        }       
        assert(index >= 0);
        #endif
        
        DType *iptr = feat + index * feat_size;
        
        send_node_list[i] = index;        
        DType *optr = send_feat_list + i * (feat_size + 1);
        optr[0] = in_degs[index];
        send_to_node_list[i] = node2part_index[id];
    
        #if DEBUG
        int32_t  p = node2partition<int32_t>(node2part_index[id], node_map, num_parts);
        if (cur_part != p)
            printf("Not matching: %d %d\n", cur_part, p);
        assert(cur_part == p);
        #endif
        
        #pragma omp simd
        for (int64_t k=0; k<feat_size; k++) {
            optr[1 + k] = iptr[k];
        }
    
    }

    free(gindex);

}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelfdrpa_gather_emb_v41_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray feat_              = args[0];
    int64_t feat_shape_        = args[1];
    NDArray adj_               = args[2];
    NDArray send_feat_list_    = args[3];
    int64_t offset_            = args[4];
    NDArray send_node_list_    = args[5];
    NDArray send_to_node_list_ = args[6];
    NDArray selected_nodes_    = args[7];
    NDArray in_degs_           = args[8];
    NDArray node2part_         = args[9];
    NDArray node2part_index_   = args[10];
    int     width_             = args[11];
    int     feat_size_         = args[12];
    int     cur_part_          = args[13];
    int64_t soffset_base_      = args[14];
    int64_t soffset_cur_       = args[15];
    NDArray node_map_          = args[16];
    int32_t num_parts_         = args[17];

    ATEN_ID_TYPE_SWITCH(adj_->dtype, IdType, {
            ATEN_FLOAT_TYPE_SWITCH(feat_->dtype, DType, "Feature data", {
                    fdrpa_gather_emb_v41<IdType, DType>(feat_, feat_shape_,
                                                       adj_, send_feat_list_, offset_,
                                                       send_node_list_, send_to_node_list_,
                                                       selected_nodes_,
                                                       in_degs_, node2part_, node2part_index_,
                                                       width_,    feat_size_, cur_part_,
                                                       soffset_base_, soffset_cur_,
                                                       node_map_,
                                                       num_parts_);

                });
        });

});


template<typename IdType, typename DType>
void scatter_reduce_v41(
    NDArray otf_,
    int64_t offsetf,
    NDArray otn_,
    int64_t offsetn,    
    NDArray feat_,    
    NDArray in_degs_,
    NDArray node_map_,
    int64_t dim,
    int64_t feat_size,
    int64_t num_parts,
    NDArray recv_list_nodes_,
    NDArray pos_,
    int64_t lim,
    int32_t cur_part)
{
    DType*   otf             = otf_.Ptr<DType>() + offsetf;
    int32_t* otn             = otn_.Ptr<int32_t>() + offsetn;
    DType*   feat            = feat_.Ptr<DType>();
    DType*  in_degs          = in_degs_.Ptr<DType>();
    int32_t* node_map        = node_map_.Ptr<int32_t>();
    int32_t* recv_list_nodes = recv_list_nodes_.Ptr<int32_t>();
    int64_t* pos             = pos_.Ptr<int64_t>();

    int64_t size = in_degs_.GetSize() >> 2;
    int64_t fsize = feat_.GetSize() >> 2;

    #pragma omp parallel for
    for (int64_t i=0; i<lim; i++)
    {
        DType *iptr = otf + i * (feat_size + 1);
        if(iptr[0] < 0) {printf("Error: -ve index, i: %d, iptr: %f\n", i, iptr[0]);fflush(0);}
        assert(iptr[0] >= 0);

        int32_t pid = -1;
        int64_t index = map2local_index_(otn[i], node_map, num_parts, pid, cur_part);
        assert(pid == cur_part);
        if (index < 0) printf("Error: index: %d\n", index);

        assert(index < size && index >=0 );
        in_degs[index] += iptr[0];

        recv_list_nodes[i] = index;

        DType *optr = feat + index * feat_size;

        if ((i+1)*(feat_size+1) > dim) {
            printf("Error: lim: %d %d\n", i+1+feat_size, dim);fflush(0);exit(EXIT_FAILURE);
        }
        if( (index+1)*feat_size > fsize) {
            printf("Error: fsize, %d %d %d\n", (index+1)*feat_size, fsize, feat_size);
            fflush(0);exit(EXIT_FAILURE);
        }
        
        #pragma omp simd
        for (int j=0; j<feat_size; j++)
            optr[j] += iptr[1 + j];
    }
    pos[0] += lim;
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelscatter_reduce_v41_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray otf_             = args[0];
    int64_t offsetf_         = args[1];
    NDArray otn_             = args[2];
    int64_t offsetn_         = args[3];    
    NDArray feat_            = args[4];
    NDArray in_degs_         = args[5];
    NDArray node_map_        = args[6];
    int64_t dim_             = args[7];
    int64_t feat_size_       = args[8];
    int64_t num_parts_       = args[9];
    NDArray recv_list_nodes_ = args[10];
    NDArray pos_             = args[11];
    int64_t lim_             = args[12];
    int64_t cur_part_        = args[13];

    ATEN_FLOAT_TYPE_SWITCH(in_degs_->dtype, IdType, "Id data", {
            ATEN_FLOAT_TYPE_SWITCH(otf_->dtype, DType, "Feature data", {
                    scatter_reduce_v41<IdType, DType>(otf_, offsetf_, otn_, offsetn_,
                                                  feat_, in_degs_, node_map_,
                                                  dim_, feat_size_, num_parts_,
                                                  recv_list_nodes_, pos_, lim_, cur_part_);
                });
        });

});


template<typename DType>
void fdrpa_gather_emb_v42(
    NDArray feat_,
    int64_t feat_shape,
    NDArray send_feat_list_,
    int64_t offset,
    NDArray recv_list_nodes_,
    int64_t lim,
    NDArray in_degs_,
    int32_t feat_size,
    int32_t cur_part,
    NDArray node_map_,
    int32_t num_parts    
    )
{
    DType*   feat            = feat_.Ptr<DType>();
    DType*   send_feat_list  = send_feat_list_.Ptr<DType>() + offset;
    int32_t* recv_list_nodes  = recv_list_nodes_.Ptr<int32_t>();
    int32_t* node_map        = node_map_.Ptr<int32_t>();
    DType*   in_degs         = in_degs_.Ptr<DType>();
    
    int32_t pos = 0, pos_out = 0;

    #pragma omp parallel for
    for (int64_t i=0; i<lim; i++)
    {
        int64_t index = recv_list_nodes[i];
        if (index >= feat_shape) {
            printf("index: %d, feat_shape: %d\n", index, feat_shape);
            fflush(0);
        }
        assert(index < feat_shape);

        DType *iptr = feat + index * feat_size;
        
        DType *optr = send_feat_list + i * (feat_size + 1);
        optr[0] = in_degs[index];
        
        #pragma omp simd
        for (int k=0; k<feat_size; k++)
            optr[1 + k] = iptr[k];
    }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelfdrpa_gather_emb_v42_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray feat_              = args[0];
    int64_t feat_shape_        = args[1];
    NDArray send_feat_list_    = args[2];
    int64_t offset_            = args[3];
    NDArray recv_list_nodes_   = args[4];
    int64_t lim                = args[5];
    NDArray in_degs_           = args[6];
    int     feat_size_         = args[7];
    int     cur_part_          = args[8];
    NDArray node_map_          = args[9];
    int32_t num_parts_         = args[10];

    ATEN_FLOAT_TYPE_SWITCH(feat_->dtype, DType, "Feature data", {
            fdrpa_gather_emb_v42<DType>(feat_, feat_shape_,
                                        send_feat_list_, offset_,
                                        recv_list_nodes_, lim,
                                        in_degs_,
                                        feat_size_,
                                        cur_part_,
                                        node_map_,
                                        num_parts_);
        });
    
});

template<typename DType>
void scatter_reduce_v42(
    NDArray otf_,
    int64_t offset,
    NDArray stn_,
    int64_t lim,
    NDArray in_degs_,
    NDArray feat_,    
    NDArray node_map_,
    int64_t dim,
    int64_t feat_size,
    int64_t num_parts,
    int32_t cur_part)
{
    DType*   otf             = otf_.Ptr<DType>() + offset;
    int32_t* stn             = stn_.Ptr<int32_t>();
    DType*   feat            = feat_.Ptr<DType>();
    int32_t* node_map        = node_map_.Ptr<int32_t>();
    DType*   in_degs         = in_degs_.Ptr<DType>();

    int64_t fsize = feat_.GetSize() >> 2;

    #pragma omp parallel for
    for (int64_t i=0; i<lim; i++)
    {
        DType *iptr = otf + i * (feat_size + 1); 

        int64_t index = stn[i];
        DType *optr = feat + index * feat_size;
        in_degs[index] = iptr[0];
        if ((i+1)*(1+feat_size) > dim) {
            printf("Error2: lim: %d %d\n", i+1+feat_size, dim);fflush(0);exit(EXIT_FAILURE);
        }
        if( (index+1)*feat_size > fsize) {
            printf("Error2: fsize, %d %d %d\n", (index+1)*feat_size, fsize, feat_size);
            fflush(0);exit(EXIT_FAILURE);
        }

        #pragma simd
        for (int j=0; j<feat_size; j++)
            optr[j] = iptr[1+j];    
    }
}


DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelscatter_reduce_v42_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray otf_             = args[0];
    int64_t offset_          = args[1];
    NDArray stn_             = args[2];
    int64_t lim_             = args[3];
    NDArray in_degs_         = args[4];
    NDArray feat_            = args[5];
    NDArray node_map_        = args[6];
    int64_t dim_             = args[7];
    int64_t feat_size_       = args[8];
    int64_t num_parts_       = args[9];
    int64_t cur_part_        = args[10];

    ATEN_FLOAT_TYPE_SWITCH(otf_->dtype, DType, "Feature data", {
            scatter_reduce_v42<DType>(otf_, offset_, stn_, lim_,
                                      in_degs_,
                                      feat_, node_map_,
                                      dim_, feat_size_, num_parts_,
                                      cur_part_);
        });
});

template<typename IdType>
void fdrpa_get_buckets_v4(
    NDArray adj_,
    NDArray selected_nodes_,
    NDArray node2part_,
    NDArray node2part_index_,
    NDArray node_map_,
    NDArray buckets_,
    NDArray lf_,
    NDArray num_sel_nodes_,
    int width,
    int num_parts,
    int cur_part)
{
    IdType*  adj             = adj_.Ptr<IdType>();
    int32_t* selected_nodes  = selected_nodes_.Ptr<int32_t>();
    int32_t* node2part       = node2part_.Ptr<int32_t>();
    int32_t* node2part_index = node2part_index_.Ptr<int32_t>();
    int32_t* node_map        = node_map_.Ptr<int32_t>();
    int32_t* buckets         = buckets_.Ptr<int32_t>();
    int32_t* lf              = lf_.Ptr<int32_t>();
    int32_t* num_sel_nodes   = num_sel_nodes_.Ptr<int32_t>();
    
    int32_t nthreads = omp_get_max_threads();
    int32_t *gbucket = (int32_t*) calloc (sizeof(int32_t), nthreads * num_parts);
    
    int32_t tnodes = 0;
    int32_t len = selected_nodes_.GetSize() >> 2;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector<std::pair<int32_t, int32_t> > cache(num_parts);
        #pragma omp for
        for(int64_t i=0; i<len; i++)
        {
            int64_t index = selected_nodes[i];
            int lf_val = lf[index];
            assert(lf_val != -200);
            int32_t p = node2partition<IdType>(lf_val, node_map, num_parts);
            
            IdType *ptr = adj + width * index;
            int32_t min_part = p;
            int32_t min_index = lf_val;
            
            if (min_part != cur_part) {
                // debug code
                bool flg = 1;
                for (int j=0; j<width; j++) {
                    if (ptr[j] >= 0)
                    {
                        if(ptr[j] == lf_val)
                            flg = 0;
                    }                
                    else                         
                        break;
                }                
                assert(flg == 0);
                // debug code ends
                gbucket[tid*num_parts + min_part] ++;
                node2part[i] = min_part;
                node2part_index[i] = min_index;
            }
            else {
                node2part[i] = -100;
            }                
        }
    }

    // aggregate
    for (int t=0; t<nthreads; t++)
    {
        for (int p=0; p<num_parts; p++)
            buckets[p] += gbucket[t*num_parts + p];
    }
    for (int p=0; p<num_parts; p++) tnodes += buckets[p];
    
    num_sel_nodes[0] = tnodes;
    free(gbucket);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelfdrpa_get_buckets_v4_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray adj_             = args[0];
    NDArray selected_nodes_  = args[1];
    NDArray node2part_       = args[2];
    NDArray node2part_index_ = args[3];
    NDArray node_map_        = args[4];
    NDArray buckets_         = args[5];
    NDArray lf_              = args[6];
    NDArray num_sel_nodes_   = args[7];
    int     width_           = args[8];
    int     num_parts_       = args[9];
    int     cur_part_        = args[10];

    ATEN_ID_TYPE_SWITCH(adj_->dtype, IdType, {
            fdrpa_get_buckets_v4<IdType>(adj_, selected_nodes_,
                                         node2part_, node2part_index_,
                                         node_map_,
                                         buckets_, lf_,
                                         num_sel_nodes_, width_,
                                         num_parts_, cur_part_);
        });
});


template<typename IdType>
void fdrpa_init_buckets_v4(
    NDArray adj_,
    NDArray selected_nodes_,
    NDArray node_map_,
    NDArray buckets_,
    NDArray lf_,
    int width,
    int num_parts,
    int cur_part)
{
    IdType*  adj             = adj_.Ptr<IdType>();
    int32_t* selected_nodes  = selected_nodes_.Ptr<int32_t>();
    int32_t* node_map        = node_map_.Ptr<int32_t>();
    int32_t* buckets         = buckets_.Ptr<int32_t>();
    int32_t* lf              = lf_.Ptr<int32_t>();
    
    int32_t nthreads = omp_get_max_threads();
    int32_t *gbucket = (int32_t*) calloc (sizeof(int32_t), nthreads * num_parts);
    
    int32_t len = selected_nodes_.GetSize() >> 2;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for(int64_t i=0; i<len; i++)
        {
            int64_t index = selected_nodes[i];
            int lf_val = lf[index];
            assert(lf_val != -200);
            int32_t p = node2partition<IdType>(lf_val, node_map, num_parts);
            
            IdType *ptr = adj + width * index;
            int32_t min_part = p;
            int32_t min_index = lf_val;
            
            if (min_part != cur_part) {
                // debug code
                bool flg = 1;
                for (int j=0; j<width; j++) {
                    if (ptr[j] >= 0)
                    {
                        if(ptr[j] == lf_val)
                            flg = 0;
                    }                
                    else                         
                        break;
                }                
                assert(flg == 0);
                // debug code ends
                gbucket[tid*num_parts + min_part] ++;
            }
        }
    }

    // aggregate
    for (int t=0; t<nthreads; t++)
    {
        for (int p=0; p<num_parts; p++)
            buckets[p] += gbucket[t*num_parts + p];
    }

    free(gbucket);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKernelfdrpa_init_buckets_v4_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray adj_             = args[0];
    NDArray selected_nodes_  = args[1];
    NDArray node_map_        = args[2];
    NDArray buckets_         = args[3];
    NDArray lf_              = args[4];
    int     width_           = args[5];
    int     num_parts_       = args[6];
    int     cur_part_        = args[7];

    ATEN_ID_TYPE_SWITCH(adj_->dtype, IdType, {
            fdrpa_init_buckets_v4<IdType>(adj_, selected_nodes_,
                                          node_map_,
                                          buckets_, lf_,
                                          width_,
                                          num_parts_,
                                          cur_part_);
        });
});


template<typename DType>
void deg_div(
    NDArray neigh_,    
    NDArray h_,        
    int64_t feat_size,
    NDArray in_degs_,  
    int64_t lim)
{
    DType*  neigh = neigh_.Ptr<DType>();
    DType*  h     = h_.Ptr<DType>();
    DType* degs  = in_degs_.Ptr<DType>();

    #pragma omp parallel for
    for (int i=0; i<lim; i++) {
        DType deg = degs[i];
        DType *ptrn = neigh + i* feat_size;
        DType *ptrh = h     + i* feat_size;
        for (int j=0; j<feat_size; j++) {

            ptrn[j] = (ptrn[j] +  ptrh[j]) / (deg + 1);
        }

    }
}
    
DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKerneldeg_div_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray neigh_    = args[0];
    NDArray h_        = args[1];
    int64_t feat_size_= args[2];
    NDArray in_degs_  = args[3];
    int64_t lim_      = args[4];

    ATEN_FLOAT_TYPE_SWITCH(neigh_->dtype, DType, "Feature data", {
            deg_div<DType>(neigh_, h_, feat_size_, in_degs_, lim_);
        });
});

template<typename DType>
void deg_div_back(
    NDArray neigh_grad_,    
    int64_t feat_size,
    NDArray in_degs_,  
    int64_t lim)
{
    DType*  neigh_grad = neigh_grad_.Ptr<DType>();
    DType* degs  = in_degs_.Ptr<DType>();

    #pragma omp parallel for
    for (int i=0; i<lim; i++) {
        DType deg = degs[i];
        DType *ptrn = neigh_grad + i* feat_size;
        for (int j=0; j<feat_size; j++) {

            ptrn[j] = ptrn[j]  / (deg + 1);
        }

    }
}
    
DGL_REGISTER_GLOBAL("sparse._CAPI_DGLKerneldeg_div_back_")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray neigh_    = args[0];
    int64_t feat_size_= args[1];
    NDArray in_degs_  = args[2];
    int64_t lim_      = args[3];

    ATEN_FLOAT_TYPE_SWITCH(neigh_->dtype, DType, "Feature data", {
            deg_div_back<DType>(neigh_, feat_size_, in_degs_, lim_);
        });
});




//**************************************************************************************
// Libra partitioning stuff
int leastload(int64_t* community_edges, int nc)
{
    /* initialize random seed: */
    srand (time(NULL));
    
    std::vector<int> score, loc;
    int min = 1e9;
    for (int i=0; i<nc; i++)
    {
        if (community_edges[i] < min) {
            min = community_edges[i];
        }
    }
    for (int i=0; i<nc; i++)
    {
        if (community_edges[i] == min) {
            loc.push_back(i);
        }
    }
    
    int r = rand() % loc.size();
    assert(loc[r] < nc);
    return loc[r];    
}

template<typename IdType, typename IdType2, typename DType>
int vertex_cut(
    int32_t nc,
    NDArray node_degree_,
    NDArray edgenum_unassigned_,
    NDArray community_weights_,
    NDArray u_,
    NDArray v_,
    NDArray w_,
    NDArray out_,
    int64_t N_n,
    int64_t N_e,
    std::string prefix
    )
{
    int32_t *out = out_.Ptr<int32_t>();
    IdType *node_degree = node_degree_.Ptr<IdType>();
    IdType *edgenum_unassigned = edgenum_unassigned_.Ptr<IdType>();
    IdType2 *uptr = u_.Ptr<IdType2>();
    IdType2 *vptr = v_.Ptr<IdType2>();
    float   *wptr = w_.Ptr<float>();
    float *community_weights = community_weights_.Ptr<float>();

    std::vector<std::vector<int32_t> > node_assignments(N_n);
    std::vector<IdType2> replication_list;
    
    // local allocations
    int64_t *community_edges = (int64_t*)calloc(sizeof(int64_t), nc);
    int64_t *cache = (int64_t*)calloc(sizeof(int64_t), nc);
        
    assert ((out_.GetSize() >> 2) == N_e);
    assert ((node_degree_.GetSize() >> 3) == N_n);
    assert ((u_.GetSize() >> 3) == N_e);
    assert ((w_.GetSize() >> 2) == N_e);
    
    for (int64_t i=0; i<N_e; i++) {
        IdType u = uptr[i];
        IdType v = vptr[i];
        float  w = wptr[i];
        assert (u < N_n);
        assert (v < N_n);
        
        if (i%10000000 == 0) {
            printf("."); fflush(0);
        }

        if (node_assignments[u].size() == 0 && node_assignments[v].size() == 0)
        {
            int c = leastload(community_edges, nc);
            out[i] = c;
            assert(c < nc);
            community_edges[c] ++;
            community_weights[c] = community_weights[c] + w;
            node_assignments[u].push_back(c);
            if (u != v)
                node_assignments[v].push_back(c);

            if (node_assignments[u].size() > nc) {
                printf("assert 1");
                for (int k=0; k<node_assignments[u].size(); k++)
                    printf("%d ", node_assignments[u][k]);
                printf("\n");
            }
            
            assert(node_assignments[u].size() <= nc);
            assert(node_assignments[v].size() <= nc);                    
            edgenum_unassigned[u] --;
            edgenum_unassigned[v] --;
        }
        else if (node_assignments[u].size() != 0 && node_assignments[v].size() == 0)
        {
            for (uint32_t j=0; j<node_assignments[u].size(); j++) {
                int cind = node_assignments[u][j];
                cache[j] = community_edges[cind];
            }
            int cindex = leastload(cache, node_assignments[u].size());
            int c = node_assignments[u][cindex];
            assert(c < nc);
            out[i] = c;
            community_edges[c] ++;
            community_weights[c] = community_weights[c] + w;
            for (uint32_t j=0; j<node_assignments[v].size(); j++) {
                assert(node_assignments[v][j] != c);
            }
            
            node_assignments[v].push_back(c);
            assert(node_assignments[v].size() <= nc);
            edgenum_unassigned[u] --;
            edgenum_unassigned[v] --;
        }
        else if (node_assignments[v].size() != 0 && node_assignments[u].size() == 0)
        {
            for (uint32_t j=0; j<node_assignments[v].size(); j++) {
                int cind = node_assignments[v][j];
                cache[j] = community_edges[cind];
            }
            int cindex = leastload(cache, node_assignments[v].size());
            int c = node_assignments[v][cindex];
            assert(c < nc);
            out[i] = c;

            community_edges[c] ++;
            community_weights[c] = community_weights[c] + w;

            for (uint32_t j=0; j<node_assignments[u].size(); j++) {
                assert(node_assignments[u][j] != c);
            }            
            node_assignments[u].push_back(c);
            if (node_assignments[u].size() > nc) {
                printf("assert 2");
                for (int k=0; k<node_assignments[u].size(); k++)
                    printf("%d ", node_assignments[u][k]);
                printf("\n");
            }
            assert(node_assignments[u].size() <= nc);
            edgenum_unassigned[u] --;
            edgenum_unassigned[v] --;
        }
        else {
            std::vector<int> setv(nc), intersetv;
            for (int j=0; j<nc; j++) setv[j] = 0;
            int interset = 0;
            
            for (int j=0; j<node_assignments[v].size(); j++) {
                if(node_assignments[v][j] >= nc) {
                    printf("assert: %d %d\n", node_assignments[v][j], nc);
                    for (int l=0; l<node_assignments[v].size(); l++)
                        printf("%d ", node_assignments[v][l]);
                    printf("\n");
                }
                assert(node_assignments[v][j] < nc);
                setv[node_assignments[v][j]] ++;
            }
            for (int j=0; j<node_assignments[u].size(); j++) {
                if(node_assignments[u][j] >= nc) {
                    printf("%d %d\n", node_assignments[u][j], nc);
                    for (int l=0; l<node_assignments[u].size(); l++)
                        printf("%d ", node_assignments[u][l]);
                    printf("\n");
                }
                
                if (node_assignments[u].size() > nc) {
                    printf("assert 3");
                    for (int k=0; k<node_assignments[u].size(); k++)
                        printf("%d ", node_assignments[u][k]);
                    printf("\n");
                }
                
                assert(node_assignments[u][j] < nc);
                setv[node_assignments[u][j]] ++;
            }
            
            for (int j=0; j<nc; j++) {
                if (setv[j] > 2) {
                    for (int l=0; l<nc; l++) 
                        printf("%d ", setv[l]); printf("\n");
                    for (int l=0; l<node_assignments[u].size(); l++)
                        printf("%d ", node_assignments[u][l]);    printf("\n");
                    for (int l=0; l<node_assignments[v].size(); l++)
                        printf("%d ", node_assignments[v][l]);    printf("\n");
                }
                assert(setv[j] <= 2);
                if (setv[j] == 2) {
                    interset++;
                    intersetv.push_back(j);
                }
            }
            if (interset) {
                for (int32_t j=0; j<intersetv.size(); j++) {
                    int cind = intersetv[j];
                    cache[j] = community_edges[cind];                    
                }
                int cindex = leastload(cache, intersetv.size());
                int c = intersetv[cindex];
                assert(c < nc);
                out[i] = c;
                community_edges[c] ++;
                community_weights[c] = community_weights[c] + w;
                edgenum_unassigned[u] --;
                edgenum_unassigned[v] --;                                
            }
            #if 1
            else {
                if (node_degree[u] < node_degree[v])
                {
                    for (uint32_t j=0; j<node_assignments[u].size(); j++) {
                        int cind = node_assignments[u][j];
                        cache[j] = community_edges[cind];
                    }
                    int cindex = leastload(cache, node_assignments[u].size());
                    int c = node_assignments[u][cindex];
                    assert(c < nc);
                    out[i] = c;
                    community_edges[c] ++;
                    community_weights[c] = community_weights[c] + w;
                    for (uint32_t j=0; j<node_assignments[v].size(); j++) {
                        assert(node_assignments[v][j] != c);
                    }
                    node_assignments[v].push_back(c);
                    assert(node_assignments[v].size() <= nc);
                    replication_list.push_back(v);
                    edgenum_unassigned[u] --;
                    edgenum_unassigned[v] --;
                }
                else
                {
                    for (uint32_t j=0; j<node_assignments[v].size(); j++) {
                        int cind = node_assignments[v][j];
                        cache[j] = community_edges[cind];
                    }
                    int cindex = leastload(cache, node_assignments[v].size());
                    int c = node_assignments[v][cindex];
                    assert(c < nc);
                    out[i] = c;
                    community_edges[c] ++;
                    community_weights[c] = community_weights[c] + w;
                    for (uint32_t j=0; j<node_assignments[u].size(); j++) {
                        if (node_assignments[u][j] == c) printf("j: %d\n", j);
                        assert(node_assignments[u][j] != c);
                    }
                    if (u != v)
                        node_assignments[u].push_back(c);

                    if (node_assignments[u].size() > nc) {
                        printf("assert 4");
                        for (int k=0; k<node_assignments[u].size(); k++)
                            printf("%d ", node_assignments[u][k]);
                        printf("\n");
                    }                    
                    assert(node_assignments[u].size() <= nc);
                    replication_list.push_back(u);
                    edgenum_unassigned[u] --;
                    edgenum_unassigned[v] --;                                                        
                }
            }
            #endif
        }
        
    }
    free(cache);

    std::string lprefix = prefix;
    printf("Writing to files\n");
    for (int64_t c=0; c < nc; c++)
    {
        std::string str = lprefix + "/" + std::to_string(nc) + "Communities/community"+ std::to_string(c) +".txt";
        // printf("=> %d %s\n", c, str.c_str()); fflush(0);        
        FILE *fp = fopen(str.c_str(), "w");
        assert(fp != NULL);
        
        for (int64_t i=0; i<N_e; i++)
        {
            if (out[i] == c) {
                fprintf(fp, "%ld,%ld,%f\n", uptr[i], vptr[i], wptr[i]);
            }
        }
        fclose(fp);
    }


    std::string str = lprefix + "/"  + std::to_string(nc) + "Communities/replicationlist.csv";
    FILE *fp = fopen(str.c_str(), "w");
    assert(fp != NULL);
    
    std::string str_ = "# The Indices of Nodes that are replicated :: Header";
    fprintf(fp, "## The Indices of Nodes that are replicated :: Header");
    printf("Total replication: %ld\n", replication_list.size());

    for (uint64_t i=0; i<replication_list.size(); i++)
    {
        fprintf(fp, "%ld\n", replication_list[i]);
    }

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

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLLibra_VC")
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
                    vertex_cut<IdType, IdType2, DType>(nc, node_degree, edgenum_unassigned, community_weights, u, v, w, out, N, N_e, prefix);
                });
        });
        });
});



void libra2dgl_built_dict(
    NDArray a_,
    NDArray b_,
    NDArray indices_,
    NDArray ldt_key_,
    NDArray gdt_key_,
    NDArray gdt_value_,
    NDArray node_map_,
    NDArray offset_,
    int32_t nc_,
    int32_t c_,
    int64_t fsize_,
    NDArray hash_nodes_,
    std::string prefix
    )
{

    int32_t *indices = indices_.Ptr<int32_t>();
    int64_t *ldt_key = ldt_key_.Ptr<int64_t>();
    int32_t *gdt_key = gdt_key_.Ptr<int32_t>();
    int32_t *gdt_value  = gdt_value_.Ptr<int32_t>();   // 2D tensor
    int32_t *node_map  = node_map_.Ptr<int32_t>();
    int32_t *offset  = offset_.Ptr<int32_t>();
    int32_t *hash_nodes  = hash_nodes_.Ptr<int32_t>();
    
    int32_t width = nc_;
    // int32_t offset = offset_;
    int32_t c = c_;
    int64_t fsize = fsize_;
        
    int64_t *a = a_.Ptr<int64_t>();
    int64_t *b = b_.Ptr<int64_t>();

    int32_t N_n = indices_.GetSize() >> 2;
    int32_t num_nodes = ldt_key_.GetSize() >> 2;

    #pragma omp simd
    for (int i=0; i<N_n; i++) {
        indices[i] = -100;
        // gdt_key[i] = 0;
    }
    
    int32_t pos = 0;
    int64_t edge = 0;

    std::string lprefix = prefix;

    // std::string str = lprefix.c_str() + std::to_string(nc_) + "Communities/community" + std::to_string(c) + ".txt";
    std::string str = lprefix + "/community" + std::to_string(c) + ".txt";
    
    FILE *fp = fopen(str.c_str(), "r");
    assert(fp != NULL);

    // uint64_t tic = _rdtsc();
    while(!feof(fp) && edge < fsize) {
        int64_t u, v;
        float w;
        fscanf(fp, "%ld,%ld,%f\n", &u, &v, &w);
            
        if (indices[u] == -100)
        {
            ldt_key[pos] = u;
            // ldt_value[pos] = pos;
            assert(pos < num_nodes);
            indices[u] = pos++;
        }
        if (indices[v] == -100)
        {
            ldt_key[pos] = v;
            // ldt_value[pos] = pos;
            assert(pos < num_nodes);
            indices[v] = pos++;
        }
        a[edge] = indices[u];
        b[edge++] = indices[v];
    }
    assert(edge <= fsize);
    fclose(fp);
    hash_nodes[0] = pos;
    hash_nodes[1] = edge;

    // int64_t toc1 = _rdtsc();
    
    for (int i=0; i<pos; i++) {
        int64_t u = ldt_key[i];
        int v = indices[u];
            
        int *ind = &gdt_key[u];
        int *ptr = gdt_value + u*width;
        ptr[*ind] = offset[0] + v;
        (*ind)++;
        assert (v != -100);
        assert(*ind <= nc_);
    }
    node_map[c] = offset[0] +  pos;
    offset[0] += pos;
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLlibra2dgl_built_dict")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray a_        = args[0];
    NDArray b_        = args[1];
    NDArray indices_  = args[2];
    NDArray ldt_key_  = args[3];
    NDArray gdt_key_  = args[4];
    NDArray gdt_value_ = args[5];
    NDArray node_map_ = args[6];
    NDArray offset_   = args[7];
    int32_t nc_       = args[8];
    int32_t c_        = args[9];
    int64_t fsize_    = args[10];
    NDArray hash_nodes_   = args[11];
    std::string prefix_ = args[12];
    libra2dgl_built_dict(a_, b_, indices_, ldt_key_, gdt_key_,
                         gdt_value_, node_map_, offset_,
                         nc_, c_, fsize_, hash_nodes_, prefix_);
});

void libra2dgl_built_adj(
    NDArray feat_,
    NDArray gfeat_,
    NDArray adj_,
    NDArray inner_node_,
    NDArray ldt_key_,
    NDArray gdt_key_,
    NDArray gdt_value_,
    NDArray node_map_,
    NDArray lf_,
    NDArray lftensor_,        
    int32_t num_nodes,
    int32_t nc,
    int32_t c,
    int32_t feat_size,
    NDArray labels_ , 
    NDArray trainm_ ,
    NDArray testm_  ,
    NDArray valm_   ,
    NDArray glabels_,
    NDArray gtrainm_,
    NDArray gtestm_ ,
    NDArray gvalm_,
    int64_t feat_shape)
{
    float *feat = feat_.Ptr<float>();      // 2D tensor
    float *gfeat = gfeat_.Ptr<float>();      // 2D tensor
    int32_t *adj = adj_.Ptr<int32_t>();      // 2D tensor
    int32_t *inner_node = inner_node_.Ptr<int32_t>();
    int64_t *ldt_key = ldt_key_.Ptr<int64_t>();
    int32_t *gdt_key = gdt_key_.Ptr<int32_t>();
    int32_t *gdt_value  = gdt_value_.Ptr<int32_t>();   // 2D tensor
    int32_t *node_map  = node_map_.Ptr<int32_t>();
    int32_t *lf      = lf_.Ptr<int32_t>();
    int32_t *lftensor  = lftensor_.Ptr<int32_t>();    
    int width = nc - 1;

    #pragma omp parallel for
    for (int i=0; i<num_nodes; i++) {

        int64_t k = ldt_key[i];
        int64_t v = i;
        int ind = gdt_key[k];
        
        int *adj_ptr = adj + v*width;
        if(ind == 1)
        {
            for (int j=0; j<width; j++) adj_ptr[j] = -1;
            inner_node[i] = 1;
        }
        else {
            lf[i] = lftensor[k];
            int *ptr = gdt_value + k*nc;
            int pos = 0;
            assert(ind <= nc);
            int flg = 0;
            for (int j=0; j<ind; j++) {
                if (ptr[j] == lf[i]) flg = 1;
                if(c != node2partition<int32_t>(ptr[j], node_map, nc) )
                    adj_ptr[pos++] = ptr[j];
            }
            assert(flg == 1);
            if(pos != ind - 1) {
                printf("Error: pos: %d, ind: %d\n", pos, ind-1);
            }
            assert(pos == ind - 1);
            for (; pos < width; pos++) adj_ptr[pos] = -1;
            inner_node[i] = 0;
        }
        
    }
    
    /* gathering */
    #pragma omp parallel for
    for (int64_t i=0; i<num_nodes; i++) {
        int64_t k = ldt_key[i];
        assert(k >=0 && k < feat_shape);
        int64_t ind = i*feat_size;
        float *optr = gfeat + ind;
        float *iptr = feat + k*feat_size;

        for (int j=0; j<feat_size; j++)
            optr[j] = iptr[j];
    }

    float *labels = labels_.Ptr<float>();
    float *glabels = glabels_.Ptr<float>();
    bool *trainm = trainm_.Ptr<bool>();
    bool *gtrainm = gtrainm_.Ptr<bool>();
    bool *testm = testm_.Ptr<bool>();
    bool *gtestm = gtestm_.Ptr<bool>();
    bool *valm = valm_.Ptr<bool>();
    bool *gvalm = gvalm_.Ptr<bool>();
    #pragma omp parallel for
    for (int64_t i=0; i<num_nodes; i++) {
        int64_t k = ldt_key[i];
        assert(k >=0 && k < feat_shape);
        glabels[i] = labels[k];
        gtrainm[i] = trainm[k];
        gtestm[i] = testm[k];
        gvalm[i] = valm[k];        
    }

}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLlibra2dgl_built_adj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray feat_       = args[0];
    NDArray gfeat_      = args[1];
    NDArray adj_        = args[2];
    NDArray inner_node_ = args[3];
    NDArray ldt_key_    = args[4];
    NDArray gdt_key_    = args[5];
    NDArray gdt_value_  = args[6];
    NDArray node_map_   = args[7];
    NDArray lf_         = args[8];
    NDArray lftensor_   = args[9];
    int32_t num_nodes   = args[10];
    int32_t nc          = args[11];
    int32_t c           = args[12];
    int32_t feat_size = args[13];
    NDArray labels_   = args[14];
    NDArray trainm_   = args[15];
    NDArray testm_    = args[16];
    NDArray valm_     = args[17];
    NDArray glabels_  = args[18];
    NDArray gtrainm_  = args[19];
    NDArray gtestm_   = args[20];
    NDArray gvalm_    = args[21];
    int64_t feat_shape = args[22];
    
    libra2dgl_built_adj(feat_, gfeat_, adj_, inner_node_, 
                        ldt_key_, gdt_key_, gdt_value_,
                        node_map_, lf_, lftensor_,
                        num_nodes, nc, c, feat_size,
                        labels_, trainm_, testm_, valm_,
                        glabels_, gtrainm_, gtestm_, gvalm_,
                        feat_shape);
});


//************************************************************************************************
// Version II:
//************************************************************************************************
void libra2dgl_fix_lf(
    NDArray gdt_key_,
    NDArray gdt_value_,
    NDArray lftensor_,
    int32_t nc_,
    int64_t Nn)
{
    // uint64_t toc1 = _rdtsc();
    
    srand (time(NULL));
    int32_t *gdt_key    = gdt_key_.Ptr<int32_t>();
    int32_t *gdt_value  = gdt_value_.Ptr<int32_t>(); // 2D tensor
    int32_t *lftensor   = lftensor_.Ptr<int32_t>();
    
    int32_t width = nc_;
    int32_t cnt = 0;
    int32_t avg_split_copy = 0, scnt = 0;
    for (int64_t i=0; i<Nn; i++) {
        
        if (gdt_key[i] <= 0) {
            cnt ++;
        }
        else {
            int val = rand() % gdt_key[i];
            assert(val >= 0 && val < gdt_key[i]);
            assert(gdt_key[i] <= nc_);
            int *ptr = gdt_value + i*width;
            lftensor[i] = ptr[val];
        }
        if (gdt_key[i] > 1) {
            avg_split_copy += gdt_key[i];
            scnt ++;
        }
    }
    fflush(0);
}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLlibra2dgl_fix_lf")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray gdt_key_   = args[0];
    NDArray gdt_value_ = args[1];
    NDArray lftensor_  = args[2];
    int32_t nc_        = args[3];
    int64_t Nn         = args[4];

    libra2dgl_fix_lf(gdt_key_, gdt_value_, lftensor_, nc_, Nn);
});


void libra2dgl_built_adj_v2(
    NDArray feat_,
    NDArray gfeat_,
    NDArray adj_,
    NDArray inner_node_,
    NDArray ldt_key_,
    NDArray gdt_key_,
    NDArray gdt_value_,
    NDArray node_map_,
    NDArray lf_,
    NDArray lftensor_,
    int32_t num_nodes,
    int32_t nc,
    int32_t c,
    int32_t feat_size)
{
    float *feat = feat_.Ptr<float>();      // 2D tensor
    float *gfeat = gfeat_.Ptr<float>();      // 2D tensor
    int32_t *adj = adj_.Ptr<int32_t>();      // 2D tensor
    int32_t *inner_node = inner_node_.Ptr<int32_t>();
    int64_t *ldt_key = ldt_key_.Ptr<int64_t>();
    int32_t *gdt_key = gdt_key_.Ptr<int32_t>();
    int32_t *gdt_value  = gdt_value_.Ptr<int32_t>();   // 2D tensor
    int32_t *node_map  = node_map_.Ptr<int32_t>();
    int32_t *lf      = lf_.Ptr<int32_t>();
    int32_t *lftensor  = lftensor_.Ptr<int32_t>();
    int width = nc - 1;

    #pragma omp parallel for
    for (int i=0; i<num_nodes; i++) {

        int64_t k = ldt_key[i];
        int64_t v = i;
        int ind = gdt_key[k];
        
        int *adj_ptr = adj + v*width;
        if(ind == 1)
        {
            for (int j=0; j<width; j++) adj_ptr[j] = -1;
            inner_node[i] = 1;
            lf[i] = -200;
        }
        else {
            lf[i] = lftensor[k];
            int *ptr = gdt_value + k*nc;
            int pos = 0;
            assert(ind <= nc);
            int flg = 0;
            for (int j=0; j<ind; j++) {
                if (ptr[j] == lf[i]) flg = 1;
                if(c != node2partition<int32_t>(ptr[j], node_map, nc) )
                    adj_ptr[pos++] = ptr[j];
            }
            assert(flg == 1);
            if(pos != ind - 1)
                printf("pos: %d, ind: %d\n", pos, ind-1);
            assert(pos == ind - 1);
            for (; pos < width; pos++) adj_ptr[pos] = -1;
            inner_node[i] = 0;
        }
        
    }
    
    /* gathering */
    #pragma omp parallel for
    for (int64_t i=0; i<num_nodes; i++) {
        int64_t k = ldt_key[i];
        int64_t ind = i*feat_size;
        float *optr = gfeat + ind;
        float *iptr = feat + k*feat_size;

        for (int j=0; j<feat_size; j++)
            optr[j] = iptr[j];
    }

}

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLlibra2dgl_built_adj_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray feat_       = args[0];
    NDArray gfeat_      = args[1];
    NDArray adj_        = args[2];
    NDArray inner_node_ = args[3];
    NDArray ldt_key_    = args[4];
    NDArray gdt_key_    = args[5];
    NDArray gdt_value_  = args[6];
    NDArray node_map_   = args[7];
    NDArray lf_         = args[8];
    NDArray lftensor_   = args[9];
    int32_t num_nodes   = args[10];
    int32_t nc          = args[11];
    int32_t c           = args[12];
    int32_t feat_size   = args[13];
    
    libra2dgl_built_adj_v2(feat_, gfeat_, adj_, inner_node_,
                           ldt_key_, gdt_key_, gdt_value_,
                           node_map_, lf_, lftensor_, num_nodes, nc,
                           c, feat_size);
});



}  // namespace aten
}  // namespace dgl
