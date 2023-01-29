/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/nodeflow.cc
 * @brief DGL NodeFlow related functions.
 */

#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>
#include <dgl/packed_func_ext.h>

#include <string>

#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;

namespace dgl {

std::vector<IdArray> GetNodeFlowSlice(
    const ImmutableGraph &graph, const std::string &fmt, size_t layer0_size,
    size_t layer1_start, size_t layer1_end, bool remap) {
  CHECK_GE(layer1_start, layer0_size);
  if (fmt == std::string("csr")) {
    dgl_id_t first_vid = layer1_start - layer0_size;
    auto csr = aten::CSRSliceRows(
        graph.GetInCSR()->ToCSRMatrix(), layer1_start, layer1_end);
    if (remap) {
      dgl_id_t *eid_data = static_cast<dgl_id_t *>(csr.data->data);
      const dgl_id_t first_eid = eid_data[0];
      IdArray new_indices = aten::Sub(csr.indices, first_vid);
      IdArray new_data = aten::Sub(csr.data, first_eid);
      return {csr.indptr, new_indices, new_data};
    } else {
      return {csr.indptr, csr.indices, csr.data};
    }
  } else if (fmt == std::string("coo")) {
    auto csr = graph.GetInCSR()->ToCSRMatrix();
    const dgl_id_t *indptr = static_cast<dgl_id_t *>(csr.indptr->data);
    const dgl_id_t *indices = static_cast<dgl_id_t *>(csr.indices->data);
    const dgl_id_t *edge_ids = static_cast<dgl_id_t *>(csr.data->data);
    int64_t nnz = indptr[layer1_end] - indptr[layer1_start];
    IdArray idx = aten::NewIdArray(2 * nnz);
    IdArray eid = aten::NewIdArray(nnz);
    int64_t *idx_data = static_cast<int64_t *>(idx->data);
    dgl_id_t *eid_data = static_cast<dgl_id_t *>(eid->data);
    size_t num_edges = 0;
    for (size_t i = layer1_start; i < layer1_end; i++) {
      for (dgl_id_t j = indptr[i]; j < indptr[i + 1]; j++) {
        // These nodes are all in a layer. We need to remap them to the node id
        // local to the layer.
        idx_data[num_edges] = remap ? i - layer1_start : i;
        num_edges++;
      }
    }
    CHECK_EQ(num_edges, nnz);
    if (remap) {
      size_t edge_start = indptr[layer1_start];
      dgl_id_t first_eid = edge_ids[edge_start];
      dgl_id_t first_vid = layer1_start - layer0_size;
      for (int64_t i = 0; i < nnz; i++) {
        CHECK_GE(indices[edge_start + i], first_vid);
        idx_data[nnz + i] = indices[edge_start + i] - first_vid;
        eid_data[i] = edge_ids[edge_start + i] - first_eid;
      }
    } else {
      std::copy(
          indices + indptr[layer1_start], indices + indptr[layer1_end],
          idx_data + nnz);
      std::copy(
          edge_ids + indptr[layer1_start], edge_ids + indptr[layer1_end],
          eid_data);
    }
    return std::vector<IdArray>{idx, eid};
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format";
    return {};
  }
}

DGL_REGISTER_GLOBAL("_deprecate.nodeflow._CAPI_NodeFlowGetBlockAdj")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphRef g = args[0];
      std::string format = args[1];
      int64_t layer0_size = args[2];
      int64_t start = args[3];
      int64_t end = args[4];
      const bool remap = args[5];
      auto ig =
          CHECK_NOTNULL(std::dynamic_pointer_cast<ImmutableGraph>(g.sptr()));
      auto res = GetNodeFlowSlice(*ig, format, layer0_size, start, end, remap);
      *rv = ConvertNDArrayVectorToPackedFunc(res);
    });

}  // namespace dgl
