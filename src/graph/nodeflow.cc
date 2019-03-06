/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/nodeflow.cc
 * \brief DGL NodeFlow related functions.
 */

#include <dgl/immutable_graph.h>
#include <dgl/nodeflow.h>

#include <string.h>

#include "../c_api_common.h"

namespace dgl {

std::vector<IdArray> GetNodeFlowSlice(const ImmutableGraph &graph, const std::string &fmt,
                                      size_t layer0_size, size_t layer1_start,
                                      size_t layer1_end, bool remap) {
  CHECK_GE(layer1_start, layer0_size);
  if (fmt == "csr") {
    dgl_id_t first_vid = layer1_start - layer0_size;
    ImmutableGraph::CSRArray arrs = graph.GetInCSRArray(layer1_start, layer1_end);
    if (remap) {
      dgl_id_t *indices_data = static_cast<dgl_id_t*>(arrs.indices->data);
      dgl_id_t *eid_data = static_cast<dgl_id_t*>(arrs.id->data);
      const size_t len = arrs.indices->shape[0];
      dgl_id_t first_eid = eid_data[0];
      for (size_t i = 0; i < len; i++) {
        CHECK_GE(indices_data[i], first_vid);
        indices_data[i] -= first_vid;
        CHECK_GE(eid_data[i], first_eid);
        eid_data[i] -= first_eid;
      }
    }
    return std::vector<IdArray>{arrs.indptr, arrs.indices, arrs.id};
  } else if (fmt == "coo") {
    ImmutableGraph::CSR::Ptr csr = graph.GetInCSR();
    int64_t nnz = csr->indptr[layer1_end] - csr->indptr[layer1_start];
    IdArray idx = IdArray::Empty({2 * nnz}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    IdArray eid = IdArray::Empty({nnz}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    int64_t *idx_data = static_cast<int64_t*>(idx->data);
    dgl_id_t *eid_data = static_cast<dgl_id_t*>(eid->data);
    size_t num_edges = 0;
    for (size_t i = layer1_start; i < layer1_end; i++) {
      for (int64_t j = csr->indptr[i]; j < csr->indptr[i + 1]; j++) {
        // These nodes are all in a layer. We need to remap them to the node id
        // local to the layer.
        idx_data[num_edges] = remap ? i - layer1_start : i;
        num_edges++;
      }
    }
    CHECK_EQ(num_edges, nnz);
    if (remap) {
      size_t edge_start = csr->indptr[layer1_start];
      dgl_id_t first_eid = csr->edge_ids[edge_start];
      dgl_id_t first_vid = layer1_start - layer0_size;
      for (int64_t i = 0; i < nnz; i++) {
        CHECK_GE(csr->indices[edge_start + i], first_vid);
        idx_data[nnz + i] = csr->indices[edge_start + i] - first_vid;
        eid_data[i] = csr->edge_ids[edge_start + i] - first_eid;
      }
    } else {
      std::copy(csr->indices.begin() + csr->indptr[layer1_start],
                csr->indices.begin() + csr->indptr[layer1_end], idx_data + nnz);
      std::copy(csr->edge_ids.begin() + csr->indptr[layer1_start],
                csr->edge_ids.begin() + csr->indptr[layer1_end], eid_data);
    }
    return std::vector<IdArray>{idx, eid};
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format";
    return std::vector<IdArray>();
  }
}

}  // namespace dgl
