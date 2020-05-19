/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/graph_serialize.cc
 * \brief Graph serialization implementation
 *
 * The storage structure is
 * {
 *   // MetaData Section
 *   uint64_t kDGLSerializeMagic
 *   uint64_t kVersion
 *   uint64_t GraphType
 *   ** Reserved Area till 4kB **
 *
 *   dgl_id_t num_graphs
 *   vector<dgl_id_t> graph_indices (start address of each graph)
 *   vector<dgl_id_t> nodes_num_list (list of number of nodes for each graph)
 *   vector<dgl_id_t> edges_num_list (list of number of edges for each graph)
 *
 *   vector<GraphData> graph_datas;
 *
 * }
 *
 * Storage of GraphData is
 * {
 *   // Everything uses in csr
 *   NDArray indptr
 *   NDArray indices
 *   NDArray edge_ids
 *   vector<pair<string, NDArray>> node_tensors;
 *   vector<pair<string, NDArray>> edge_tensors;
 * }
 *
 */
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../heterograph.h"
#include "./graph_serialize.h"

namespace dgl {
namespace serialize {

using namespace dgl::runtime;
using dmlc::SeekStream;

bool SaveHeteroGraphs(std::string filename, List<HeteroGraphData> hdata) {
  auto *fs =
    dynamic_cast<SeekStream *>(SeekStream::Create(filename.c_str(), "w", true));
  CHECK(fs) << "File name is not a valid local file name";

  // Write DGL MetaData
  const uint64_t kVersion = 2;
  fs->Write(kDGLSerializeMagic);
  fs->Write(kVersion);
  fs->Write(GraphType::kHeteroGraph);
  fs->Seek(4096);

  // Write Graph Meta Data
  uint64_t num_graph = hdata.size();

  std::vector<uint64_t> graph_indices(num_graph);

  fs->Write(num_graph);
  uint64_t indices_start_ptr = fs->Tell();
  fs->Write(graph_indices);
  // fs->Write(labels_list);

  // Write HeteroGraphData
  for (uint64_t i = 0; i < num_graph; ++i) {
    graph_indices[i] = fs->Tell();
    HeteroGraphDataObject gdata = *hdata[i].as<HeteroGraphDataObject>();
    fs->Write(gdata);
  }

  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);

  delete fs;
  return true;
}

StorageMetaData LoadHeteroGraphs(dmlc::SeekStream *fs,
                                 std::vector<dgl_id_t> idx_list) {
  uint64_t num_graph;
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  std::vector<uint64_t> graph_indices(num_graph);
  CHECK(fs->Read(&graph_indices)) << "Invalid graph indices";

  StorageMetaData metadata = StorageMetaData::Create();

  std::vector<HeteroGraphData> gdata_refs;
  if (idx_list.empty()) {
    // Read All Graphs
    gdata_refs.reserve(num_graph);
    for (uint64_t i = 0; i < num_graph; ++i) {
      HeteroGraphData gdata = HeteroGraphData::Create();
      auto hetero_data = gdata.sptr();
      fs->Read(&hetero_data);
      gdata_refs.push_back(gdata);
    }
  } else {
    // Read Selected Graphss
    gdata_refs.reserve(idx_list.size());
    // Would be better if idx_list is sorted. However the returned the graphs
    // should be the same order as the idx_list
    for (uint64_t i = 0; i < idx_list.size(); ++i) {
      fs->Seek(graph_indices[idx_list[i]]);
      HeteroGraphData gdata = HeteroGraphData::Create();
      auto hetero_data = gdata.sptr();
      fs->Read(&hetero_data);
      gdata_refs.push_back(gdata);
    }
  }

  metadata->SetHeteroGraphData(gdata_refs);

  delete fs;
  return metadata;
}

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_MakeHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    List<Map<std::string, Value>> edata = args[1];
    List<Map<std::string, Value>> ndata = args[2];
    List<Value> ntype_names = args[3];
    List<Value> etype_names = args[4];
    *rv = HeteroGraphData::Create(hg.sptr(), ndata, edata, ntype_names,
                                  etype_names);
  });

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_SaveHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    List<HeteroGraphData> hgdata = args[1];
    *rv = dgl::serialize::SaveHeteroGraphs(filename, hgdata);
  });

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_GetGindexFromHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphData hdata = args[0];
    *rv = HeteroGraphRef(hdata->gptr);
  });

void StorageMetaDataObject::SetHeteroGraphData(
  std::vector<HeteroGraphData> gdata) {
  this->heterograph_data = List<HeteroGraphData>(gdata);
}

}  // namespace serialize
}  // namespace dgl
