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
#include "graph_serialize.h"
#include "dmlc/logging.h"

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

using namespace dgl::runtime;

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dgl::runtime::NDArray;
using dgl::serialize::GraphData;
using dgl::serialize::GraphDataObject;
using dmlc::SeekStream;
using dmlc::Stream;
using std::vector;

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, GraphDataObject, true);
}

namespace dgl {
namespace serialize {

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_MakeGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    GraphRef gptr = args[0];
    ImmutableGraphPtr imGPtr = ToImmutableGraph(gptr.sptr());
    Map<std::string, Value> node_tensors = args[1];
    Map<std::string, Value> edge_tensors = args[2];
    GraphData gd = GraphData::Create();
    gd->SetData(imGPtr, node_tensors, edge_tensors);
    *rv = gd;
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_DGLSaveGraphs")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    List<GraphData> graph_data = args[1];
    Map<std::string, Value> labels = args[2];
    std::vector<NamedTensor> labels_list;
    for (auto kv : labels) {
      std::string name = kv.first;
      Value v = kv.second;
      NDArray ndarray = static_cast<NDArray>(v->data);
      labels_list.emplace_back(name, ndarray);
    }
    SaveDGLGraphs(filename, graph_data, labels_list);
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadDGLGraphFiles")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    List<Value> idxs = args[1];
    bool onlyMeta = args[2];
    std::vector<dgl_id_t> idx_list(idxs.size());
    for (uint64_t i = 0; i < idxs.size(); ++i) {
      idx_list[i] = static_cast<dgl_id_t>(idxs[i]->data);
    }
    *rv = LoadDGLGraphFiles(filename, idx_list, onlyMeta);
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GDataGraphHandle")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    GraphData gdata = args[0];
    *rv = gdata->gptr;
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GDataNodeTensors")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    GraphData gdata = args[0];
    Map<std::string, Value> rvmap;
    for (auto kv : gdata->node_tensors) {
      rvmap.Set(kv.first, Value(MakeValue(kv.second)));
    }
    *rv = rvmap;
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GDataEdgeTensors")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    GraphData gdata = args[0];
    Map<std::string, Value> rvmap;
    for (auto kv : gdata->edge_tensors) {
      rvmap.Set(kv.first, Value(MakeValue(kv.second)));
    }
    *rv = rvmap;
  });

uint64_t GetFileVersion(const std::string &filename) {
  auto fs = Stream::Create(filename.c_str(), "r", true);
  CHECK(fs) << "File " << filename << " not found";
  uint64_t magicNum, graphType, version;
  fs->Read(&magicNum);
  fs->Read(&graphType);
  fs->Read(&version);
  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  return version;
}

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_GetFileVersion")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    *rv = static_cast<int64_t>(GetFileVersion(filename));
  });

StorageMetaData LoadGraph_V0(const std::string &filename,
                             std::vector<dgl_id_t> idx_list, bool onlyMeta) {
                               
                             }

StorageMetaData LoadDGLGraphFiles(const std::string &filename,
                                  std::vector<dgl_id_t> idx_list,
                                  bool onlyMeta) {
  auto fs = std::unique_ptr<SeekStream>(dynamic_cast<SeekStream *>(
    SeekStream::CreateForRead(filename.c_str(), true)));
  CHECK(fs) << "Filename is invalid";
  // Read DGL MetaData
  uint64_t magicNum, graphType, version;
  fs->Read(&magicNum);
  fs->Read(&version);
  fs->Read(&graphType);
  fs->Seek(4096);

  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  StorageMetaData ret;
  if (version == 1 && graphType == kImmutableGraph) {
    ret = LoadDGLGraphs(fs, idx_list, onlyMeta);
  } else if (version == 2 && graphType == kHeteroGraph) {
    ret = LoadHeteroGraphs(fs, idx_list);
  } else {
    LOG(FATAL) << "Invalid DGL graph file";
  }

  return ret;
}

}  // namespace serialize
}  // namespace dgl
