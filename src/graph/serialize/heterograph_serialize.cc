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
class HeteroGraphDataObject;
}
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::serialize::HeteroGraphDataObject, true);
}

namespace dgl {
namespace serialize {

using namespace dgl::runtime;
using dmlc::SeekStream;

class HeteroGraphDataObject : public runtime::Object {
 public:
  std::shared_ptr<HeteroGraph> gptr;
  std::vector<std::vector<NamedTensor>> node_tensors;
  std::vector<std::vector<NamedTensor>> edge_tensors;
  std::vector<std::string> etype_names;
  std::vector<std::string> ntype_names;

  static constexpr const char *_type_key =
    "heterograph_serialize.HeteroGraphData";

  HeteroGraphDataObject(HeteroGraphPtr gptr,
                        List<Map<std::string, Value>> ndata,
                        List<Map<std::string, Value>> edata,
                        List<Value> ntype_names, List<Value> etype_names) {
    this->gptr = std::dynamic_pointer_cast<HeteroGraph>(gptr);
    CHECK_NOTNULL(this->gptr);
    for (auto nd_dict : ndata) {
      node_tensors.emplace_back();
      for (auto kv : nd_dict) {
        auto last = &node_tensors.back();
        NDArray ndarray = kv.second->data;
        last->emplace_back(kv.first, ndarray);
      }
    }
    for (auto nd_dict : edata) {
      edge_tensors.emplace_back();
      for (auto kv : nd_dict) {
        auto last = &edge_tensors.back();
        NDArray ndarray = kv.second->data;
        last->emplace_back(kv.first, ndarray);
      }
    }
    for (auto v : ntype_names) {
      std::string name = v->data;
      this->ntype_names.push_back(name);
    }

    for (auto v : etype_names) {
      std::string name = v->data;
      this->etype_names.push_back(name);
    }
  }

  void Save(dmlc::Stream *fs) const {
    fs->Write(gptr);
    fs->Write(node_tensors);
    fs->Write(edge_tensors);
    fs->Write(ntype_names);
    fs->Write(etype_names);
  }

  bool Load(dmlc::Stream *fs) {
    fs->Read(&gptr);
    fs->Read(&node_tensors);
    fs->Read(&edge_tensors);
    fs->Read(&ntype_names);
    fs->Read(&etype_names);
    return true;
  }

  DGL_DECLARE_OBJECT_TYPE_INFO(HeteroGraphDataObject, runtime::Object);
};

class HeteroGraphData : public runtime::ObjectRef {
 public:
  DGL_DEFINE_OBJECT_REF_METHODS(HeteroGraphData, runtime::ObjectRef,
                                HeteroGraphDataObject);

  /*! \brief create a new GraphData reference */
  static HeteroGraphData Create(HeteroGraphPtr gptr,
                                List<Map<std::string, Value>> node_tensors,
                                List<Map<std::string, Value>> edge_tensors,
                                List<Value> ntype_names,
                                List<Value> etype_names) {
    return HeteroGraphData(std::make_shared<HeteroGraphDataObject>(
      gptr, node_tensors, edge_tensors, ntype_names, etype_names));
  }
};

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
    // Map<std::string, Value> labels = args[2];

    // std::vector<NamedTensor> labels_list;
    // for (auto kv : labels) {
    //   std::string name = kv.first;
    //   Value v = kv.second;
    //   NDArray ndarray = v->data;
    //   labels_list.emplace_back(name, ndarray);
    // }

    *rv = dgl::serialize::SaveHeteroGraphs(filename, hgdata);
  });

}  // namespace serialize
}  // namespace dgl
