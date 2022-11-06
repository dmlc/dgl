/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/serialize/heterograph_data.h
 * @brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_HETEROGRAPH_DATA_H_
#define DGL_GRAPH_SERIALIZE_HETEROGRAPH_DATA_H_

#include <dgl/array.h>
#include <dgl/graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../c_api_common.h"
#include "../heterograph.h"

using dgl::runtime::NDArray;
using namespace dgl::runtime;

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;
class HeteroGraphDataObject : public runtime::Object {
 public:
  std::shared_ptr<HeteroGraph> gptr;
  std::vector<std::vector<NamedTensor>> node_tensors;
  std::vector<std::vector<NamedTensor>> edge_tensors;
  std::vector<std::string> etype_names;
  std::vector<std::string> ntype_names;

  static constexpr const char *_type_key =
      "heterograph_serialize.HeteroGraphData";

  HeteroGraphDataObject() {}

  HeteroGraphDataObject(
      HeteroGraphPtr gptr, List<Map<std::string, Value>> ndata,
      List<Map<std::string, Value>> edata, List<Value> ntype_names,
      List<Value> etype_names) {
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

    this->ntype_names = ListValueToVector<std::string>(ntype_names);
    this->etype_names = ListValueToVector<std::string>(etype_names);
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
  DGL_DEFINE_OBJECT_REF_METHODS(
      HeteroGraphData, runtime::ObjectRef, HeteroGraphDataObject);

  /** @brief create a new GraphData reference */
  static HeteroGraphData Create(
      HeteroGraphPtr gptr, List<Map<std::string, Value>> node_tensors,
      List<Map<std::string, Value>> edge_tensors, List<Value> ntype_names,
      List<Value> etype_names) {
    return HeteroGraphData(std::make_shared<HeteroGraphDataObject>(
        gptr, node_tensors, edge_tensors, ntype_names, etype_names));
  }

  /** @brief create an empty GraphData reference */
  static HeteroGraphData Create() {
    return HeteroGraphData(std::make_shared<HeteroGraphDataObject>());
  }
};
}  // namespace serialize
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::serialize::HeteroGraphDataObject, true);
}

#endif  // DGL_GRAPH_SERIALIZE_HETEROGRAPH_DATA_H_
