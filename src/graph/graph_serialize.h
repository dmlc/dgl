/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/graph_serialize.h
 * \brief Graph serialization header
 */
#ifndef DGL_GRAPH_GRAPH_SERIALIZE_H_
#define DGL_GRAPH_GRAPH_SERIALIZE_H_

#include "../c_api_common.h"
#include <dgl/graph.h>
#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/packed_func_ext.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

using dgl::runtime::NDArray;
using dgl::ImmutableGraph;
using namespace dgl::runtime;

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;

class GraphDataObject : public runtime::Object {
 public:
  ImmutableGraphPtr gptr;
  std::vector<NamedTensor> node_tensors;
  std::vector<NamedTensor> edge_tensors;
  static constexpr const char *_type_key = "graph_serialize.GraphData";

  void setData(ImmutableGraphPtr gptr,
               Map<std::string, Value> node_tensors,
               Map<std::string, Value> edge_tensors) {
    this->gptr = gptr;

    for (auto kv : node_tensors) {
      std::string name = kv.first;
      Value v = kv.second;
      NDArray ndarray = static_cast<NDArray>(v->data);
      this->node_tensors.emplace_back(name, ndarray);
    }
    for (auto kv : edge_tensors) {
      std::string &name = kv.first;
      Value v = kv.second;
      const NDArray &ndarray = static_cast<NDArray>(v->data);
      this->edge_tensors.emplace_back(name, ndarray);
    }
  }

  void Save(dmlc::Stream *fs) const {
    const CSRPtr g_csr = this->gptr->GetInCSR();
    fs->Write(g_csr->indptr());
    fs->Write(g_csr->indices());
    fs->Write(g_csr->edge_ids());
    fs->Write(node_tensors);
    fs->Write(edge_tensors);
  }

  bool Load(dmlc::Stream *fs) {
    NDArray indptr, indices, edge_ids;
    fs->Read(&indptr);
    fs->Read(&indices);
    fs->Read(&edge_ids);
    this->gptr = ImmutableGraph::CreateFromCSR(indptr, indices, edge_ids, "in");

    fs->Read(&this->node_tensors);
    fs->Read(&this->edge_tensors);
    return true;
  }

  DGL_DECLARE_OBJECT_TYPE_INFO(GraphDataObject, runtime::Object);
};


class GraphData : public runtime::ObjectRef {
 public:
  GraphData() {}
  explicit GraphData(std::shared_ptr<runtime::Object> obj) : runtime::ObjectRef(obj) {}
  const GraphDataObject *operator->() const {
    return static_cast<const GraphDataObject *>(obj_.get());
  }
  GraphDataObject *operator->() {
    return static_cast<GraphDataObject *>(obj_.get());
  }
  using ContainerType = GraphDataObject;
  /*! \brief create a new GraphData reference */
  static GraphData Create() {
    return GraphData(std::make_shared<GraphDataObject>());
  }
};

bool SaveDGLGraphs(std::string filename,
                   List<GraphData> graph_data);

std::vector<GraphData> LoadDGLGraphs(const std::string &filename,
                                     std::vector<dgl_id_t> idx_list);

ImmutableGraphPtr ToImmutableGraph(GraphPtr g);

}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_GRAPH_SERIALIZE_H_
