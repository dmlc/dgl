/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/graph_serialize.h
 * \brief Graph serialization header
 */
#ifndef DGL_GRAPH_GRAPH_SERIALIZE_H_
#define DGL_GRAPH_GRAPH_SERIALIZE_H_

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
#include "../c_api_common.h"

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

  void SetData(ImmutableGraphPtr gptr,
               Map<std::string, Value> node_tensors,
               Map<std::string, Value> edge_tensors);

  void Save(dmlc::Stream *fs) const;

  bool Load(dmlc::Stream *fs);

  DGL_DECLARE_OBJECT_TYPE_INFO(GraphDataObject, runtime::Object);
};


class GraphData : public runtime::ObjectRef {
 public:
  DGL_DEFINE_OBJECT_REF_METHODS(GraphData, runtime::ObjectRef, GraphDataObject);

  /*! \brief create a new GraphData reference */
  static GraphData Create() {
    return GraphData(std::make_shared<GraphDataObject>());
  }
};

class StorageMetaDataObject : public runtime::Object {
 public:
  dgl_id_t num_graph;
  Value nodes_num_list;
  Value edges_num_list;
  Map<std::string, Value> labels_list;
  List<GraphData> graph_data;
  static constexpr const char *_type_key = "graph_serialize.StorageMetaData";

  void SetMetaData(dgl_id_t num_graph,
                   std::vector<int64_t> nodes_num_list,
                   std::vector<int64_t> edges_num_list,
                   std::vector<NamedTensor> labels_list);

  void SetGraphData(std::vector<GraphData> gdata);

  void VisitAttrs(AttrVisitor *v) final {
    v->Visit("num_graph", &num_graph);
    v->Visit("nodes_num_list", &nodes_num_list);
    v->Visit("edges_num_list", &edges_num_list);
    v->Visit("labels", &labels_list);
    v->Visit("graph_data", &graph_data);
  }

  DGL_DECLARE_OBJECT_TYPE_INFO(StorageMetaDataObject, runtime::Object);
};


class StorageMetaData : public runtime::ObjectRef {
 public:
  DGL_DEFINE_OBJECT_REF_METHODS(StorageMetaData, runtime::ObjectRef, StorageMetaDataObject);

  /*! \brief create a new StorageMetaData reference */
  static StorageMetaData Create() {
    return StorageMetaData(std::make_shared<StorageMetaDataObject>());
  }
};


bool SaveDGLGraphs(std::string filename,
                   List<GraphData> graph_data,
                   std::vector<NamedTensor> labels_list);

StorageMetaData LoadDGLGraphs(const std::string &filename,
                              std::vector<dgl_id_t> idx_list,
                              bool onlyMeta = false);

ImmutableGraphPtr ToImmutableGraph(GraphPtr g);

}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_GRAPH_SERIALIZE_H_
