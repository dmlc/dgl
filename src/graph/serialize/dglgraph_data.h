/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/dglgraph_data.h
 * \brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_DGLGRAPH_DATA_H_
#define DGL_GRAPH_SERIALIZE_DGLGRAPH_DATA_H_

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
#include <memory>
#include "../../c_api_common.h"

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


ImmutableGraphPtr ToImmutableGraph(GraphPtr g);

}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZE_DGLGRAPH_DATA_H_
