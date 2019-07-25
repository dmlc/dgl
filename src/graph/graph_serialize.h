//
// Created by Allen Zhou on 2019/6/19.
//

#ifndef DGL_GRAPH_SERIALIZE_H
#define DGL_GRAPH_SERIALIZE_H

#include <dgl/graph.h>
#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"
#include <dgl/runtime/object.h>
#include <dgl/packed_func_ext.h>


using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;
using dgl::ImmutableGraph;
using namespace dgl::runtime;

namespace dgl {
namespace serialize {


typedef std::pair<std::string, NDArray> NamedTensor;

class GraphDataObject : public runtime::Object {

public:
  ImGraphPtr gptr;
  std::vector<NamedTensor> node_tensors;
  std::vector<NamedTensor> edge_tensors;
  static constexpr const char *_type_key = "graph_serialize.GraphData";

  void setData(ImGraphPtr gptr,
               Map<std::string, Value> node_tensors,
               Map<std::string, Value> edge_tensors) {
    this->gptr = gptr;

    for (auto kv: node_tensors) {
      std::string name = kv.first;
      Value v = kv.second;
      NDArray ndarray = static_cast<NDArray>(v->data);
      this->node_tensors.emplace_back(name, ndarray);
    }
    for (auto kv: edge_tensors) {
      std::string &name = kv.first;
      Value v = kv.second;
      const NDArray &ndarray = static_cast<NDArray>(v->data);
      this->edge_tensors.emplace_back(name, ndarray);
    }
  }

  static bool CheckND(const NDArray nd){
    LOG(INFO) << "111111";
    return (nd->dtype.lanes != 0);
  }

  void Save(dmlc::Stream *fs) const {
    LOG(INFO) << "Save111";
    const CSRPtr g_csr = this->gptr->GetInCSR();
    fs->Write(g_csr->indptr());
    CHECK(CheckND(g_csr->indptr())) << "Invalid Lanes";
    fs->Write(g_csr->indices());
    CHECK(CheckND(g_csr->indices())) << "Invalid Lanes";
    fs->Write(g_csr->edge_ids());
    CHECK(CheckND(g_csr->edge_ids())) << "Invalid Lanes";
    fs->Write(node_tensors);
    CHECK(CheckND(node_tensors[0].second)) << "Invalid Lanes";
    fs->Write(edge_tensors);
//    fs->Write(this->node_tensors.size());
//    fs->Write(this->edge_tensors.size());
//    for (const auto &node_tensorkv: this->node_tensors) {
//      std::string name = node_tensorkv.first;
//      NDArray tensor = node_tensorkv.second;
//      fs->Write(name);
//      fs->Write(tensor);
//    }
//    for (const auto &edge_tensorskv: this->node_tensors) {
//      std::string name = edge_tensorskv.first;
//      NDArray tensor = edge_tensorskv.second;
//      fs->Write(name);
//      fs->Write(tensor);
//    }
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


} // namespace serialize
} //namespace dgl

#endif //DGL_GRAPH_SERIALIZE_H
