//
// Created by Allen Zhou on 2019/6/19.
//

#ifndef DGL_GRAPH_SERIALIZE_H
#define DGL_GRAPH_SERIALIZE_H

#include <dgl/graph.h>
#include <dgl/array.h>
#include <dgl/immutable_graph.h>
#include <dmlc/io.h>
#include <dgl/runtime/ndarray.h>

#include "../c_api_common.h"

using dgl::runtime::DGLArgs;
using dgl::runtime::DGLArgValue;
using dgl::runtime::DGLRetValue;
using dgl::runtime::PackedFunc;
using dgl::runtime::NDArray;
using dgl::ImmutableGraph;

namespace dgl {
namespace serialize {

typedef std::pair<std::string, NDArray> NamedTensor;

static const ImmutableGraph *ToImmutableGraph(const GraphInterface *g);

//std::vector<NamedTensor> ToNamedTensorList(void *pytensorlist, void *pynamelist, int list_size);

static bool SaveDGLGraphs(std::vector<const ImmutableGraph *> graph_list,
                          std::vector<NDArray> node_feats,
                          std::vector<NDArray> edge_feats,
                          std::vector<std::string> node_names,
                          std::vector<std::string> edge_names,
//                          NDArray label_list,
                          const std::string &filename);

static void ToNameAndTensorList(void *pytensorlist, void *pynamelist, int list_size,
                                std::vector<std::string> &name_listptr,
                                std::vector<NDArray> &tensor_listptr);

static bool LoadDGLGraphs(const std::string &filename,
                          std::vector<uint32_t> idx_list);

static std::vector<NamedTensor> ToNamedTensorList(void *pytensorlist, void *pynamelist, int list_size);

} // namespace serialize
} //namespace dgl

#endif //DGL_GRAPH_SERIALIZE_H
