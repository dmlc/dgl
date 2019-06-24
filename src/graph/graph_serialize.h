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


struct DGLGraphSerialize {
  GraphHandle g_handle;
  uint32_t num_node_feats;
  uint32_t num_edge_feats;
  const char **node_names;
  NDArray **node_feats;
  const char **edge_names;
  NDArray **edge_feats;
};

static const ImmutableGraph *ToImmutableGraph(const GraphInterface *g);

bool SaveDGLGraphs(std::string filename, uint32_t num_graph, const DGLGraphSerialize *gstructs);

std::vector<DGLGraphSerialize> LoadDGLGraphs(const std::string &filename,
                                             std::vector<uint32_t> idx_list);

} // namespace serialize
} //namespace dgl

#endif //DGL_GRAPH_SERIALIZE_H
