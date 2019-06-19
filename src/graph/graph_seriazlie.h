//
// Created by Allen Zhou on 2019/6/19.
//

#ifndef DGL_GRAPH_SERIAZLIE_H
#define DGL_GRAPH_SERIAZLIE_H

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
typedef std::pair<std::string, dgl::runtime::NDArray> Feat_Dict;

namespace dgl {
namespace serialize {

static const ImmutableGraph* ToImmutableGraph(const GraphInterface *g);

static bool SaveDGLGraphs(std::string filename,
                          std::vector<ImmutableGraph> graph_list,
                          std::vector<>>

);


} // namespace serialize
} //namespace dgl

#endif //DGL_GRAPH_SERIAZLIE_H
