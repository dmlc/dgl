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
typedef std::pair<std::string, NDArray> NamedTensor;

namespace dgl {
namespace serialize {

static const ImmutableGraph* ToImmutableGraph(const GraphInterface *g);

} // namespace serialize
} //namespace dgl

#endif //DGL_GRAPH_SERIALIZE_H
