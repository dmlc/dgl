/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/traversal.cc
 * \brief Graph traversal implementation
 */
#include <algorithm>
#include <stack>
#include "./traversal.h"
#include "../c_api_common.h"

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMArgValue;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;
using tvm::runtime::NDArray;

namespace dgl {
namespace traverse {
namespace {
/*!\brief A utility view class for a range of data in a vector */
template<typename DType>
struct VectorView {
  const std::vector<DType>* vec;
  size_t range_start, range_end;

  explicit VectorView(const std::vector<DType>* vec): vec(vec) {}

  auto begin() const -> decltype(vec->begin()) {
    return vec->begin() + range_start;
  }

  auto end() const -> decltype(vec->end()) {
    return vec->begin() + range_end;
  }

  size_t size() const { return range_end - range_start; }
};
}  // namespace

/*!
 * \brief Class for representing frontiers.
 *
 * Each frontier is a list of nodes/edges (specified by their ids).
 * An optional tag can be specified on each node/edge (represented by an int value).
 */
struct Frontiers {
  /*!\brief a vector store for the edges in all the fronties */
  std::vector<dgl_id_t> ids;

  /*!\brief a vector store for edge tags. The vector is empty is no tags. */
  std::vector<int64_t> tags;

  /*!\brief a section vector to indicate each frontier */
  std::vector<int64_t> sections;
};

Frontiers BFSNodesFrontiers(const Graph& graph, IdArray source, bool reversed) {
  Frontiers front;
  size_t i = 0;
  VectorView<dgl_id_t> front_view(&front.ids);
  auto visit = [&] (const dgl_id_t v) { front.ids.push_back(v); };
  auto make_frontier = [&] () {
      front_view.range_start = i;
      front_view.range_end = front.ids.size();
      if (front.ids.size() != i) {
        // do not push zero-length frontier
        front.sections.push_back(front.ids.size() - i);
      }
      i = front.ids.size();
      return front_view;
    };
  BFSNodes(graph, source, reversed, visit, make_frontier);
  return front;
}

TVM_REGISTER_GLOBAL("traversal._CAPI_DGLBFSNodes")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray src = IdArray::FromDLPack(CreateTmpDLManagedTensor(args[1]));
    bool reversed = args[2];
    const auto& front = BFSNodesFrontiers(*gptr, src, reversed);
    IdArray node_ids = CopyVectorToNDArray(front.ids);
    IdArray sections = CopyVectorToNDArray(front.sections);
    *rv = ConvertNDArrayVectorToPackedFunc({node_ids, sections});
  });

Frontiers TopologicalNodesFrontiers(const Graph& graph, bool reversed) {
  Frontiers front;
  size_t i = 0;
  VectorView<dgl_id_t> front_view(&front.ids);
  auto visit = [&] (const dgl_id_t v) { front.ids.push_back(v); };
  auto make_frontier = [&] () {
      front_view.range_start = i;
      front_view.range_end = front.ids.size();
      if (front.ids.size() != i) {
        // do not push zero-length frontier
        front.sections.push_back(front.ids.size() - i);
      }
      i = front.ids.size();
      return front_view;
    };
  TopologicalNodes(graph, reversed, visit, make_frontier);
  return front;
}

TVM_REGISTER_GLOBAL("traversal._CAPI_DGLTopologicalNodes")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    bool reversed = args[1];
    const auto& front = TopologicalNodesFrontiers(*gptr, reversed);
    IdArray node_ids = CopyVectorToNDArray(front.ids);
    IdArray sections = CopyVectorToNDArray(front.sections);
    *rv = ConvertNDArrayVectorToPackedFunc({node_ids, sections});
  });

//TVM_REGISTER_GLOBAL("graph_index._CAPI_DGLGraphDFSLabeledEdges")
//.set_body([] (TVMArgs args, TVMRetValue* rv) {
//    GraphHandle ghandle = args[0];
//    Graph* gptr = static_cast<Graph*>(ghandle);
//    IdArray src = args[1];
//    bool out = args[2];
//    bool reverse_edge = args[3];
//    bool nontree_edge = args[4];
//    auto ret = gptr->DFSLabeledEdges(src, out, reverse_edge, nontree_edge);
//    *rv = ConvertIdArrayQuadrupleToPackedFunc(ret);
//  });
//

//Frontiers DFSEdges(const Graph& graph,
//                   IdArray sources,
//                   bool reversed,
//                   bool has_reverse_edge,
//                   bool has_nontree_edge) {
//  Frontiers ret;
//  const auto next_iter = reversed? &Graph::PredVec : &Graph::SuccVec;
//  return ret;
//}

/*std::tuple<IdVector, IdVector, IdVector> Graph::DFSLabeledEdges_(
  dgl_id_t source, bool out, bool reverse_edge, bool nontree_edge) const {
  enum EdgeType { forward, reverse, nontree };
  IdVector src;
  IdVector dst;
  IdVector type;
  const auto &adj = out ? adjlist_ : reverse_adjlist_;
  typedef std::pair<dgl_id_t, std::vector<dgl_id_t>::const_iterator> crux_t;
  std::stack<crux_t> stack;
  stack.push(std::make_pair(source, adj[source].succ.begin()));
  std::vector<bool> visited(adj.size());
  visited[source] = true;
  while (!stack.empty()) {
    auto parent = stack.top().first;
    auto children = stack.top().second;
    if (children != adj[parent].succ.end()) {
      auto child = *children;
      ++stack.top().second;
      if (visited[child] && nontree_edge) {
        src.push_back(parent);
        dst.push_back(child);
        type.push_back(nontree);
      } else if (!visited[child]) {
        src.push_back(parent);
        dst.push_back(child);
        type.push_back(forward);
        visited[child] = true;
        stack.push(std::make_pair(child, adj[child].succ.begin()));
      }
    } else {
      stack.pop();
      if (!stack.empty() && reverse_edge) {
        src.push_back(stack.top().first);
        dst.push_back(parent);
        type.push_back(reverse);
      }
    }
  }
  return std::make_tuple(src, dst, type);
}*/

}  // namespace traverse
}  // namespace dgl
