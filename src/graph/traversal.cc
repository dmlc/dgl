/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/traversal.cc
 * @brief Graph traversal implementation
 */
#include "./traversal.h"

#include <dgl/packed_func_ext.h>

#include <algorithm>
#include <queue>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace traverse {
namespace {
// A utility view class to wrap a vector into a queue.
template <typename DType>
struct VectorQueueWrapper {
  std::vector<DType>* vec;
  size_t head = 0;

  explicit VectorQueueWrapper(std::vector<DType>* vec) : vec(vec) {}

  void push(const DType& elem) { vec->push_back(elem); }

  DType top() const { return vec->operator[](head); }

  void pop() { ++head; }

  bool empty() const { return head == vec->size(); }

  size_t size() const { return vec->size() - head; }
};

// Internal function to merge multiple traversal traces into one ndarray.
// It is similar to zip the vectors together.
template <typename DType>
IdArray MergeMultipleTraversals(const std::vector<std::vector<DType>>& traces) {
  int64_t max_len = 0, total_len = 0;
  for (size_t i = 0; i < traces.size(); ++i) {
    const int64_t tracelen = traces[i].size();
    max_len = std::max(max_len, tracelen);
    total_len += traces[i].size();
  }
  IdArray ret = IdArray::Empty(
      {total_len}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  int64_t* ret_data = static_cast<int64_t*>(ret->data);
  for (int64_t i = 0; i < max_len; ++i) {
    for (size_t j = 0; j < traces.size(); ++j) {
      const int64_t tracelen = traces[j].size();
      if (i >= tracelen) {
        continue;
      }
      *(ret_data++) = traces[j][i];
    }
  }
  return ret;
}

// Internal function to compute sections if multiple traversal traces
// are merged into one ndarray.
template <typename DType>
IdArray ComputeMergedSections(const std::vector<std::vector<DType>>& traces) {
  int64_t max_len = 0;
  for (size_t i = 0; i < traces.size(); ++i) {
    const int64_t tracelen = traces[i].size();
    max_len = std::max(max_len, tracelen);
  }
  IdArray ret = IdArray::Empty(
      {max_len}, DGLDataType{kDGLInt, 64, 1}, DGLContext{kDGLCPU, 0});
  int64_t* ret_data = static_cast<int64_t*>(ret->data);
  for (int64_t i = 0; i < max_len; ++i) {
    int64_t sec_len = 0;
    for (size_t j = 0; j < traces.size(); ++j) {
      const int64_t tracelen = traces[j].size();
      if (i < tracelen) {
        ++sec_len;
      }
    }
    *(ret_data++) = sec_len;
  }
  return ret;
}

}  // namespace

/**
 * @brief Class for representing frontiers.
 *
 * Each frontier is a list of nodes/edges (specified by their ids).
 * An optional tag can be specified on each node/edge (represented by an int
 * value).
 */
struct Frontiers {
  /** @brief a vector store for the nodes/edges in all the frontiers */
  std::vector<dgl_id_t> ids;

  /**
   * @brief a vector store for node/edge tags. Empty if no tags are requested
   */
  std::vector<int64_t> tags;

  /** @brief a section vector to indicate each frontier */
  std::vector<int64_t> sections;
};

Frontiers BFSNodesFrontiers(
    const GraphInterface& graph, IdArray source, bool reversed) {
  Frontiers front;
  VectorQueueWrapper<dgl_id_t> queue(&front.ids);
  auto visit = [&](const dgl_id_t v) {};
  auto make_frontier = [&]() {
    if (!queue.empty()) {
      // do not push zero-length frontier
      front.sections.push_back(queue.size());
    }
  };
  BFSNodes(graph, source, reversed, &queue, visit, make_frontier);
  return front;
}

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray src = args[1];
      bool reversed = args[2];
      const auto& front = BFSNodesFrontiers(*(g.sptr()), src, reversed);
      IdArray node_ids = CopyVectorToNDArray<int64_t>(front.ids);
      IdArray sections = CopyVectorToNDArray<int64_t>(front.sections);
      *rv = ConvertNDArrayVectorToPackedFunc({node_ids, sections});
    });

Frontiers BFSEdgesFrontiers(
    const GraphInterface& graph, IdArray source, bool reversed) {
  Frontiers front;
  // NOTE: std::queue has no top() method.
  std::vector<dgl_id_t> nodes;
  VectorQueueWrapper<dgl_id_t> queue(&nodes);
  auto visit = [&](const dgl_id_t e) { front.ids.push_back(e); };
  bool first_frontier = true;
  auto make_frontier = [&] {
    if (first_frontier) {
      first_frontier = false;  // do not push the first section when doing edges
    } else if (!queue.empty()) {
      // do not push zero-length frontier
      front.sections.push_back(queue.size());
    }
  };
  BFSEdges(graph, source, reversed, &queue, visit, make_frontier);
  return front;
}

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray src = args[1];
      bool reversed = args[2];
      const auto& front = BFSEdgesFrontiers(*(g.sptr()), src, reversed);
      IdArray edge_ids = CopyVectorToNDArray<int64_t>(front.ids);
      IdArray sections = CopyVectorToNDArray<int64_t>(front.sections);
      *rv = ConvertNDArrayVectorToPackedFunc({edge_ids, sections});
    });

Frontiers TopologicalNodesFrontiers(
    const GraphInterface& graph, bool reversed) {
  Frontiers front;
  VectorQueueWrapper<dgl_id_t> queue(&front.ids);
  auto visit = [&](const dgl_id_t v) {};
  auto make_frontier = [&]() {
    if (!queue.empty()) {
      // do not push zero-length frontier
      front.sections.push_back(queue.size());
    }
  };
  TopologicalNodes(graph, reversed, &queue, visit, make_frontier);
  return front;
}

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLTopologicalNodes")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      bool reversed = args[1];
      const auto& front = TopologicalNodesFrontiers(*g.sptr(), reversed);
      IdArray node_ids = CopyVectorToNDArray<int64_t>(front.ids);
      IdArray sections = CopyVectorToNDArray<int64_t>(front.sections);
      *rv = ConvertNDArrayVectorToPackedFunc({node_ids, sections});
    });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray source = args[1];
      const bool reversed = args[2];
      CHECK(aten::IsValidIdArray(source)) << "Invalid source node id array.";
      const int64_t len = source->shape[0];
      const int64_t* src_data = static_cast<int64_t*>(source->data);
      std::vector<std::vector<dgl_id_t>> edges(len);
      for (int64_t i = 0; i < len; ++i) {
        auto visit = [&](dgl_id_t e, int tag) { edges[i].push_back(e); };
        DFSLabeledEdges(*g.sptr(), src_data[i], reversed, false, false, visit);
      }
      IdArray ids = MergeMultipleTraversals(edges);
      IdArray sections = ComputeMergedSections(edges);
      *rv = ConvertNDArrayVectorToPackedFunc({ids, sections});
    });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSLabeledEdges")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      GraphRef g = args[0];
      const IdArray source = args[1];
      const bool reversed = args[2];
      const bool has_reverse_edge = args[3];
      const bool has_nontree_edge = args[4];
      const bool return_labels = args[5];

      CHECK(aten::IsValidIdArray(source)) << "Invalid source node id array.";
      const int64_t len = source->shape[0];
      const int64_t* src_data = static_cast<int64_t*>(source->data);

      std::vector<std::vector<dgl_id_t>> edges(len);
      std::vector<std::vector<int64_t>> tags;
      if (return_labels) {
        tags.resize(len);
      }
      for (int64_t i = 0; i < len; ++i) {
        auto visit = [&](dgl_id_t e, int tag) {
          edges[i].push_back(e);
          if (return_labels) {
            tags[i].push_back(tag);
          }
        };
        DFSLabeledEdges(
            *g.sptr(), src_data[i], reversed, has_reverse_edge,
            has_nontree_edge, visit);
      }

      IdArray ids = MergeMultipleTraversals(edges);
      IdArray sections = ComputeMergedSections(edges);
      if (return_labels) {
        IdArray labels = MergeMultipleTraversals(tags);
        *rv = ConvertNDArrayVectorToPackedFunc({ids, labels, sections});
      } else {
        *rv = ConvertNDArrayVectorToPackedFunc({ids, sections});
      }
    });

}  // namespace traverse
}  // namespace dgl
