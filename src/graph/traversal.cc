/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/traversal.cc
 * \brief Graph traversal implementation
 */
#include <algorithm>
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
// A utility view class for a range of data in a vector.
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

// Internal function to merge multiple traversal traces into one ndarray.
// It is similar to zip the vectors together.
template<typename DType>
IdArray MergeMultipleTraversals(
    const std::vector<std::vector<DType>>& traces) {
  int64_t max_len = 0, total_len = 0;
  for (size_t i = 0; i < traces.size(); ++i) {
    const int64_t tracelen = traces[i].size();
    max_len = std::max(max_len, tracelen);
    total_len += traces[i].size();
  }
  IdArray ret = IdArray::Empty({total_len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
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
template<typename DType>
IdArray ComputeMergedSections(
    const std::vector<std::vector<DType>>& traces) {
  int64_t max_len = 0;
  for (size_t i = 0; i < traces.size(); ++i) {
    const int64_t tracelen = traces[i].size();
    max_len = std::max(max_len, tracelen);
  }
  IdArray ret = IdArray::Empty({max_len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
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


TVM_REGISTER_GLOBAL("traversal._CAPI_DGLDFSEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray source = args[1];
    const bool reversed = args[2];
    CHECK(IsValidIdArray(source)) << "Invalid source node id array.";
    const int64_t len = source->shape[0];
    const int64_t* src_data = static_cast<int64_t*>(source->data);
    std::vector<std::vector<dgl_id_t>> edges(len);
    for (int64_t i = 0; i < len; ++i) {
      auto visit = [&] (dgl_id_t e, int tag) { edges[i].push_back(e); };
      DFSLabeledEdges(*gptr, src_data[i], reversed, false, false, visit);
    }
    IdArray ids = MergeMultipleTraversals(edges);
    IdArray sections = ComputeMergedSections(edges);
    *rv = ConvertNDArrayVectorToPackedFunc({ids, sections});
  });

TVM_REGISTER_GLOBAL("traversal._CAPI_DGLDFSLabeledEdges")
.set_body([] (TVMArgs args, TVMRetValue* rv) {
    GraphHandle ghandle = args[0];
    const Graph* gptr = static_cast<Graph*>(ghandle);
    const IdArray source = args[1];
    const bool reversed = args[2];
    const bool has_reverse_edge = args[3];
    const bool has_nontree_edge = args[4];
    const bool return_labels = args[5];

    CHECK(IsValidIdArray(source)) << "Invalid source node id array.";
    const int64_t len = source->shape[0];
    const int64_t* src_data = static_cast<int64_t*>(source->data);

    std::vector<std::vector<dgl_id_t>> edges(len);
    std::vector<std::vector<int64_t>> tags;
    if (return_labels) {
      tags.resize(len);
    }
    for (int64_t i = 0; i < len; ++i) {
      auto visit = [&] (dgl_id_t e, int tag) {
        edges[i].push_back(e);
        if (return_labels) {
          tags[i].push_back(tag);
        }
      };
      DFSLabeledEdges(*gptr, src_data[i], reversed,
          has_reverse_edge, has_nontree_edge, visit);
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
