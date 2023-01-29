/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/traversal.cc
 * @brief Graph traversal implementation
 */

#include "./traversal.h"

#include <dgl/graph_traversal.h>

#include <algorithm>
#include <queue>

namespace dgl {
namespace aten {
namespace impl {
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
      {total_len}, DGLDataType{kDGLInt, sizeof(DType) * 8, 1},
      DGLContext{kDGLCPU, 0});
  DType* ret_data = static_cast<DType*>(ret->data);
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

template <DGLDeviceType XPU, typename IdType>
Frontiers BFSNodesFrontiers(const CSRMatrix& csr, IdArray source) {
  std::vector<IdType> ids;
  std::vector<int64_t> sections;
  VectorQueueWrapper<IdType> queue(&ids);
  auto visit = [&](const int64_t v) {};
  auto make_frontier = [&]() {
    if (!queue.empty()) {
      // do not push zero-length frontier
      sections.push_back(queue.size());
    }
  };
  BFSTraverseNodes<IdType>(csr, source, &queue, visit, make_frontier);

  Frontiers front;
  front.ids = VecToIdArray(ids, sizeof(IdType) * 8);
  front.sections = VecToIdArray(sections, sizeof(int64_t) * 8);
  return front;
}

template Frontiers BFSNodesFrontiers<kDGLCPU, int32_t>(
    const CSRMatrix&, IdArray);
template Frontiers BFSNodesFrontiers<kDGLCPU, int64_t>(
    const CSRMatrix&, IdArray);

template <DGLDeviceType XPU, typename IdType>
Frontiers BFSEdgesFrontiers(const CSRMatrix& csr, IdArray source) {
  std::vector<IdType> ids;
  std::vector<int64_t> sections;
  // NOTE: std::queue has no top() method.
  std::vector<IdType> nodes;
  VectorQueueWrapper<IdType> queue(&nodes);
  auto visit = [&](const IdType e) { ids.push_back(e); };
  bool first_frontier = true;
  auto make_frontier = [&] {
    if (first_frontier) {
      first_frontier = false;  // do not push the first section when doing edges
    } else if (!queue.empty()) {
      // do not push zero-length frontier
      sections.push_back(queue.size());
    }
  };
  BFSTraverseEdges<IdType>(csr, source, &queue, visit, make_frontier);

  Frontiers front;
  front.ids = VecToIdArray(ids, sizeof(IdType) * 8);
  front.sections = VecToIdArray(sections, sizeof(int64_t) * 8);
  return front;
}

template Frontiers BFSEdgesFrontiers<kDGLCPU, int32_t>(
    const CSRMatrix&, IdArray);
template Frontiers BFSEdgesFrontiers<kDGLCPU, int64_t>(
    const CSRMatrix&, IdArray);

template <DGLDeviceType XPU, typename IdType>
Frontiers TopologicalNodesFrontiers(const CSRMatrix& csr) {
  std::vector<IdType> ids;
  std::vector<int64_t> sections;
  VectorQueueWrapper<IdType> queue(&ids);
  auto visit = [&](const uint64_t v) {};
  auto make_frontier = [&]() {
    if (!queue.empty()) {
      // do not push zero-length frontier
      sections.push_back(queue.size());
    }
  };
  TopologicalNodes<IdType>(csr, &queue, visit, make_frontier);

  Frontiers front;
  front.ids = VecToIdArray(ids, sizeof(IdType) * 8);
  front.sections = VecToIdArray(sections, sizeof(int64_t) * 8);
  return front;
}

template Frontiers TopologicalNodesFrontiers<kDGLCPU, int32_t>(
    const CSRMatrix&);
template Frontiers TopologicalNodesFrontiers<kDGLCPU, int64_t>(
    const CSRMatrix&);

template <DGLDeviceType XPU, typename IdType>
Frontiers DGLDFSEdges(const CSRMatrix& csr, IdArray source) {
  const int64_t len = source->shape[0];
  const IdType* src_data = static_cast<IdType*>(source->data);
  std::vector<std::vector<IdType>> edges(len);

  for (int64_t i = 0; i < len; ++i) {
    auto visit = [&](IdType e, int tag) { edges[i].push_back(e); };
    DFSLabeledEdges<IdType>(csr, src_data[i], false, false, visit);
  }

  Frontiers front;
  front.ids = MergeMultipleTraversals(edges);
  front.sections = ComputeMergedSections(edges);
  return front;
}

template Frontiers DGLDFSEdges<kDGLCPU, int32_t>(const CSRMatrix&, IdArray);
template Frontiers DGLDFSEdges<kDGLCPU, int64_t>(const CSRMatrix&, IdArray);

template <DGLDeviceType XPU, typename IdType>
Frontiers DGLDFSLabeledEdges(
    const CSRMatrix& csr, IdArray source, const bool has_reverse_edge,
    const bool has_nontree_edge, const bool return_labels) {
  const int64_t len = source->shape[0];
  const IdType* src_data = static_cast<IdType*>(source->data);
  std::vector<std::vector<IdType>> edges(len);
  std::vector<std::vector<int64_t>> tags;

  if (return_labels) {
    tags.resize(len);
  }

  for (int64_t i = 0; i < len; ++i) {
    auto visit = [&](IdType e, int64_t tag) {
      edges[i].push_back(e);
      if (return_labels) {
        tags[i].push_back(tag);
      }
    };
    DFSLabeledEdges<IdType>(
        csr, src_data[i], has_reverse_edge, has_nontree_edge, visit);
  }

  Frontiers front;
  front.ids = MergeMultipleTraversals(edges);
  front.sections = ComputeMergedSections(edges);
  if (return_labels) {
    front.tags = MergeMultipleTraversals(tags);
  }

  return front;
}

template Frontiers DGLDFSLabeledEdges<kDGLCPU, int32_t>(
    const CSRMatrix&, IdArray, const bool, const bool, const bool);
template Frontiers DGLDFSLabeledEdges<kDGLCPU, int64_t>(
    const CSRMatrix&, IdArray, const bool, const bool, const bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
