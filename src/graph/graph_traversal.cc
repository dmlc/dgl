/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/graph_traversal.cc
 * \brief Graph traversal implementation
 */
#include <dgl/graph.h>
#include <algorithm>
#include <stack>

namespace dgl {
namespace {
/*!
 * \brief Convert an IdVector to an IdArray.
 */
inline IdArray IdVectorToIdArray(const IdVector& v) {
  const int64_t len = v.size();
  auto a = IdArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  std::copy(v.begin(), v.end(), static_cast<dgl_id_t*>(a->data));
  return a;
}

/*!
 * \brief Analoguous to Python's zip, ZipIdVectorsInIdArray zips a collections of IdVectorsand return the result in a format that can be easily returned to Python.
 */
inline std::pair<IdArray, IdArray> ZipIdVectorsInIdArray(const std::vector<IdVector> &id_vectors) {
  auto size = [](const IdVector &x) { return x.size(); };
  std::vector<size_t> s(id_vectors.size());
  std::transform(id_vectors.begin(), id_vectors.end(), s.begin(), size);
  auto max_s = *std::max_element(s.begin(), s.end());
  std::set<dgl_id_t> included;
  for (size_t i = 0; i < id_vectors.size(); ++i) {
    included.insert(i);
  }
  std::vector<size_t> t(max_s);
  for (size_t i = 0; i < max_s; ++i) {
    for (auto j : included) {
      if (i == id_vectors[j].size()) {
        included.erase(j);
      }
    }
    t[i] = included.size();
  }

  for (size_t i = 0; i < id_vectors.size(); ++i) {
    included.insert(i);
  }
  const int64_t sum_t = std::accumulate(t.begin(), t.end(), 0);
  auto id_array = IdArray::Empty({sum_t}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  auto id_data = static_cast<int64_t*>(id_array->data);
  for (size_t i = 0; i < max_s; ++i) {
    for (auto j : included) {
      if (i < id_vectors[j].size()) {
        id_data[i * id_vectors.size() + j] = id_vectors[j][i];
      } else {
        included.erase(j);
      }
    }
  }
  auto s_array = IdVectorToIdArray(t);
  return std::make_pair(id_array, s_array);
}

}  // namespace

std::pair<IdArray, IdArray> Graph::BFS(IdArray src, bool out) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  std::vector<dgl_id_t> vv;
  auto src_data = static_cast<int64_t*>(src->data);
  std::copy(src_data, src_data + src->shape[0], std::back_inserter(vv));
  size_t i = 0;
  size_t j = vv.size();
  const auto &adj = out ? adjlist_ : reverse_adjlist_;
  std::vector<bool> visited(adj.size());
  for (auto v : vv) {
    visited[v] = true;
  }
  std::vector<size_t> ss;
  while (i < j) {
    for (size_t k = i; k != j; k++) {
      for (auto v : adj[vv[k]].succ) {
        if (!visited[v]) {
            visited[v] = true;
            vv.push_back(v);
        }
      }
    }
    ss.push_back(j - i);
    i = j;
    j = vv.size();
  }
  return std::make_pair(IdVectorToIdArray(vv), IdVectorToIdArray(ss));
}

std::tuple<IdVector, IdVector, IdVector> Graph::DFSLabeledEdges_(
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
}

std::tuple<IdArray, IdArray, IdArray, IdArray> Graph::DFSLabeledEdges(
  IdArray source, bool out, bool reverse_edge, bool nontree_edge) const {
  CHECK(IsValidIdArray(vids)) << "Invalid vertex id array.";
  int64_t *source_data = static_cast<int64_t*>(source->data);
  std::vector<IdVector> src(source->shape[0]);
  std::vector<IdVector> dst(source->shape[0]);
  std::vector<IdVector> type(source->shape[0]);
  for (int64_t i = 0; i < source->shape[0]; i++) {
    auto ret = DFSLabeledEdges_(source_data[i], out, reverse_edge, nontree_edge);
    src[i] = std::get<0>(ret);
    dst[i] = std::get<1>(ret);
    type[i] = std::get<2>(ret);
  }
  auto src_pair = ZipIdVectorsInIdArray(src);
  auto dst_pair = ZipIdVectorsInIdArray(dst);
  auto type_pair = ZipIdVectorsInIdArray(type);
  return std::make_tuple(src_pair.first, dst_pair.first, type_pair.first, src_pair.second);
}

std::pair<IdArray, IdArray> Graph::TopologicalTraversal(bool out) const {
  IdVector uu(adjlist_.size());
  std::iota(uu.begin(), uu.end(), 0);
  auto u_array = IdVectorToIdArray(uu);
  auto deg = out ? InDegrees(u_array) : OutDegrees(u_array);
  auto deg_data = static_cast<int64_t*>(deg->data);
  IdVector vv;
  for (int64_t k = 0; k < deg->shape[0]; k++) {
    if (deg_data[k] == 0) {
      vv.push_back(k);
    }
  }
  size_t i = 0;
  size_t j = vv.size();
  const auto &adj = out ? adjlist_ : reverse_adjlist_;
  IdVector ss;
  while (i < j) {
    for (size_t k = i; k != j; k++) {
      for (auto v : adj[vv[k]].succ) {
        if (--(deg_data[v]) == 0) {
          vv.push_back(v);
        }
      }
    }
    ss.push_back(j - i);
    i = j;
    j = vv.size();
  }
  return std::make_pair(IdVectorToIdArray(vv), IdVectorToIdArray(ss));
}

}  // namespace dgl
