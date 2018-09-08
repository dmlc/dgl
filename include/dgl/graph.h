// DGL Graph interface
#ifndef DGL_DGLGRAPH_H_
#define DGL_DGLGRAPH_H_

#include <vector>
#include <stdint.h>
#include "runtime/ndarray.h"

namespace dgl {

typedef uint64_t dgl_id_t;
typedef tvm::runtime::NDArray DGLIdArray;
typedef tvm::runtime::NDArray DegreeArray;

/*!
 * \brief Base dgl graph class.
 * DGLGraph is a directed graph.
 */
class DGLGraph {
 public:
  void AddVertices(uint64_t num_vertices);

  void AddEdge(dgl_id_t src, dgl_id_t dst);

  void AddEdges(DGLIdArray src_ids, DGLIdArray dst_ids);

  void Clear();

  uint64_t NumVertices() const;

  uint64_t NumEdges() const;

  bool HasVertex(dgl_id_t vid) const;

  tvm::runtime::NDArray HasVertices(DGLIdArray vids) const;

  bool HasEdge(dgl_id_t src, dgl_id_t dst) const;

  tvm::runtime::NDArray HasEdges(DGLIdArray src_ids, DGLIdArray dst_ids) const;

  DGLIdArray Predecessors(dgl_id_t vid) const;

  DGLIdArray Successors(dgl_id_t vid) const;

  dgl_id_t GetEdgeId(dgl_id_t src, dgl_id_t dst) const;

  DGLIdArray GetEdgeIds(DGLIdArray src, DGLIdArray dst) const;

  std::pair<DGLIdArray, DGLIdArray> GetInEdge(dgl_id_t vid) const;

  std::pair<DGLIdArray, DGLIdArray> GetInEdges(DGLIdArray vids) const;
  
  std::pair<DGLIdArray, DGLIdArray> GetOutEdge(dgl_id_t vid) const;

  std::pair<DGLIdArray, DGLIdArray> GetOutEdges(DGLIdArray vids) const;

  uint64_t InDegree(dgl_id_t vid) const;

  DegreeArray InDegrees(DGLIdArray vids) const;

  uint64_t OutDegree(dgl_id_t vid) const;

  DegreeArray OutDegrees(DGLIdArray vids) const;

  DGLGraph Subgraph(DGLIdArray vids) const;

  DGLGraph EdgeSubgraph(DGLIdArray src, DGLIdArray dst) const;

  DGLGraph Reverse() const;
};

}  // namespace dgl

#endif  // DGL_DGLGRAPH_H_
