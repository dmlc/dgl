/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/metis_partition.cc
 * \brief Call Metis partitioning
 */

#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

#if !defined(_WIN32)

#include <metis.h>
#include <dgl/graph_op.h>

using namespace dgl::runtime;

namespace dgl {

IdArray GraphOp::MetisPartition(GraphPtr g, int k) {
  // The index type of Metis needs to be compatible with DGL index type.
  CHECK_EQ(sizeof(idx_t), sizeof(dgl_id_t));
  ImmutableGraphPtr ig = std::dynamic_pointer_cast<ImmutableGraph>(g);
  CHECK(ig) << "The input graph must be an immutable graph.";
  // This is a symmetric graph, so in-csr and out-csr are the same.
  const auto mat = ig->GetInCSR()->ToCSRMatrix();

  idx_t nvtxs = g->NumVertices();
  idx_t ncon = 1;        // # balacing constraints.
  idx_t *xadj = static_cast<idx_t*>(mat.indptr->data);
  idx_t *adjncy = static_cast<idx_t*>(mat.indices->data);
  idx_t nparts = k;
  IdArray part_arr = aten::NewIdArray(nvtxs);
  idx_t objval = 0;
  idx_t *part = static_cast<idx_t*>(part_arr->data);
  int ret = METIS_PartGraphKway(&nvtxs,      // The number of vertices
                                &ncon,       // The number of balancing constraints.
                                xadj,        // indptr
                                adjncy,      // indices
                                NULL,        // the weights of the vertices
                                NULL,        // The size of the vertices for computing
                                // the total communication volume
                                NULL,        // The weights of the edges
                                &nparts,     // The number of partitions.
                                NULL,        // the desired weight for each partition and constraint
                                NULL,        // the allowed load imbalance tolerance
                                NULL,        // the array of options
                                &objval,      // the edge-cut or the total communication volume of
                                // the partitioning solution
                                part);
  LOG(INFO) << "Partition a graph with " << g->NumVertices()
      << " nodes and " << g->NumEdges()
      << " edges into " << k
      << " parts and get " << objval << " edge cuts";
  switch (ret) {
    case METIS_OK:
      return part_arr;
    case METIS_ERROR_INPUT:
      LOG(FATAL) << "Error in Metis partitioning: input error";
    case METIS_ERROR_MEMORY:
      LOG(FATAL) << "Error in Metis partitioning: cannot allocate memory";
    default:
      LOG(FATAL) << "Error in Metis partitioning: other errors";
  }
  // return an array of 0 elements to indicate the error.
  return aten::NullArray();
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLMetisPartition")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    int k = args[1];
    *rv = GraphOp::MetisPartition(g.sptr(), k);
  });

}   // namespace dgl

#else   // defined(_WIN32)

using namespace dgl::runtime;

namespace dgl {

DGL_REGISTER_GLOBAL("transform._CAPI_DGLMetisPartition")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    int k = args[1];
    LOG(WARNING) << "DGL doesn't support METIS partitioning in Windows";
    *rv = aten::NullArray();
  });


}  // namespace dgl
#endif  // !defined(_WIN32)
