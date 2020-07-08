/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/metis_partition.cc
 * \brief Call Metis partitioning
 */

#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

#if !defined(_WIN32)

#include <mtmetis.h>
#include <dgl/graph_op.h>

using namespace dgl::runtime;

namespace dgl {

IdArray GraphOp::MetisPartition(GraphPtr g, int k, NDArray vwgt_arr) {
  // The index type of Metis needs to be compatible with DGL index type.
  CHECK_EQ(sizeof(mtmetis_vtx_type), sizeof(dgl_id_t));
  ImmutableGraphPtr ig = std::dynamic_pointer_cast<ImmutableGraph>(g);
  CHECK(ig) << "The input graph must be an immutable graph.";
  // This is a symmetric graph, so in-csr and out-csr are the same.
  const auto mat = ig->GetInCSR()->ToCSRMatrix();

  mtmetis_vtx_type nvtxs = g->NumVertices();
  mtmetis_vtx_type ncon = 1;        // # balacing constraints.
  CHECK_EQ(sizeof(mtmetis_adj_type), mat.indptr->dtype.bits / 8)
      << "The dtype of indptr does not match";
  mtmetis_adj_type *xadj = static_cast<mtmetis_adj_type*>(mat.indptr->data);
  CHECK_EQ(sizeof(mtmetis_vtx_type), mat.indices->dtype.bits / 8)
      << "The dtype of indices does not match";
  mtmetis_vtx_type *adjncy = static_cast<mtmetis_vtx_type*>(mat.indices->data);
  mtmetis_pid_type nparts = k;
  IdArray part_arr = aten::NewIdArray(nvtxs);
  mtmetis_wgt_type objval = 0;
  CHECK_EQ(sizeof(mtmetis_pid_type), part_arr->dtype.bits / 8)
      << "The dtype of partition Ids does not match";
  mtmetis_pid_type *part = static_cast<mtmetis_pid_type*>(part_arr->data);

  int64_t vwgt_len = vwgt_arr->shape[0];
  CHECK_EQ(sizeof(mtmetis_wgt_type), vwgt_arr->dtype.bits / 8)
      << "The vertex weight array doesn't have right type";
  CHECK(vwgt_len % g->NumVertices() == 0)
      << "The vertex weight array doesn't have right number of elements";
  mtmetis_wgt_type *vwgt = NULL;
  if (vwgt_len > 0) {
    ncon = vwgt_len / g->NumVertices();
    vwgt = static_cast<mtmetis_wgt_type*>(vwgt_arr->data);
  }

  int ret = MTMETIS_PartGraphKway(&nvtxs,      // The number of vertices
                                &ncon,       // The number of balancing constraints.
                                xadj,        // indptr
                                adjncy,      // indices
                                vwgt,        // the weights of the vertices
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
    case MTMETIS_SUCCESS:
      return part_arr;
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
    NDArray vwgt = args[2];
    *rv = GraphOp::MetisPartition(g.sptr(), k, vwgt);
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
