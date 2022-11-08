/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/traversal.cc
 * @brief Graph traversal implementation
 */
#include <dgl/graph_traversal.h>
#include <dgl/packed_func_ext.h>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace traverse {

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSNodes_v2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef g = args[0];
      const IdArray src = args[1];
      bool reversed = args[2];
      aten::CSRMatrix csr;
      if (reversed) {
        csr = g.sptr()->GetCSCMatrix(0);
      } else {
        csr = g.sptr()->GetCSRMatrix(0);
      }
      const auto& front = aten::BFSNodesFrontiers(csr, src);
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
    });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSEdges_v2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef g = args[0];
      const IdArray src = args[1];
      bool reversed = args[2];
      aten::CSRMatrix csr;
      if (reversed) {
        csr = g.sptr()->GetCSCMatrix(0);
      } else {
        csr = g.sptr()->GetCSRMatrix(0);
      }

      const auto& front = aten::BFSEdgesFrontiers(csr, src);
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
    });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLTopologicalNodes_v2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef g = args[0];
      bool reversed = args[1];
      aten::CSRMatrix csr;
      if (reversed) {
        csr = g.sptr()->GetCSCMatrix(0);
      } else {
        csr = g.sptr()->GetCSRMatrix(0);
      }

      const auto& front = aten::TopologicalNodesFrontiers(csr);
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
    });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSEdges_v2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef g = args[0];
      const IdArray source = args[1];
      const bool reversed = args[2];
      CHECK(aten::IsValidIdArray(source)) << "Invalid source node id array.";
      aten::CSRMatrix csr;
      if (reversed) {
        csr = g.sptr()->GetCSCMatrix(0);
      } else {
        csr = g.sptr()->GetCSRMatrix(0);
      }
      const auto& front = aten::DGLDFSEdges(csr, source);
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
    });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSLabeledEdges_v2")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef g = args[0];
      const IdArray source = args[1];
      const bool reversed = args[2];
      const bool has_reverse_edge = args[3];
      const bool has_nontree_edge = args[4];
      const bool return_labels = args[5];
      aten::CSRMatrix csr;
      if (reversed) {
        csr = g.sptr()->GetCSCMatrix(0);
      } else {
        csr = g.sptr()->GetCSRMatrix(0);
      }

      const auto& front = aten::DGLDFSLabeledEdges(
          csr, source, has_reverse_edge, has_nontree_edge, return_labels);

      if (return_labels) {
        *rv = ConvertNDArrayVectorToPackedFunc(
            {front.ids, front.tags, front.sections});
      } else {
        *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
      }
    });

}  // namespace traverse
}  // namespace dgl
