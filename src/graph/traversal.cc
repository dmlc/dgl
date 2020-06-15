/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/traversal.cc
 * \brief Graph traversal implementation
 */
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <algorithm>
#include <queue>
#include "./traversal.h"
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace traverse {

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSNodes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray src = args[1];
    bool reversed = args[2];
    const auto& front = BFSNodesFrontiers(*(g.sptr()), src, reversed);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLBFSEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray src = args[1];
    bool reversed = args[2];
    const auto& front = BFSEdgesFrontiers(*(g.sptr()), src, reversed);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLTopologicalNodes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    bool reversed = args[1];
    const auto& front = TopologicalNodesFrontiers(*g.sptr(), reversed);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });


DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray source = args[1];
    const bool reversed = args[2];
    CHECK(aten::IsValidIdArray(source)) << "Invalid source node id array.";
    const auto& front = DGLDFSEdges(*g.sptr(), source, reversed);
    *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
  });

DGL_REGISTER_GLOBAL("traversal._CAPI_DGLDFSLabeledEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const IdArray source = args[1];
    const bool reversed = args[2];
    const bool has_reverse_edge = args[3];
    const bool has_nontree_edge = args[4];
    const bool return_labels = args[5];

    const auto& front = DGLDFSLabeledEdges(*g.sptr(),
                                           source,
                                           reversed,
                                           has_reverse_edge,
                                           has_nontree_edge,
                                           return_labels);

    if (return_labels) {
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.tags, front.sections});
    } else {
      *rv = ConvertNDArrayVectorToPackedFunc({front.ids, front.sections});
    }
  });

}  // namespace traverse
}  // namespace dgl
