/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/transform/line_graph.cc
 * @brief Line graph implementation
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/transform.h>

#include <utility>
#include <vector>

#include "../../c_api_common.h"
#include "../heterograph.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace transform {

/**
 * @brief Create Line Graph.
 * @param hg Graph.
 * @param backtracking whether the pair of (v, u) (u, v) edges are treated as
 *        linked.
 * @return The Line Graph.
 */
HeteroGraphPtr CreateLineGraph(HeteroGraphPtr hg, bool backtracking) {
  const auto hgp = std::dynamic_pointer_cast<HeteroGraph>(hg);
  return hgp->LineGraph(backtracking);
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLHeteroLineGraph")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      bool backtracking = args[1];

      auto hgptr = CreateLineGraph(hg.sptr(), backtracking);
      *rv = HeteroGraphRef(hgptr);
    });

};  // namespace transform
};  // namespace dgl
