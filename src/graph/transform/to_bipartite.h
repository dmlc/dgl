/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/transform/to_bipartite.h
 * \brief Array operator templates
 */

#ifndef DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_
#define DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

template<DLDeviceType XPU, typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlock(HeteroGraphPtr graph, const std::vector<IdArray> &rhs_nodes,
        bool include_rhs_in_lhs);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_TO_BIPARTITE_H_
