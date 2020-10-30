/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */


#ifndef DGL_GRAPH_TRANSFORM_CUDA_TO_BLOCK_H_
#define DGL_GRAPH_TRANSFORM_CUDA_TO_BLOCK_H_

#include <vector>
#include <tuple>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>

namespace dgl {
namespace transform {

std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
CudaToBlock(
    HeteroGraphPtr graph,
    const std::vector<IdArray>& rhs_nodes,
    const bool include_rhs_in_lhs);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_CUDA_TO_BLOCK_H_

