/**
 *  Copyright 2021 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * @file graph/transform/to_block.h
 * @brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

#ifndef DGL_GRAPH_TRANSFORM_TO_BLOCK_H_
#define DGL_GRAPH_TRANSFORM_TO_BLOCK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <functional>
#include <tuple>
#include <vector>

namespace dgl {
namespace transform {

/** @brief Mapper used in block generation which maps left and right Id arrays
 * in the original MFG to new arrays with continuous numbers.
 */
using IdsMapper =
    std::function<std::tuple<std::vector<IdArray>, std::vector<IdArray>>(
        const HeteroGraphPtr&, bool, int64_t, const DGLContext&,
        const std::vector<int64_t>&, const std::vector<EdgeArray>&,
        const std::vector<IdArray>&, const std::vector<IdArray>&,
        std::vector<IdArray>* const, std::vector<int64_t>* const)>;

/**
 * @brief Create a graph block from the set of
 * src and dst nodes (lhs and rhs respectively).
 *
 * @tparam XPU The type of device to operate on.
 * @tparam IdType The type to use as an index.
 * @param graph The graph from which to extract the block.
 * @param rhs_nodes The destination nodes of the block.
 * @param include_rhs_in_lhs Whether or not to include the
 * destination nodes of the block in the sources nodes.
 * @param [in/out] lhs_nodes The source nodes of the block.
 *
 * @return The block and the induced edges.
 */
template <DGLDeviceType XPU, typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlock(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* lhs_nodes);

/**
 * @brief A warpper function shared by CPU and GPU ```ToBlock```
 * which deal with the common preprocess and postprocess work of them.
 *
 * @tparam IdType The type to use as an index.
 * @param graph The graph from which to extract the block.
 * @param rhs_nodes The destination nodes of the block.
 * @param include_rhs_in_lhs Whether or not to include the
 * destination nodes of the block in the sources nodes.
 * @param [in/out] lhs_nodes The source nodes of the block.
 * @param MappingIdsFunc  The function to get mapped ids from original ids.
 *
 * @return The block and the induced edges.
 */
template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ProcessToBlock(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* const lhs_nodes_ptr,
    IdsMapper&& get_maping_ids);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_TO_BLOCK_H_
