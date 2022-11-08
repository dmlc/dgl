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
 * @file graph/transform/compact.h
 * @brief Functions to find and eliminate the common isolated nodes across
 * all given graphs with the same set of nodes.
 */

#ifndef DGL_GRAPH_TRANSFORM_COMPACT_H_
#define DGL_GRAPH_TRANSFORM_COMPACT_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <utility>
#include <vector>

namespace dgl {
namespace transform {

/**
 * @brief Given a list of graphs with the same set of nodes, find and eliminate
 * the common isolated nodes across all graphs.
 *
 * @tparam XPU The type of device to operate on.
 * @tparam IdType The type to use as an index.
 * @param graphs The list of graphs to be compacted.
 * @param always_preserve The vector of nodes to be preserved.
 *
 * @return The vector of compacted graphs and the vector of induced nodes.
 */
template <DGLDeviceType XPU, typename IdType>
std::pair<std::vector<HeteroGraphPtr>, std::vector<IdArray>> CompactGraphs(
    const std::vector<HeteroGraphPtr> &graphs,
    const std::vector<IdArray> &always_preserve);

}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_COMPACT_H_
