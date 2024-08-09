/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file graphbolt/cuda_sampling_ops.h
 * @brief Available CUDA sampling operations in Graphbolt.
 */
#ifndef GRAPHBOLT_CUDA_SAMPLING_OPS_H_
#define GRAPHBOLT_CUDA_SAMPLING_OPS_H_

#include <graphbolt/fused_sampled_subgraph.h>
#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief Sample neighboring edges of the given nodes and return the induced
 * subgraph.
 *
 * @param indptr Index pointer array of the CSC.
 * @param indices Indices array of the CSC.
 * @param seeds The nodes from which to sample neighbors. If not provided,
 * assumed to be equal to torch.arange(indptr.size(0) - 1).
 * @param seed_offsets The offsets of the given seeds,
 * seeds[seed_offsets[i]: seed_offsets[i + 1]] has node type i.
 * @param fanouts The number of edges to be sampled for each node with or
 * without considering edge types.
 *   - When the length is 1, it indicates that the fanout applies to all
 * neighbors of the node as a collective, regardless of the edge type.
 *   - Otherwise, the length should equal to the number of edge types, and
 * each fanout value corresponds to a specific edge type of the node.
 * The value of each fanout should be >= 0 or = -1.
 *   - When the value is -1, all neighbors will be chosen for sampling. It is
 * equivalent to selecting all neighbors with non-zero probability when the
 * fanout is >= the number of neighbors (and replacement is set to false).
 *   - When the value is a non-negative integer, it serves as a minimum
 * threshold for selecting neighbors.
 * @param replace Boolean indicating whether the sample is preformed with or
 * without replacement. If True, a value can be selected multiple times.
 * Otherwise, each value can be selected only once.
 * @param layer Boolean indicating whether neighbors should be sampled in a
 * layer sampling fashion. Uses the LABOR-0 algorithm to increase overlap of
 * sampled edges, see arXiv:2210.13339.
 * @param returning_indices_is_optional Boolean indicating whether returning
 * indices tensor is optional.
 * @param type_per_edge A tensor representing the type of each edge, if present.
 * @param probs_or_mask An optional tensor with (unnormalized) probabilities
 * corresponding to each neighboring edge of a node. It must be
 * a 1D tensor, with the number of elements equaling the total number of edges.
 * @param node_type_to_id A dictionary mapping node type names to type IDs. The
 * length of it is equal to the number of node types. The key is the node type
 * name, and the value is the corresponding type ID.
 * @param edge_type_to_id A dictionary mapping edge type names to type IDs. The
 * length of it is equal to the number of edge types. The key is the edge type
 * name, and the value is the corresponding type ID.
 * @param random_seed The random seed for the sampler for layer=True.
 * @param seed2_contribution The contribution of the second random seed, [0, 1)
 * for layer=True.
 * @param seeds_timestamp The timestamp of the seeds.
 * @param seeds_pre_time_window The time window of the seeds represents a period
 * of time before `seeds_timestamp`. If provided, only neighbors and related
 * edges whose timestamps fall within
 * `[seeds_timestamp - seeds_pre_time_window, seeds_timestamp]` will be
 * filtered.
 * @param node_timestamp An optional tensor that contains the timestamp of nodes
 * in the graph.
 * @param edge_timestamp An optional tensor that contains the timestamp of edges
 * in the graph.
 *
 * @return An intrusive pointer to a FusedSampledSubgraph object containing
 * the sampled graph's information.
 */
c10::intrusive_ptr<sampling::FusedSampledSubgraph> SampleNeighbors(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> seeds,
    torch::optional<std::vector<int64_t>> seed_offsets,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool returning_indices_is_optional,
    torch::optional<torch::Tensor> type_per_edge = torch::nullopt,
    torch::optional<torch::Tensor> probs_or_mask = torch::nullopt,
    torch::optional<torch::Tensor> node_type_offset = torch::nullopt,
    torch::optional<torch::Dict<std::string, int64_t>> node_type_to_id =
        torch::nullopt,
    torch::optional<torch::Dict<std::string, int64_t>> edge_type_to_id =
        torch::nullopt,
    torch::optional<torch::Tensor> random_seed = torch::nullopt,
    float seed2_contribution = .0f,
    // Optional temporal sampling arguments begin.
    torch::optional<torch::Tensor> seeds_timestamp = torch::nullopt,
    torch::optional<torch::Tensor> seeds_pre_time_window = torch::nullopt,
    torch::optional<torch::Tensor> node_timestamp = torch::nullopt,
    torch::optional<torch::Tensor> edge_timestamp = torch::nullopt
    // Optional temporal sampling arguments end.
);

/**
 * @brief Return the subgraph induced on the inbound edges of the given nodes.
 * @param nodes Type agnostic node IDs to form the subgraph.
 *
 * @return FusedSampledSubgraph.
 */
c10::intrusive_ptr<sampling::FusedSampledSubgraph> InSubgraph(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<torch::Tensor> type_per_edge);

}  //  namespace ops
}  //  namespace graphbolt

#endif  // GRAPHBOLT_CUDA_SAMPLING_OPS_H_
