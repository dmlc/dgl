/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/serialize.h
 * @brief Utility functions for serialize and deserialize.
 */

#ifndef GRAPHBOLT_SERIALIZE_H_
#define GRAPHBOLT_SERIALIZE_H_

#include <graphbolt/fused_csc_sampling_graph.h>
#include <torch/torch.h>

#include <string>
#include <vector>

/**
 * @brief Overload stream operator to enable `torch::save()` and `torch.load()`
 * for FusedCSCSamplingGraph.
 */
namespace torch {

/**
 * @brief Overload input stream operator for FusedCSCSamplingGraph
 * deserialization.
 * @param archive Input stream for deserializing.
 * @param graph FusedCSCSamplingGraph.
 *
 * @return archive
 */
inline serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    graphbolt::sampling::FusedCSCSamplingGraph& graph);

/**
 * @brief Overload output stream operator for FusedCSCSamplingGraph
 * serialization.
 * @param archive Output stream for serializing.
 * @param graph FusedCSCSamplingGraph.
 *
 * @return archive
 */
inline serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const graphbolt::sampling::FusedCSCSamplingGraph& graph);

}  // namespace torch

namespace graphbolt {

/**
 * @brief Load FusedCSCSamplingGraph from file.
 * @param filename File name to read.
 *
 * @return FusedCSCSamplingGraph.
 */
c10::intrusive_ptr<sampling::FusedCSCSamplingGraph> LoadFusedCSCSamplingGraph(
    const std::string& filename);

/**
 * @brief Save FusedCSCSamplingGraph to file.
 * @param graph FusedCSCSamplingGraph to save.
 * @param filename File name to save.
 *
 */
void SaveFusedCSCSamplingGraph(
    c10::intrusive_ptr<sampling::FusedCSCSamplingGraph> graph,
    const std::string& filename);

/**
 * @brief Read data from archive.
 * @param archive Input archive.
 * @param key Key name of data.
 *
 * @return data.
 */
torch::IValue read_from_archive(
    torch::serialize::InputArchive& archive, const std::string& key);

}  // namespace graphbolt

#endif  // GRAPHBOLT_SERIALIZE_H_
