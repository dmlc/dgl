/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/serialize.h
 * @brief Utility functions for serialize and deserialize.
 */

#ifndef GRAPHBOLT_SERIALIZE_H_
#define GRAPHBOLT_SERIALIZE_H_

#include <graphbolt/csc_sampling_graph.h>
#include <torch/torch.h>

#include <string>
#include <vector>

/**
 * @brief Overload stream operator to enable `torch::save()` and `torch.load()`
 * for CSCSamplingGraph.
 */
namespace torch {

/**
 * @brief Overload input stream operator for CSCSamplingGraph deserialization.
 * @param archive Input stream for deserializing.
 * @param graph CSCSamplingGraph.
 *
 * @return archive
 */
inline serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    graphbolt::sampling::CSCSamplingGraph& graph);

/**
 * @brief Overload output stream operator for CSCSamplingGraph serialization.
 * @param archive Output stream for serializing.
 * @param graph CSCSamplingGraph.
 *
 * @return archive
 */
inline serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const graphbolt::sampling::CSCSamplingGraph& graph);

}  // namespace torch

namespace graphbolt {

/**
 * @brief Load CSCSamplingGraph from file.
 * @param filename File name to read.
 *
 * @return CSCSamplingGraph.
 */
c10::intrusive_ptr<sampling::CSCSamplingGraph> LoadCSCSamplingGraph(
    const std::string& filename);

/**
 * @brief Save CSCSamplingGraph to file.
 * @param graph CSCSamplingGraph to save.
 * @param filename File name to save.
 *
 */
void SaveCSCSamplingGraph(
    c10::intrusive_ptr<sampling::CSCSamplingGraph> graph,
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
