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
 * deserialization. This enables `torch::load()` for FusedCSCSamplingGraph.
 *
 * @param archive Input stream for deserializing.
 * @param graph FusedCSCSamplingGraph.
 *
 * @return archive
 *
 * @code
 * auto&& graph = c10::make_intrusive<sampling::FusedCSCSamplingGraph>();
 * torch::load(*graph, filename);
 */
inline serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    graphbolt::sampling::FusedCSCSamplingGraph& graph);

/**
 * @brief Overload output stream operator for FusedCSCSamplingGraph
 * serialization. This enables `torch::save()` for FusedCSCSamplingGraph.
 * @param archive Output stream for serializing.
 * @param graph FusedCSCSamplingGraph.
 *
 * @return archive
 *
 * @code
 * auto&& graph = c10::make_intrusive<sampling::FusedCSCSamplingGraph>();
 * torch::save(*graph, filename);
 */
inline serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const graphbolt::sampling::FusedCSCSamplingGraph& graph);

}  // namespace torch

namespace graphbolt {

/**
 * @brief Read data from archive and format to specified type.
 * @param archive Input archive.
 * @param key Key name of data.
 *
 * @return data.
 */
template <typename T>
T read_from_archive(
    torch::serialize::InputArchive& archive, const std::string& key) {
  torch::IValue data;
  archive.read(key, data);
  return data.to<T>();
}

}  // namespace graphbolt

#endif  // GRAPHBOLT_SERIALIZE_H_
