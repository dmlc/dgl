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
 * @brief Read data from archive.
 * @param archive Input archive.
 * @param key Key name of data.
 *
 * @return data.
 */
torch::IValue read_from_archive(
    torch::serialize::InputArchive& archive, const std::string& key);

/**
 * @brief Read dictionary from archive.
 * @param archive Input archive.
 * @param key Key name of dictionary.
 *
 * @return dictionary.
 */
template <typename DictT>
DictT read_dict_from_archive(
    torch::serialize::InputArchive& archive, const std::string& key) {
  using KeyT = typename DictT::key_type;
  using ValueT = typename DictT::mapped_type;
  static_assert(std::is_same<KeyT, std::string>::value, "KeyT must be string");
  static_assert(
      std::is_same<ValueT, int64_t>::value ||
          std::is_same<ValueT, torch::Tensor>::value,
      "ValueT must be int64_t or torch::Tensor");
  torch::Dict<torch::IValue, torch::IValue> generic_dict =
      read_from_archive(archive, "FusedCSCSamplingGraph/node_type_to_id")
          .toGenericDict();
  DictT dict;
  for (const auto& pair : generic_dict) {
    std::string key = pair.key().toStringRef();
    if constexpr (std::is_same<ValueT, int64_t>::value) {
      int64_t value = pair.value().toInt();
      dict.insert(std::move(key), value);
    } else {
      torch::Tensor value = pair.value().toTensor();
      dict.insert(std::move(key), value);
    }
  }
  return dict;
}

}  // namespace graphbolt

#endif  // GRAPHBOLT_SERIALIZE_H_
