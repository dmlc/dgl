/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csc_sampling_graph.h
 * @brief Header file of csc sampling graph.
 */

#include <torch/torch.h>

#include <string>
#include <vector>

namespace graphbolt {
namespace utils {

/**
 * @brief Utility function to write to archive.
 * @param archive Output archive.
 * @param key Key name used in saving.
 * @param data Data that could be constructed as `torch::IValue`.
*/
template <typename DataT>
void write_to_archive(
    torch::serialize::OutputArchive& archive,
    const std::string& key,
    const DataT& data) {
  archive.write(key, data);
}

/**
 * @brief Specialization utility function to save vector.
 * @param archive Output archive.
 * @param key Key name used in saving.
 * @param data Vector of string.
 *
 *
*/
template<>
void write_to_archive<std::vector<std::string>>(
    torch::serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<std::string>& data) {
  archive.write(
      key + "/size", torch::tensor(static_cast<int64_t>(data.size())));
  for (const auto index : c10::irange(data.size())) {
    archive.write(
        key + "/" + std::to_string(index), data[index]);
  }
}

/**
 * @brief Utility function to read from archive.
 * @param archive Input archive.
 * @param key Key name used in reading.
 * @param data Data that could be constructed as `torch::IValue`.
*/
template <typename DataT=torch::IValue>
void read_from_archive(
    torch::serialize::InputArchive& archive,
    const std::string& key,
    DataT& data) {
  archive.read(key, data);
}

/**
 * @brief Specialization utility function to read from archive.
 * @param archive Input archive.
 * @param key Key name used in reading.
 * @param data Data that is `bool`.
*/
template <>
void read_from_archive<bool>(
    torch::serialize::InputArchive& archive,
    const std::string& key,
    bool& data) {
  torch::IValue iv_data;
  archive.read(key, iv_data);
  data = iv_data.toBool();
}

/**
 * @brief Specialization utility function to read from archive.
 * @param archive Input archive.
 * @param key Key name used in reading.
 * @param data Data that is `int64_t`.
*/
template <>
void read_from_archive<int64_t>(
    torch::serialize::InputArchive& archive,
    const std::string& key,
    int64_t& data) {
  torch::IValue iv_data;
  archive.read(key, iv_data);
  data = iv_data.toInt();
}

/**
 * @brief Specialization utility function to read from archive.
 * @param archive Input archive.
 * @param key Key name used in reading.
 * @param data Data that is `std::string`.
*/
template <>
void read_from_archive<std::string>(
    torch::serialize::InputArchive& archive,
    const std::string& key,
    std::string& data) {
  torch::IValue iv_data;
  archive.read(key, iv_data);
  data = iv_data.toString();
}

/**
 * @brief Specialization utility function to read from archive.
 * @param archive Input archive.
 * @param key Key name used in reading.
 * @param data Data that is `torch::Tensor`.
*/
template <>
void read_from_archive<torch::Tensor>(
    torch::serialize::InputArchive& archive,
    const std::string& key,
    torch::Tensor& data) {
  torch::IValue iv_data;
  archive.read(key, iv_data);
  data = iv_data.toTensor();
}

/**
 * @brief Specialization utility function to read to vector.
 * @param archive Output archive.
 * @param key Key name used in saving.
 * @param data Vector of string.
*/
template<>
void read_from_archive<std::vector<std::string>>(
    torch::serialize::InputArchive& archive,
    const std::string& key,
    std::vector<std::string>& data) {
  int64_t size = 0;
  read_from_archive<int64_t>(archive, key + "/size", size);
  data.resize(static_cast<size_t>(size));
  std::string element;
  for (int64_t index = 0; index < size; ++index) {
    read_from_archive<std::string>(archive,
        key + "/" + std::to_string(index), element);
    data[index] = element;
  }
}

}  // namespace utils
}  // namespace graphbolt
