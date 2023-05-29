/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file shared_memory_utils.cc
 * @brief Share memory utility function implementation.
 */
#include "./shared_memory_utils.h"

#include <graphbolt/serialize.h>
#include <graphbolt/shared_memory.h>

namespace graphbolt {
namespace sampling {

static SharedMemoryPtr CopyTorchArchiveToSharedMemory(
    const std::string& name, int64_t size,
    torch::serialize::OutputArchive& archive) {
  std::stringstream serialized;
  archive.save_to(serialized);
  auto serialized_str = serialized.str();
  auto shm = std::make_unique<SharedMemory>(name);
  auto mem_buf = shm->Create(size);
  // Use the first 8 bytes to store the size of the serialized string.
  static_cast<int64_t*>(mem_buf)[0] = serialized_str.size();
  memcpy(
      (char*)mem_buf + sizeof(int64_t), serialized_str.data(),
      serialized_str.size());
  return shm;
}

static SharedMemoryPtr LoadTorchArchiveFromSharedMemory(
    const std::string& name, int64_t max_meta_size,
    torch::serialize::InputArchive& archive) {
  auto shm = std::make_unique<SharedMemory>(name);
  auto mem_buf = shm->Open(max_meta_size);
  int64_t meta_size = static_cast<int64_t*>(mem_buf)[0];
  archive.load_from(
      static_cast<const char*>(mem_buf) + sizeof(int64_t), meta_size);
  return shm;
}

static SharedMemoryPtr CopyTensorsDataToSharedMemory(
    const std::string& name, const std::vector<torch::Tensor>& tensors,
    const std::vector<torch::optional<torch::Tensor>>& optional_tensors) {
  int64_t memory_size = 0;
  for (const auto& tensor : tensors) {
    memory_size += tensor.numel() * tensor.element_size();
  }
  for (const auto& optional_tensor : optional_tensors) {
    if (optional_tensor.has_value()) {
      auto tensor = optional_tensor.value();
      memory_size += tensor.numel() * tensor.element_size();
    }
  }
  auto shm = std::make_unique<SharedMemory>(name);
  auto mem_buf = shm->Create(memory_size);
  for (auto tensor : tensors) {
    tensor = tensor.contiguous();
    int64_t size = tensor.numel() * tensor.element_size();
    memcpy(mem_buf, tensor.data_ptr(), size);
    mem_buf = static_cast<char*>(mem_buf) + size;
  }
  for (auto optional_tensor : optional_tensors) {
    if (optional_tensor.has_value()) {
      auto tensor = optional_tensor.value().contiguous();
      int64_t size = tensor.numel() * tensor.element_size();
      memcpy(mem_buf, tensor.data_ptr(), size);
      mem_buf = static_cast<char*>(mem_buf) + size;
    }
  }
  return shm;
}

/**
 * @brief Load tensors data from shared memory.
 * @param name The name of shared memory.
 * @param tensor_metas The meta info of tensors, including tensor shape and
 * dtype.
 * @param optional_tensor_metas The meta info of optional tensors, including a
 * flag indicating whether the optional tensor has value, tensor shape and
 * dtype.
 *
 * @return A tuple of shared memory holing the tensors and optional
 * tensors.
 */
static std::tuple<
    SharedMemoryPtr, std::vector<torch::Tensor>,
    std::vector<torch::optional<torch::Tensor>>>
LoadTensorsDataFromSharedMemory(
    const std::string& name,
    const std::vector<std::pair<std::vector<int64_t>, torch::ScalarType>>&
        tensor_metas,
    const std::vector<
        std::tuple<bool, std::vector<int64_t>, torch::ScalarType>>&
        optional_tensor_metas) {
  auto shm = std::make_unique<SharedMemory>(name);
  int64_t memory_size = 0;
  for (const auto& meta : tensor_metas) {
    int64_t size = std::accumulate(
        meta.first.begin(), meta.first.end(), 1, std::multiplies<int64_t>());
    memory_size += size * torch::elementSize(meta.second);
  }
  for (const auto& meta : optional_tensor_metas) {
    if (std::get<0>(meta)) {
      int64_t size = std::accumulate(
          std::get<1>(meta).begin(), std::get<1>(meta).end(), 1,
          std::multiplies<int64_t>());
      memory_size += size * torch::elementSize(std::get<2>(meta));
    }
  }
  auto mem_buf = shm->Open(memory_size);
  std::vector<torch::Tensor> tensors;
  for (const auto& meta : tensor_metas) {
    auto tensor = torch::from_blob(mem_buf, meta.first, meta.second);
    tensors.push_back(tensor);
    int64_t size = std::accumulate(
        meta.first.begin(), meta.first.end(), 1, std::multiplies<int64_t>());
    mem_buf =
        static_cast<char*>(mem_buf) + size * torch::elementSize(meta.second);
  }
  std::vector<torch::optional<torch::Tensor>> optional_tensors;
  for (const auto& meta : optional_tensor_metas) {
    if (std::get<0>(meta)) {
      auto tensor =
          torch::from_blob(mem_buf, std::get<1>(meta), std::get<2>(meta));
      optional_tensors.push_back(tensor);
      int64_t size = std::accumulate(
          std::get<1>(meta).begin(), std::get<1>(meta).end(), 1,
          std::multiplies<int64_t>());
      mem_buf = static_cast<char*>(mem_buf) +
                size * torch::elementSize(std::get<2>(meta));
    } else {
      optional_tensors.push_back(torch::nullopt);
    }
  }
  return std::make_tuple(
      std::move(shm), std::move(tensors), std::move(optional_tensors));
}

SharedMemoryTensors CopyTensorsToSharedMemory(
    const std::string& name, const std::vector<torch::Tensor>& tensors,
    const std::vector<torch::optional<torch::Tensor>>& optional_tensors,
    int64_t meta_memory_size) {
  torch::serialize::OutputArchive archive;
  archive.write("num_tensors", static_cast<int64_t>(tensors.size()));
  for (size_t i = 0; i < tensors.size(); ++i) {
    archive.write("tensor_" + std::to_string(i) + "_shape", tensors[i].sizes());
    archive.write(
        "tensor_" + std::to_string(i) + "_dtype", tensors[i].scalar_type());
  }
  archive.write(
      "num_optional_tensors", static_cast<int64_t>(optional_tensors.size()));
  for (size_t i = 0; i < optional_tensors.size(); ++i) {
    if (optional_tensors[i].has_value()) {
      archive.write(
          "optional_tensor_" + std::to_string(i) + "_has_value", true);
      archive.write(
          "optional_tensor_" + std::to_string(i) + "_shape",
          optional_tensors[i].value().sizes());
      archive.write(
          "optional_tensor_" + std::to_string(i) + "_dtype",
          optional_tensors[i].value().scalar_type());
    } else {
      archive.write(
          "optional_tensor_" + std::to_string(i) + "_has_value", false);
    }
  }
  auto meta_shm =
      CopyTorchArchiveToSharedMemory(name + "_meta", meta_memory_size, archive);
  auto data_shm =
      CopyTensorsDataToSharedMemory(name + "_data", tensors, optional_tensors);

  std::vector<torch::Tensor> ret_tensors;
  std::vector<torch::optional<torch::Tensor>> ret_optional_tensors;
  auto mem_buf = data_shm->GetMemory();
  for (auto tensor : tensors) {
    ret_tensors.push_back(
        torch::from_blob(mem_buf, tensor.sizes(), tensor.dtype()));
    int64_t size = tensor.numel() * tensor.element_size();
    mem_buf = static_cast<char*>(mem_buf) + size;
  }
  for (auto optional_tensor : optional_tensors) {
    if (optional_tensor.has_value()) {
      auto tensor = optional_tensor.value();
      ret_optional_tensors.push_back(
          torch::from_blob(mem_buf, tensor.sizes(), tensor.dtype()));
      int64_t size = tensor.numel() * tensor.element_size();
      mem_buf = static_cast<char*>(mem_buf) + size;
    } else {
      ret_optional_tensors.push_back(torch::nullopt);
    }
  }
  return std::make_tuple(
      std::move(meta_shm), std::move(data_shm), std::move(ret_tensors),
      std::move(ret_optional_tensors));
}

SharedMemoryTensors LoadTensorsFromSharedMemory(
    const std::string& name, int64_t meta_memory_size) {
  torch::serialize::InputArchive archive;
  auto meta_shm = LoadTorchArchiveFromSharedMemory(
      name + "_meta", meta_memory_size, archive);
  std::vector<std::pair<std::vector<int64_t>, torch::ScalarType>> metas;
  std::vector<std::tuple<bool, std::vector<int64_t>, torch::ScalarType>>
      optional_metas;
  int64_t num_tensors = read_from_archive(archive, "num_tensors").toInt();
  for (int64_t i = 0; i < num_tensors; ++i) {
    auto shape =
        read_from_archive(archive, "tensor_" + std::to_string(i) + "_shape")
            .toIntVector();
    auto dtype =
        read_from_archive(archive, "tensor_" + std::to_string(i) + "_dtype")
            .toScalarType();
    metas.push_back({shape, dtype});
  }
  int64_t num_optional_tensors =
      read_from_archive(archive, "num_optional_tensors").toInt();
  for (int64_t i = 0; i < num_optional_tensors; ++i) {
    bool has_value =
        read_from_archive(
            archive, "optional_tensor_" + std::to_string(i) + "_has_value")
            .toBool();
    if (has_value) {
      auto shape =
          read_from_archive(
              archive, "optional_tensor_" + std::to_string(i) + "_shape")
              .toIntVector();
      auto dtype =
          read_from_archive(
              archive, "optional_tensor_" + std::to_string(i) + "_dtype")
              .toScalarType();
      optional_metas.push_back({true, shape, dtype});
    } else {
      optional_metas.push_back({false, {}, torch::ScalarType::Undefined});
    }
  }
  SharedMemoryPtr data_shm;
  std::vector<torch::Tensor> ret_tensors;
  std::vector<torch::optional<torch::Tensor>> ret_optional_tensors;
  std::tie(data_shm, ret_tensors, ret_optional_tensors) =
      LoadTensorsDataFromSharedMemory(name + "_data", metas, optional_metas);
  return std::make_tuple(
      std::move(meta_shm), std::move(data_shm), std::move(ret_tensors),
      std::move(ret_optional_tensors));
}

}  // namespace sampling
}  // namespace graphbolt
