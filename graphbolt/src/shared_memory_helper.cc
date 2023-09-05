/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file shared_memory_helper.cc
 * @brief Share memory helper implementation.
 */
#include "./shared_memory_helper.h"

#include <graphbolt/serialize.h>
#include <graphbolt/shared_memory.h>
#include <torch/torch.h>

#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace graphbolt {
namespace sampling {

static std::string GetSharedMemoryMetadataName(const std::string& name) {
  return name + "_metadata";
}

static std::string GetSharedMemoryDataName(const std::string& name) {
  return name + "_data";
}

// To avoid unaligned memory access, we round the size of the binary buffer to
// the nearest multiple of 8 bytes.
inline static int64_t GetRoundedSize(int64_t size) {
  constexpr int64_t ALIGNED_SIZE = 8;
  return (size + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
}

SharedMemoryHelper::SharedMemoryHelper(
    const std::string& name, int64_t max_metadata_size)
    : name_(name),
      max_metadata_size_(max_metadata_size),
      metadata_shared_memory_(nullptr),
      data_shared_memory_(nullptr),
      metadata_offset_(0),
      data_offset_(0) {}

void SharedMemoryHelper::InitializeRead() {
  metadata_offset_ = 0;
  data_offset_ = 0;
  if (metadata_shared_memory_ == nullptr) {
    // Reader process opens the shared memory.
    metadata_shared_memory_ =
        std::make_unique<SharedMemory>(GetSharedMemoryMetadataName(name_));
    metadata_shared_memory_->Open(max_metadata_size_);
    auto archive = this->ReadTorchArchive();
    int64_t data_size = read_from_archive(archive, "data_size").toInt();
    data_shared_memory_ =
        std::make_unique<SharedMemory>(GetSharedMemoryDataName(name_));
    data_shared_memory_->Open(data_size);
  } else {
    // Writer process already has the shared memory.
    // Skip the first archive recording data size before read.
    this->ReadTorchArchive();
  }
}

void SharedMemoryHelper::WriteTorchArchive(
    torch::serialize::OutputArchive&& archive) {
  metadata_to_write_.emplace_back(std::move(archive));
}

torch::serialize::InputArchive SharedMemoryHelper::ReadTorchArchive() {
  auto metadata_ptr = this->GetCurrentMetadataPtr();
  int64_t metadata_size = static_cast<int64_t*>(metadata_ptr)[0];
  torch::serialize::InputArchive archive;
  archive.load_from(
      static_cast<const char*>(metadata_ptr) + sizeof(int64_t), metadata_size);
  auto rounded_size = GetRoundedSize(metadata_size);
  this->MoveMetadataPtr(sizeof(int64_t) + rounded_size);
  return archive;
}

void SharedMemoryHelper::WriteTorchTensor(
    torch::optional<torch::Tensor> tensor) {
  torch::serialize::OutputArchive archive;
  archive.write("has_value", tensor.has_value());
  if (tensor.has_value()) {
    archive.write("shape", tensor.value().sizes());
    archive.write("dtype", tensor.value().scalar_type());
  }
  this->WriteTorchArchive(std::move(archive));
  tensors_to_write_.push_back(tensor);
}

torch::optional<torch::Tensor> SharedMemoryHelper::ReadTorchTensor() {
  auto archive = this->ReadTorchArchive();
  bool has_value = read_from_archive(archive, "has_value").toBool();
  if (has_value) {
    auto shape = read_from_archive(archive, "shape").toIntVector();
    auto dtype = read_from_archive(archive, "dtype").toScalarType();
    auto data_ptr = this->GetCurrentDataPtr();
    auto tensor = torch::from_blob(data_ptr, shape, dtype);
    auto rounded_size = GetRoundedSize(tensor.numel() * tensor.element_size());
    this->MoveDataPtr(rounded_size);
    return tensor;
  } else {
    return torch::nullopt;
  }
}

void SharedMemoryHelper::WriteTorchTensorDict(
    torch::optional<torch::Dict<std::string, torch::Tensor>> tensor_dict) {
  torch::serialize::OutputArchive archive;
  if (!tensor_dict.has_value()) {
    archive.write("has_value", false);
    this->WriteTorchArchive(std::move(archive));
    return;
  }
  archive.write("has_value", true);
  auto dict_value = tensor_dict.value();
  archive.write("num_tensors", static_cast<int64_t>(dict_value.size()));
  int counter = 0;
  for (auto it = dict_value.begin(); it != dict_value.end(); ++it) {
    archive.write(std::string("key_") + std::to_string(counter), it->key());
    counter++;
  }
  this->WriteTorchArchive(std::move(archive));
  for (auto it = dict_value.begin(); it != dict_value.end(); ++it) {
    this->WriteTorchTensor(it->value());
  }
}

torch::optional<torch::Dict<std::string, torch::Tensor>>
SharedMemoryHelper::ReadTorchTensorDict() {
  auto archive = this->ReadTorchArchive();
  if (!read_from_archive(archive, "has_value").toBool()) {
    return torch::nullopt;
  }
  int64_t num_tensors = read_from_archive(archive, "num_tensors").toInt();
  torch::Dict<std::string, torch::Tensor> tensor_dict;
  for (int64_t i = 0; i < num_tensors; ++i) {
    auto key =
        read_from_archive(archive, std::string("key_") + std::to_string(i))
            .toStringRef();
    auto tensor = this->ReadTorchTensor();
    tensor_dict.insert(key, tensor.value());
  }
  return tensor_dict;
}

void SharedMemoryHelper::WriteTorchArchiveInternal(
    torch::serialize::OutputArchive& archive) {
  std::stringstream serialized;
  archive.save_to(serialized);
  auto serialized_str = serialized.str();
  auto metadata_ptr = this->GetCurrentMetadataPtr();
  static_cast<int64_t*>(metadata_ptr)[0] = serialized_str.size();
  memcpy(
      static_cast<char*>(metadata_ptr) + sizeof(int64_t), serialized_str.data(),
      serialized_str.size());
  int64_t rounded_size = GetRoundedSize(serialized_str.size());
  this->MoveMetadataPtr(sizeof(int64_t) + rounded_size);
}

void SharedMemoryHelper::WriteTorchTensorInternal(
    torch::optional<torch::Tensor> tensor) {
  if (tensor.has_value()) {
    size_t memory_size = tensor.value().numel() * tensor.value().element_size();
    auto data_ptr = this->GetCurrentDataPtr();
    auto contiguous_tensor = tensor.value().contiguous();
    memcpy(data_ptr, contiguous_tensor.data_ptr(), memory_size);
    this->MoveDataPtr(GetRoundedSize(memory_size));
  }
}

void SharedMemoryHelper::Flush() {
  // The first archive records the size of the tensor data.
  torch::serialize::OutputArchive archive;
  size_t data_size = 0;
  for (auto tensor : tensors_to_write_) {
    if (tensor.has_value()) {
      auto tensor_size = tensor.value().numel() * tensor.value().element_size();
      data_size += GetRoundedSize(tensor_size);
    }
  }
  archive.write("data_size", static_cast<int64_t>(data_size));
  metadata_shared_memory_ =
      std::make_unique<SharedMemory>(GetSharedMemoryMetadataName(name_));
  metadata_shared_memory_->Create(max_metadata_size_);
  metadata_offset_ = 0;
  this->WriteTorchArchiveInternal(archive);
  for (auto& archive : metadata_to_write_) {
    this->WriteTorchArchiveInternal(archive);
  }

  data_shared_memory_ =
      std::make_unique<SharedMemory>(GetSharedMemoryDataName(name_));
  data_shared_memory_->Create(data_size);
  data_offset_ = 0;
  for (auto tensor : tensors_to_write_) {
    this->WriteTorchTensorInternal(tensor);
  }
  metadata_to_write_.clear();
  tensors_to_write_.clear();
}

std::pair<SharedMemoryPtr, SharedMemoryPtr>
SharedMemoryHelper::ReleaseSharedMemory() {
  return std::make_pair(
      std::move(metadata_shared_memory_), std::move(data_shared_memory_));
}

}  // namespace sampling
}  // namespace graphbolt
