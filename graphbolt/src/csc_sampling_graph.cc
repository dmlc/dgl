/**
 *  Copyright (c) 2023 by Contributors
 * @file csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/serialize.h>

#include "./shm/utils.h"

namespace graphbolt {
namespace sampling {

CSCSamplingGraph::CSCSamplingGraph(
    torch::Tensor& indptr, torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge,
    std::vector<std::unique_ptr<SharedMemory>>&& used_shared_memory)
    : indptr_(indptr),
      indices_(indices),
      node_type_offset_(node_type_offset),
      type_per_edge_(type_per_edge),
      used_shared_memory_(std::move(used_shared_memory)) {
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSC(
    torch::Tensor indptr, torch::Tensor indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge) {
  if (node_type_offset.has_value()) {
    auto& offset = node_type_offset.value();
    TORCH_CHECK(offset.dim() == 1);
  }
  if (type_per_edge.has_value()) {
    TORCH_CHECK(type_per_edge.value().dim() == 1);
    TORCH_CHECK(type_per_edge.value().size(0) == indices.size(0));
  }

  return c10::make_intrusive<CSCSamplingGraph>(
      indptr, indices, node_type_offset, type_per_edge);
}

void CSCSamplingGraph::Load(torch::serialize::InputArchive& archive) {
  const int64_t magic_num =
      read_from_archive(archive, "CSCSamplingGraph/magic_num").toInt();
  TORCH_CHECK(
      magic_num == kCSCSamplingGraphSerializeMagic,
      "Magic numbers mismatch when loading CSCSamplingGraph.");
  indptr_ = read_from_archive(archive, "CSCSamplingGraph/indptr").toTensor();
  indices_ = read_from_archive(archive, "CSCSamplingGraph/indices").toTensor();
}

void CSCSamplingGraph::Save(torch::serialize::OutputArchive& archive) const {
  archive.write("CSCSamplingGraph/magic_num", kCSCSamplingGraphSerializeMagic);
  archive.write("CSCSamplingGraph/indptr", indptr_);
  archive.write("CSCSamplingGraph/indices", indices_);
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::CopyToSharedMemory(
    const std::string& shared_memory_name) {
  std::vector<torch::Tensor> tensors = {indptr_, indices_};
  if (node_type_offset_.has_value()) {
    tensors.push_back(node_type_offset_.value());
    tensors.push_back(type_per_edge_.value());
  }
  std::vector<std::unique_ptr<SharedMemory>> shm;
  std::vector<torch::Tensor> shared_tensors;
  std::tie(shm, shared_tensors) = CopyTensorsToSharedMemory(
      shared_memory_name, tensors, SERIALIZED_METAINFO_SIZE_MAX);
  return c10::make_intrusive<CSCSamplingGraph>(
      shared_tensors[0], shared_tensors[1],
      shared_tensors.size() > 2
          ? torch::optional<torch::Tensor>(shared_tensors[2])
          : torch::optional<torch::Tensor>(),
      shared_tensors.size() > 3
          ? torch::optional<torch::Tensor>(shared_tensors[3])
          : torch::optional<torch::Tensor>(),
      std::move(shm));
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::LoadFromSharedMemory(
    const std::string& shared_memory_name) {
  std::vector<std::unique_ptr<SharedMemory>> shm;
  std::vector<torch::Tensor> tensors;
  std::tie(shm, tensors) = LoadTensorsFromSharedMemory(
      shared_memory_name, SERIALIZED_METAINFO_SIZE_MAX);
  return c10::make_intrusive<CSCSamplingGraph>(
      tensors[0], tensors[1],
      tensors.size() > 2 ? torch::optional<torch::Tensor>(tensors[2])
                         : torch::optional<torch::Tensor>(),
      tensors.size() > 3 ? torch::optional<torch::Tensor>(tensors[3])
                         : torch::optional<torch::Tensor>(),
      std::move(shm));
}

}  // namespace sampling
}  // namespace graphbolt
