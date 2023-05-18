/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/include/csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include "csc_sampling_graph.h"

#include "serialize.h"

namespace graphbolt {
namespace sampling {

void HeteroInfo::Load(torch::serialize::InputArchive& archive) {
  int64_t magic_num = 0x0;
  utils::read_from_archive(archive, "HeteroInfo/magic_num", magic_num);
  TORCH_CHECK(
      magic_num == kHeteroInfoSerializeMagic,
      "Magic numbers mismatch when loading HeteroInfo.");
  utils::read_from_archive(archive, "HeteroInfo/node_types", node_types);
  utils::read_from_archive(archive, "HeteroInfo/edge_types", edge_types);
  utils::read_from_archive(
      archive, "HeteroInfo/node_type_offset", node_type_offset);
  utils::read_from_archive(archive, "HeteroInfo/type_per_edge", type_per_edge);
}

void HeteroInfo::Save(torch::serialize::OutputArchive& archive) const {
  utils::write_to_archive(
      archive, "HeteroInfo/magic_num", kHeteroInfoSerializeMagic);
  utils::write_to_archive(archive, "HeteroInfo/node_types", node_types);
  utils::write_to_archive(archive, "HeteroInfo/edge_types", edge_types);
  utils::write_to_archive(
      archive, "HeteroInfo/node_type_offset", node_type_offset);
  utils::write_to_archive(archive, "HeteroInfo/type_per_edge", type_per_edge);
}

CSCSamplingGraph::CSCSamplingGraph(
    int64_t num_nodes, torch::Tensor& indptr, torch::Tensor& indices,
    const std::shared_ptr<HeteroInfo>& hetero_info)
    : num_nodes_(num_nodes),
      indptr_(indptr),
      indices_(indices),
      hetero_info_(hetero_info) {
  TORCH_CHECK(num_nodes >= 0);
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.size(0) == num_nodes + 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSC(
    int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices) {
  return c10::make_intrusive<CSCSamplingGraph>(
      num_nodes, indptr, indices, nullptr);
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSCWithHeteroInfo(
    int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices,
    const StringList& ntypes, const StringList& etypes,
    torch::Tensor node_type_offset, torch::Tensor type_per_edge) {
  TORCH_CHECK(node_type_offset.size(0) > 0);
  TORCH_CHECK(node_type_offset.dim() == 1);
  TORCH_CHECK(type_per_edge.size(0) > 0);
  TORCH_CHECK(type_per_edge.dim() == 1);
  TORCH_CHECK(node_type_offset.device() == type_per_edge.device());
  TORCH_CHECK(type_per_edge.device() == indices.device());
  TORCH_CHECK(!ntypes.empty());
  TORCH_CHECK(!etypes.empty());
  auto hetero_info = std::make_shared<HeteroInfo>(
      ntypes, etypes, node_type_offset, type_per_edge);

  return c10::make_intrusive<CSCSamplingGraph>(
      num_nodes, indptr, indices, hetero_info);
}

void CSCSamplingGraph::Load(torch::serialize::InputArchive& archive) {
  int64_t magic_num = 0x0;
  utils::read_from_archive(archive, "CSCSamplingGraph/magic_num", magic_num);
  TORCH_CHECK(
      magic_num == kCSCSamplingGraphSerializeMagic,
      "Magic numbers mismatch when loading CSCSamplingGraph.");
  utils::read_from_archive(archive, "CSCSamplingGraph/num_nodes", num_nodes_);
  utils::read_from_archive(archive, "CSCSamplingGraph/indptr", indptr_);
  utils::read_from_archive(archive, "CSCSamplingGraph/indices", indices_);
  bool is_heterogeneous = false;
  utils::read_from_archive(
      archive, "CSCSamplingGraph/is_hetero", is_heterogeneous);
  if (is_heterogeneous) {
    hetero_info_ = std::make_shared<HeteroInfo>();
    hetero_info_->Load(archive);
  }
}

void CSCSamplingGraph::Save(torch::serialize::OutputArchive& archive) const {
  archive.write(
      "CSCSamplingGraph/magic_num",
      torch::IValue(kCSCSamplingGraphSerializeMagic));
  archive.write("CSCSamplingGraph/num_nodes", torch::IValue(num_nodes_));
  archive.write("CSCSamplingGraph/indptr", indptr_);
  archive.write("CSCSamplingGraph/indices", indices_);
  archive.write("CSCSamplingGraph/is_hetero", torch::IValue(IsHeterogeneous()));
  if (IsHeterogeneous()) {
    hetero_info_->Save(archive);
  }
}

}  // namespace sampling
}  // namespace graphbolt

namespace torch {

serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    graphbolt::sampling::CSCSamplingGraph& graph) {
  graph.Load(archive);
  return archive;
}

serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const graphbolt::sampling::CSCSamplingGraph& graph) {
  graph.Save(archive);
  return archive;
}

}  // namespace torch
