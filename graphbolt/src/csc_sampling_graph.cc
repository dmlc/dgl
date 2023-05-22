/**
 *  Copyright (c) 2023 by Contributors
 * @file csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/serialize.h>

namespace graphbolt {
namespace sampling {

CSCSamplingGraph::CSCSamplingGraph(
    int64_t num_nodes, torch::Tensor& indptr, torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge)
    : num_nodes_(num_nodes),
      indptr_(indptr),
      indices_(indices),
      node_type_offset_(node_type_offset),
      type_per_edge_(type_per_edge) {
  TORCH_CHECK(num_nodes >= 0);
  TORCH_CHECK(indptr.dim() == 1);
  TORCH_CHECK(indices.dim() == 1);
  TORCH_CHECK(indptr.size(0) == num_nodes + 1);
  TORCH_CHECK(indptr.device() == indices.device());
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSC(
    int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices) {
  return c10::make_intrusive<CSCSamplingGraph>(
      num_nodes, indptr, indices, torch::nullopt, torch::nullopt);
}

c10::intrusive_ptr<CSCSamplingGraph> CSCSamplingGraph::FromCSCWithHeteroInfo(
    int64_t num_nodes, torch::Tensor indptr, torch::Tensor indices,
    torch::Tensor node_type_offset, torch::Tensor type_per_edge) {
  TORCH_CHECK(node_type_offset.dim() == 1);
  TORCH_CHECK(type_per_edge.size(0) == indices.size(0));
  TORCH_CHECK(type_per_edge.dim() == 1);
  TORCH_CHECK(node_type_offset.device() == type_per_edge.device());
  TORCH_CHECK(type_per_edge.device() == indices.device());

  return c10::make_intrusive<CSCSamplingGraph>(
      num_nodes, indptr, indices, node_type_offset, type_per_edge);
}

void CSCSamplingGraph::Load(torch::serialize::InputArchive& archive) {
  const int64_t magic_num =
      read_from_archive(archive, "CSCSamplingGraph/magic_num").toInt();
  TORCH_CHECK(
      magic_num == kCSCSamplingGraphSerializeMagic,
      "Magic numbers mismatch when loading CSCSamplingGraph.");
  num_nodes_ = read_from_archive(archive, "CSCSamplingGraph/num_nodes").toInt();
  indptr_ = read_from_archive(archive, "CSCSamplingGraph/indptr").toTensor();
  indices_ = read_from_archive(archive, "CSCSamplingGraph/indices").toTensor();
}

void CSCSamplingGraph::Save(torch::serialize::OutputArchive& archive) const {
  archive.write("CSCSamplingGraph/magic_num", kCSCSamplingGraphSerializeMagic);
  archive.write("CSCSamplingGraph/num_nodes", num_nodes_);
  archive.write("CSCSamplingGraph/indptr", indptr_);
  archive.write("CSCSamplingGraph/indices", indices_);
}

}  // namespace sampling
}  // namespace graphbolt
