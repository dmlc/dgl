/**
 *  Copyright (c) 2023 by Contributors
 * @file csc_sampling_graph.cc
 * @brief Source file of sampling graph.
 */

#include <graphbolt/csc_sampling_graph.h>
#include <graphbolt/serialize.h>

#include "columnwise_pick.h"
#include "macro.h"

namespace graphbolt {
namespace sampling {

CSCSamplingGraph::CSCSamplingGraph(
    torch::Tensor& indptr, torch::Tensor& indices,
    const torch::optional<torch::Tensor>& node_type_offset,
    const torch::optional<torch::Tensor>& type_per_edge)
    : indptr_(indptr),
      indices_(indices),
      node_type_offset_(node_type_offset),
      type_per_edge_(type_per_edge) {
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

c10::intrusive_ptr<SampledSubgraph> CSCSamplingGraph::SampleEtypeNeighbors(
    const torch::Tensor& seed_nodes, const torch::Tensor& fanouts, bool replace,
    bool return_eids, bool consider_etype, const torch::optional<torch::Tensor>& probs) {
  torch::Tensor picked_row_ptr, picked_cols, picked_etypes, picked_eids;
  torch::optional<torch::Tensor> picked_eids_or_null = torch::nullopt;
  if (return_eids) picked_eids_or_null = torch::tensor({}, indptr_.options());

  auto pick_fn = GetRangePickFn(probs, replace);
  auto num_pick_fn = GetNumPickFn(probs, replace);

  c10::intrusive_ptr<SampledSubgraph> ret;
  auto dtype = consider_etype ? type_per_edge_.value().scalar_type() : torch::kUInt8;
  ATEN_ETYPE_TYPE_SWITCH(
    dtype, EtypeType, {
      ret = ColumnWisePickPerEtype<EtypeType>(this, seed_nodes, fanouts, probs, return_eids, replace, consider_etype, num_pick_fn, pick_fn);
    });

  return ret;
}

void CSCSamplingGraph::Load(torch::serialize::InputArchive& archive) {
  const int64_t magic_num =
      read_from_archive(archive, "CSCSamplingGraph/magic_num").toInt();
  TORCH_CHECK(
      magic_num == kCSCSamplingGraphSerializeMagic,
      "Magic numbers mismatch when loading CSCSamplingGraph.");
  indptr_ = read_from_archive(archive, "CSCSamplingGraph/indptr").toTensor();
  indices_ = read_from_archive(archive, "CSCSamplingGraph/indices").toTensor();
  if (read_from_archive(archive, "CSCSamplingGraph/has_node_type_offset")
          .toBool()) {
    node_type_offset_ =
        read_from_archive(archive, "CSCSamplingGraph/node_type_offset")
            .toTensor();
  }
  if (read_from_archive(archive, "CSCSamplingGraph/has_type_per_edge")
          .toBool()) {
    type_per_edge_ =
        read_from_archive(archive, "CSCSamplingGraph/type_per_edge").toTensor();
  }
}

void CSCSamplingGraph::Save(torch::serialize::OutputArchive& archive) const {
  archive.write("CSCSamplingGraph/magic_num", kCSCSamplingGraphSerializeMagic);
  archive.write("CSCSamplingGraph/indptr", indptr_);
  archive.write("CSCSamplingGraph/indices", indices_);
  archive.write(
      "CSCSamplingGraph/has_node_type_offset", node_type_offset_.has_value());
  if (node_type_offset_) {
    archive.write(
        "CSCSamplingGraph/node_type_offset", node_type_offset_.value());
  }
  archive.write(
      "CSCSamplingGraph/has_type_per_edge", type_per_edge_.has_value());
  if (type_per_edge_) {
    archive.write("CSCSamplingGraph/type_per_edge", type_per_edge_.value());
  }
}

}  // namespace sampling
}  // namespace graphbolt
