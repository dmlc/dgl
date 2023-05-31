/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/columnwise_pick.cc
 * @brief Contains the function definition for column wise pick.
 */

#include "columnwise_pick.h"

using namespace graphbolt::runtime;

namespace graphbolt {
namespace sampling {

RangePickFn GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace) {
  RangePickFn pick_fn;
  if (probs.has_value()) {
    if (probs.value().dtype() == torch::kBool) {
      pick_fn = [probs, replace](
                    int64_t start, int64_t end, int64_t num_samples,
                    torch::Tensor& out) {
        auto local_probs = probs.value().slice(0, start, end);
        auto true_indices = local_probs.nonzero().view(-1);
        auto true_num = true_indices.size(0);
        if (replace) {
          UniformRangePickWithReplacement(0, true_num, num_samples, out);
        } else {
          UniformRangePickWithoutReplacement(0, true_num, num_samples, out);
        }

        auto index = out.clone();
        torch::index_select_out(out, true_indices, 0, index);
        out += start;
      };
    } else {
      pick_fn = [probs, replace](
                    int64_t start, int64_t end, int64_t num_samples,
                    torch::Tensor& out) {
        auto local_probs = probs.value().slice(0, start, end);
        torch::multinomial_out(out, local_probs, num_samples, replace);
        out += start;
      };
    }
  } else {
    pick_fn = replace ? UniformRangePickWithReplacement
                      : UniformRangePickWithoutReplacement;
  }
  return pick_fn;
}

NumPickFn GetNumPickFn(
    const torch::optional<torch::Tensor>& probs, bool replace) {
  NumPickFn num_pick_fn;
  if (!probs.has_value()) {
    num_pick_fn =
        [replace](int64_t start, int64_t end, int64_t num_samples) -> int64_t {
      auto len = end - start;
      const int64_t max_num_picks = (num_samples == -1) ? len : num_samples;
      if (replace) {
        return len == 0 ? 0 : max_num_picks;
      } else {
        return std::min(max_num_picks, len);
      }
    };
  } else {
    num_pick_fn = [probs, replace](
                      int64_t start, int64_t end,
                      int64_t num_samples) -> int64_t {
      const auto probs_data = probs.value().slice(0, start, end);
      auto nnz = (probs_data > 0).sum().item<int64_t>();
      const int64_t max_num_picks = (num_samples == -1) ? nnz : num_samples;
      if (replace) {
        return nnz == 0 ? 0 : max_num_picks;
      } else {
        return std::min(max_num_picks, nnz);
      }
    };
  }
  return num_pick_fn;
}

template <typename EtypeType>
int64_t GetPickNumOfOneColumnConsiderEtype(
    std::vector<int64_t>& picked_nums, int64_t seed_node_off, int64_t begin_eid,
    int64_t end_eid, const torch::Tensor& fanouts,
    const EtypeType* type_per_edge_data, NumPickFn num_pick_fn) {
  int64_t count = 0;
  auto num_fanouts = fanouts.size(0);
  for (int64_t r = begin_eid, l = begin_eid; r < end_eid; l = r) {
    auto etype = type_per_edge_data[r];
    while (r < end_eid && type_per_edge_data[r] == etype) r++;
    auto fanout = fanouts[static_cast<int>(etype)].item<int64_t>();
    auto picked_num = num_pick_fn(l, r, fanout);
    picked_nums[seed_node_off * num_fanouts + etype] = picked_num;
    count += picked_num;
  }

  return count;
}

int64_t GetPickNumOfOneColumn(
    std::vector<int64_t>& picked_nums, int64_t seed_node_off, int64_t begin_eid,
    int64_t end_eid, const torch::Tensor& fanouts, NumPickFn num_pick_fn) {
  int64_t count = 0;
  auto fanout = fanouts[0].item<int64_t>();
  auto picked_num = num_pick_fn(begin_eid, end_eid, fanout);
  picked_nums[seed_node_off] = picked_num;
  return count;
}

template <typename EtypeType>
void SampleForOneColumnConsiderEtype(
    int64_t out_off, const std::vector<int64_t>& picked_nums,
    torch::Tensor& picked_indices, int64_t seed_node_off, int64_t begin_eid,
    int64_t end_eid, int64_t num_fanouts, const EtypeType* type_per_edge_data,
    RangePickFn pick_fn) {
  for (int64_t r = begin_eid, l = begin_eid; r < end_eid; l = r) {
    auto etype = type_per_edge_data[r];
    while (r < end_eid && type_per_edge_data[r] == etype) r++;
    auto picked_num = picked_nums[seed_node_off * num_fanouts + etype];
    if (picked_num == 0) continue;
    auto out = picked_indices.slice(0, out_off, out_off + picked_num);

    pick_fn(l, r, picked_num, out);

    out_off += picked_num;
  }
}

void SampleForOneColumn(
    int64_t out_off, const std::vector<int64_t>& picked_nums,
    torch::Tensor& picked_indices, int64_t seed_node_off, int64_t begin_eid,
    int64_t end_eid, RangePickFn pick_fn) {
  auto picked_num = picked_nums[seed_node_off];
  if (picked_num == 0) return;

  auto out = picked_indices.slice(0, out_off, out_off + picked_num);

  pick_fn(begin_eid, end_eid, picked_num, out);
}

template <typename EtypeType>
c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn pick_fn) {
  const auto indptr = graph->CSCIndptr();
  const auto indices = graph->Indices();
  const auto type_per_edge = graph->TypePerEdge();
  const EtypeType* type_per_edge_data;
  if (consider_etype)
    type_per_edge_data =
        reinterpret_cast<EtypeType*>(type_per_edge.value().data_ptr());
  const auto num_columns = columns.size(0);
  const auto num_fanouts = fanouts.size(0);
  const auto sample_num = num_columns * num_fanouts;
  auto num_threads = compute_num_threads(0, num_columns, kDefaultPickGrainSize);
  // Store how much neighbors per seed node (per etype if set) will be picked.
  std::vector<int64_t> picked_nums(sample_num);
  std::vector<int64_t> block_offset(num_threads + 1);
  torch::Tensor picked_columns_ptr =
      torch::zeros({num_columns + 1}, indptr.options());
  // 1. Calculate the memory usage per seed node (per etype).
  torch::parallel_for(
      0, num_columns, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        int64_t total_count(0);
        for (int i = b; i < e; i++) {
          const auto column_id = columns[i].item<int64_t>();
          TORCH_CHECK(
              column_id >= 0 && column_id < graph->NumNodes(),
              "Seed node id should be in range (0, num_graph_nodes).");
          auto begin = indptr[column_id].item<int64_t>();
          auto end = indptr[column_id + 1].item<int64_t>();
          int64_t count = 0;
          if (consider_etype) {
            count = GetPickNumOfOneColumnConsiderEtype(
                picked_nums, i, begin, end, fanouts, type_per_edge_data,
                num_pick_fn);
          } else {
            count = GetPickNumOfOneColumn(
                picked_nums, i, begin, end, fanouts, num_pick_fn);
          }
          total_count += count;
          picked_columns_ptr[i + 1] = total_count;
        }
        auto thread_id = torch::get_thread_num();
        block_offset[thread_id + 1] = total_count;
      });

  // 2. Get total required space and columns pointer in result csc.
  if (num_threads > 1) {
    std::partial_sum(
        block_offset.begin() + 1, block_offset.end(), block_offset.begin() + 1);
    torch::parallel_for(
        0, num_columns, kDefaultPickGrainSize, [&](int64_t b, int64_t e) {
          auto tid = torch::get_thread_num();
          auto off = block_offset[tid];
          for (int64_t i = b; i < e; i++) {
            picked_columns_ptr[i + 1] += off;
          }
        });
  }

  // 3. Pre-allocate.
  auto total_length = block_offset[num_threads];
  torch::Tensor picked_indices = torch::empty({total_length}, indptr.options());
  torch::Tensor picked_rows = torch::empty({total_length}, indices.options());
  torch::optional<torch::Tensor> picked_etypes = torch::nullopt;
  if (consider_etype)
    picked_etypes =
        torch::empty({total_length}, type_per_edge.value().options());

  // 4. Sample neighbors for each seed node to fill in the `picked_indices`
  // tensor.
  torch::parallel_for(
      0, num_columns, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        for (int i = b; i < e; i++) {
          const auto column_id = columns[i].item<int64_t>();
          auto begin = indptr[column_id].item<int64_t>();
          auto end = indptr[column_id + 1].item<int64_t>();
          auto out_off = picked_columns_ptr[i].item<int64_t>();
          if (consider_etype) {
            SampleForOneColumnConsiderEtype<EtypeType>(
                out_off, picked_nums, picked_indices, i, begin, end,
                num_fanouts, type_per_edge_data, pick_fn);
          } else {
            SampleForOneColumn(
                out_off, picked_nums, picked_indices, i, begin, end, pick_fn);
          }
        }
        // 5. Select according to indices at the end of the thread.
        auto thread_id = torch::get_thread_num();
        auto begin = block_offset[thread_id];
        auto end = block_offset[thread_id + 1];

        auto picked_indices_this_thread = picked_indices.slice(0, begin, end);
        auto picked_rows_this_thread = picked_rows.slice(0, begin, end);
        torch::index_select_out(
            picked_rows_this_thread, indices, 0, picked_indices_this_thread);
        if (consider_etype) {
          auto picked_etypes_this_thread =
              picked_etypes.value().slice(0, begin, end);
          torch::index_select_out(
              picked_etypes_this_thread, type_per_edge.value(), 0,
              picked_indices_this_thread);
        }
      });

  // 6. Return sampled graph
  torch::optional<torch::Tensor> picked_eids = torch::nullopt;
  if (return_eids) picked_eids = std::move(picked_indices);
  return c10::make_intrusive<SampledSubgraph>(
      picked_columns_ptr, picked_rows, columns, torch::nullopt, picked_eids,
      picked_etypes);
}

template c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype<uint8_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn pick_fn);

template c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype<uint16_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn pick_fn);

template c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype<uint64_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn pick_fn);

}  // namespace sampling
}  // namespace graphbolt
