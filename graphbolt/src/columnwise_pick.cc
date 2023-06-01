/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/columnwise_pick.cc
 * @brief Contains the function definition for column wise pick.
 */

#include "columnwise_pick.h"
#include "random.h"
#include <chrono>
using namespace graphbolt::runtime;

namespace graphbolt {
namespace sampling {

template <typename IdxType, typename ProbType>
inline RangePickFn<IdxType> GetRangePickFn(
    const torch::optional<torch::Tensor>& probs, bool replace) {
  RangePickFn<IdxType> pick_fn;
  if (probs.has_value()) {
    ProbType* probs_data =
        reinterpret_cast<ProbType*>(probs.value().data_ptr());
    pick_fn = [probs_data, replace](
                  IdxType off, IdxType len, IdxType num_samples,
                  IdxType* out_idx) {
      RandomEngine::getInstance().Choice<IdxType, ProbType>(
          num_samples, probs_data + off, len, out_idx, replace);
      for (int64_t j = 0; j < num_samples; ++j) {
        out_idx[j] += off;
      }
    };
  } else {
    pick_fn = [replace](
                  IdxType off, IdxType len, IdxType num_samples,
                  IdxType* out_idx) {
      RandomEngine::getInstance().UniformChoice<IdxType>(
          num_samples, len, out_idx, replace);
      for (int64_t j = 0; j < num_samples; ++j) {
        out_idx[j] += off;
      }
    };
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

template <typename EtypeIdType>
int64_t GetPickNumOfOneColumnConsiderEtype(
    std::vector<int64_t>& picked_nums, int64_t seed_node_off, int64_t begin_eid,
    int64_t end_eid, int64_t* fanouts, int64_t num_fanouts,
    const EtypeIdType* type_per_edge_data, NumPickFn num_pick_fn) {
  int64_t count = 0;
  for (int64_t r = begin_eid, l = begin_eid; r < end_eid; l = r) {
    auto etype = type_per_edge_data[r];
    while (r < end_eid && type_per_edge_data[r] == etype) r++;
    auto fanout = fanouts[static_cast<int>(etype)];
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

template <typename EdgeIdType, typename EtypeIdType>
void SampleForOneColumnConsiderEtype(
    int64_t out_off, const std::vector<int64_t>& picked_nums,
    EdgeIdType* picked_indices_data, int64_t seed_node_off, int64_t begin_eid,
    int64_t end_eid, int64_t num_fanouts, const EtypeIdType* type_per_edge_data,
    RangePickFn<EdgeIdType> pick_fn) {
  for (int64_t r = begin_eid, l = begin_eid; r < end_eid; l = r) {
    auto etype = type_per_edge_data[r];
    while (r < end_eid && type_per_edge_data[r] == etype) r++;
    auto picked_num = picked_nums[seed_node_off * num_fanouts + etype];
    if (picked_num == 0) continue;

    pick_fn(l, r - l, picked_num, picked_indices_data + out_off);

    out_off += picked_num;
  }
}

template <typename NodeIdType, typename EdgeIdType, typename EtypeIdType>
c10::intrusive_ptr<SampledSubgraph> ColumnWisePickPerEtype(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<EdgeIdType> pick_fn) {
  const auto indptr = graph->CSCIndptr();
  const auto indices = graph->Indices();
  const auto type_per_edge = graph->TypePerEdge();
  const EtypeIdType* type_per_edge_data;
  if (consider_etype)
    type_per_edge_data =
        reinterpret_cast<EtypeIdType*>(type_per_edge.value().data_ptr());
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
  auto start = std::chrono::high_resolution_clock::now();

  auto fanouts_ptr = reinterpret_cast<int64_t*>(fanouts.data_ptr());
  auto columns_data = reinterpret_cast<int64_t*>(columns.data_ptr());
  auto indptr_data = reinterpret_cast<int64_t*>(indptr.data_ptr());
  auto picked_columns_ptr_data =
      reinterpret_cast<int64_t*>(picked_columns_ptr.data_ptr());

  torch::parallel_for(
      0, num_columns, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        int64_t total_count(0);
        for (int i = b; i < e; i++) {
          const auto column_id = columns_data[i];
          TORCH_CHECK(
              column_id >= 0 && column_id < graph->NumNodes(),
              "Seed node id should be in range (0, num_graph_nodes).");
          auto begin = indptr_data[column_id];
          auto end = indptr_data[column_id + 1];
          int64_t count = 0;
          if (consider_etype) {
            count = GetPickNumOfOneColumnConsiderEtype<EtypeIdType>(
                picked_nums, i, begin, end, fanouts_ptr, num_fanouts,
                type_per_edge_data, num_pick_fn);
          }
          total_count += count;
          picked_columns_ptr_data[i + 1] = total_count;
        }
        auto thread_id = torch::get_thread_num();
        block_offset[thread_id + 1] = total_count;
      });
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Num Pick Stage elapsed time: " << duration.count() << " seconds"
            << std::endl;

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
  auto picked_indices_data =
      reinterpret_cast<EdgeIdType*>(picked_indices.data_ptr());
  start = std::chrono::high_resolution_clock::now();
  torch::parallel_for(
      0, num_columns, kDefaultPickGrainSize, [&](size_t b, size_t e) {
        for (int i = b; i < e; i++) {
          const auto column_id = columns_data[i];
          auto begin = indptr_data[column_id];
          auto end = indptr_data[column_id + 1];
          auto out_off = picked_columns_ptr_data[i];
          if (consider_etype) {
            SampleForOneColumnConsiderEtype<EdgeIdType, EtypeIdType>(
                out_off, picked_nums, picked_indices_data, i, begin, end,
                num_fanouts, type_per_edge_data, pick_fn);
          }
        }
      });

  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Range Pick Stage elapsed time: " << duration.count()
            << " seconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  // 5. Select according to indices at the end of the thread.
  torch::index_select_out(picked_rows, indices, 0, picked_indices);

  if (consider_etype) {
    torch::index_select_out(
        picked_etypes.value(), type_per_edge.value(), 0, picked_indices);
  }

  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Index select Stage elapsed time: " << duration.count()
            << " seconds" << std::endl;

  // 6. Return sampled graph
  torch::optional<torch::Tensor> picked_eids = torch::nullopt;
  if (return_eids) picked_eids = std::move(picked_indices);
  return c10::make_intrusive<SampledSubgraph>(
      picked_columns_ptr, picked_rows, columns, torch::nullopt, picked_eids,
      picked_etypes);
}

template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int32_t, int32_t, uint8_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int32_t> pick_fn);

template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int32_t, int32_t, uint16_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int32_t> pick_fn);

template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int32_t, int64_t, uint8_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int64_t> pick_fn);

template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int32_t, int64_t, uint16_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int64_t> pick_fn);

template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int64_t, int64_t, uint8_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int64_t> pick_fn);

template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int64_t, int64_t, uint16_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int64_t> pick_fn);

// For test
template c10::intrusive_ptr<SampledSubgraph>
ColumnWisePickPerEtype<int64_t, int64_t, uint64_t>(
    const CSCPtr graph, const torch::Tensor& columns,
    const torch::Tensor& fanouts, const torch::optional<torch::Tensor>& probs,
    bool return_eids, bool replace, bool consider_etype, NumPickFn num_pick_fn,
    RangePickFn<int64_t> pick_fn);

c10::intrusive_ptr<SampledSubgraph> ColumnWiseSampling(
    const CSCPtr graph, const torch::Tensor& seed_nodes, const torch::Tensor& fanouts, bool replace,
    bool return_eids, bool consider_etype,
    const torch::optional<torch::Tensor>& probs) {
  auto num_pick_fn = GetNumPickFn(probs, replace);

  c10::intrusive_ptr<SampledSubgraph> ret;
  auto node_id_dtype = graph->Indices().scalar_type();
  auto edge_id_dtype = graph->CSCIndptr().scalar_type();
  auto etype_dtype =
      consider_etype ? graph->TypePerEdge().value().scalar_type() : torch::kUInt8;
  auto prob_type =
      probs.has_value() ? probs.value().scalar_type() : torch::kBool;

  ATEN_PROB_TYPE_SWITCH(prob_type, ProbType, {
    ATEN_ID_TYPE_SWITCH(node_id_dtype, NodeIdType, {
      ATEN_ID_TYPE_SWITCH(edge_id_dtype, EdgeIdType, {
        ATEN_ETYPE_TYPE_SWITCH(etype_dtype, EtypeIdType, {
          auto pick_fn = GetRangePickFn<EdgeIdType, ProbType>(probs, replace);
          ret = ColumnWisePickPerEtype<NodeIdType, EdgeIdType, EtypeIdType>(
              graph, seed_nodes, fanouts, probs, return_eids, replace,
              consider_etype, num_pick_fn, pick_fn);
        });
      });
    });
  });

  return ret;
}

}  // namespace sampling
}  // namespace graphbolt
