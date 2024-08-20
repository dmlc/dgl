/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include "./index_select.h"

#include <graphbolt/cuda_ops.h>
#include <graphbolt/fused_csc_sampling_graph.h>

#include <cstring>
#include <numeric>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

constexpr int kIntGrainSize = 64;

torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index) {
  if (utils::is_on_gpu(index)) {
    if (input.is_pinned()) {
      GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
          c10::DeviceType::CUDA, "UVAIndexSelect",
          { return UVAIndexSelectImpl(input, index); });
    } else {
      return torch::index_select(input, 0, index);
    }
  }
  auto output_shape = input.sizes().vec();
  output_shape[0] = index.numel();
  auto result = torch::empty(
      output_shape, index.options()
                        .dtype(input.dtype())
                        .pinned_memory(utils::is_pinned(index)));
  auto result_ptr = reinterpret_cast<std::byte*>(result.data_ptr());
  const auto input_ptr = reinterpret_cast<std::byte*>(input.data_ptr());
  const auto row_bytes = input.slice(0, 0, 1).numel() * input.element_size();
  const auto stride = input.stride(0) * input.element_size();
  const auto num_input_rows = input.size(0);
  AT_DISPATCH_INDEX_TYPES(
      index.scalar_type(), "IndexSelect::index::scalar_type()", ([&] {
        const auto index_ptr = index.data_ptr<index_t>();
        graphbolt::parallel_for(
            0, index.size(0), kIntGrainSize, [&](int64_t begin, int64_t end) {
              for (int64_t i = begin; i < end; i++) {
                auto idx = index_ptr[i];
                if (idx < 0) idx += num_input_rows;
                if (idx < 0 || idx >= num_input_rows) {
                  // Throw IndexError via torch.
                  idx += input[num_input_rows].item<index_t>();
                }
                std::memcpy(
                    result_ptr + i * row_bytes, input_ptr + idx * stride,
                    row_bytes);
              }
            });
      }));
  return result;
}

c10::intrusive_ptr<Future<torch::Tensor>> IndexSelectAsync(
    torch::Tensor input, torch::Tensor index) {
  TORCH_CHECK(!utils::is_on_gpu(index) && !utils::is_on_gpu(input));
  return async([=] { return IndexSelect(input, index); });
}

c10::intrusive_ptr<Future<torch::Tensor>> ScatterAsync(
    torch::Tensor input, torch::Tensor index, torch::Tensor src) {
  TORCH_CHECK(
      !utils::is_on_gpu(input) && !utils::is_on_gpu(index) &&
      !utils::is_on_gpu(src));
  TORCH_CHECK(index.sizes().size() == 1, "index tensor needs to be 1d.");
  for (size_t i = 1; i < input.sizes().size(); i++) {
    TORCH_CHECK(
        input.size(i) == src.size(i),
        "dimension mismatch between input and src at ", i,
        "th dimension: ", input.size(i), " != ", src.size(i), ".");
  }
  return async([=] {
    const auto row_bytes = src.slice(0, 0, 1).numel() * src.element_size();
    const auto src_ptr = reinterpret_cast<std::byte*>(src.data_ptr());
    auto input_ptr = reinterpret_cast<std::byte*>(input.data_ptr());
    AT_DISPATCH_INDEX_TYPES(
        index.scalar_type(), "ScatterAsync::index::scalar_type()", ([&] {
          const auto index_ptr = index.data_ptr<index_t>();
          graphbolt::parallel_for(
              0, index.size(0), kIntGrainSize, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; i++) {
                  std::memcpy(
                      input_ptr + index_ptr[i] * row_bytes,
                      src_ptr + i * row_bytes, row_bytes);
                }
              });
        }));
    return input;
  });
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    torch::optional<int64_t> output_size) {
  TORCH_CHECK(
      indices.sizes().size() == 1, "IndexSelectCSC only supports 1d tensors");
  if (utils::is_on_gpu(nodes) && utils::is_accessible_from_gpu(indptr) &&
      utils::is_accessible_from_gpu(indices)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IndexSelectCSCImpl",
        { return IndexSelectCSCImpl(indptr, indices, nodes, output_size); });
  }
  auto [output_indptr, results] = IndexSelectCSCBatched(
      indptr, std::vector{indices}, nodes, false, output_size);
  return std::make_tuple(output_indptr, results.at(0));
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> IndexSelectCSCBatched(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, bool with_edge_ids,
    torch::optional<int64_t> output_size) {
  for (auto& indices : indices_list) {
    TORCH_CHECK(
        indices.sizes().size() == 1,
        "IndexSelectCSCBatched only supports 1d tensors");
  }
  if (utils::is_on_gpu(nodes) && utils::is_accessible_from_gpu(indptr) &&
      utils::are_accessible_from_gpu(indices_list)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IndexSelectCSCImpl", {
          return IndexSelectCSCBatchedImpl(
              indptr, indices_list, nodes, with_edge_ids, output_size);
        });
  }
  constexpr int kDefaultGrainSize = 128;
  const auto num_nodes = nodes.size(0);
  torch::Tensor output_indptr = torch::empty(
      {num_nodes + 1}, nodes.options().dtype(indptr.scalar_type()));
  std::vector<torch::Tensor> results;
  torch::optional<torch::Tensor> edge_ids;
  AT_DISPATCH_INDEX_TYPES(
      indptr.scalar_type(), "IndexSelectCSCBatched::indptr", ([&] {
        using indptr_t = index_t;
        const auto indptr_data = indptr.data_ptr<indptr_t>();
        auto out_indptr_data = output_indptr.data_ptr<indptr_t>();
        out_indptr_data[0] = 0;
        AT_DISPATCH_INDEX_TYPES(
            nodes.scalar_type(), "IndexSelectCSCBatched::nodes", ([&] {
              const auto nodes_data = nodes.data_ptr<index_t>();
              torch::parallel_for(
                  0, num_nodes, kDefaultGrainSize,
                  [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; i++) {
                      const auto node_id = nodes_data[i];
                      const auto degree =
                          indptr_data[node_id + 1] - indptr_data[node_id];
                      out_indptr_data[i + 1] = degree;
                    }
                  });
              output_indptr = output_indptr.cumsum(0, indptr.scalar_type());
              out_indptr_data = output_indptr.data_ptr<indptr_t>();
              TORCH_CHECK(
                  !output_size.has_value() ||
                      out_indptr_data[num_nodes] == *output_size,
                  "An incorrect output_size argument was provided.");
              output_size = out_indptr_data[num_nodes];
              for (const auto& indices : indices_list) {
                results.push_back(torch::empty(
                    *output_size,
                    nodes.options().dtype(indices.scalar_type())));
              }
              if (with_edge_ids) {
                edge_ids = torch::empty(
                    *output_size, nodes.options().dtype(indptr.scalar_type()));
              }
              torch::parallel_for(
                  0, num_nodes, kDefaultGrainSize,
                  [&](int64_t begin, int64_t end) {
                    for (int64_t i = begin; i < end; i++) {
                      const auto output_offset = out_indptr_data[i];
                      const auto numel = out_indptr_data[i + 1] - output_offset;
                      const auto input_offset = indptr_data[nodes_data[i]];
                      for (size_t tensor_id = 0;
                           tensor_id < indices_list.size(); tensor_id++) {
                        auto output = reinterpret_cast<std::byte*>(
                            results[tensor_id].data_ptr());
                        const auto input = reinterpret_cast<std::byte*>(
                            indices_list[tensor_id].data_ptr());
                        const auto element_size =
                            indices_list[tensor_id].element_size();
                        std::memcpy(
                            output + output_offset * element_size,
                            input + input_offset * element_size,
                            element_size * numel);
                      }
                      if (edge_ids.has_value()) {
                        auto output = edge_ids->data_ptr<indptr_t>();
                        std::iota(
                            output + output_offset,
                            output + output_offset + numel, input_offset);
                      }
                    }
                  });
            }));
      }));
  if (edge_ids) results.push_back(*edge_ids);
  return std::make_tuple(output_indptr, results);
}

c10::intrusive_ptr<
    Future<std::tuple<torch::Tensor, std::vector<torch::Tensor>>>>
IndexSelectCSCBatchedAsync(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, bool with_edge_ids,
    torch::optional<int64_t> output_size) {
  return async(
      [=] {
        return IndexSelectCSCBatched(
            indptr, indices_list, nodes, with_edge_ids, output_size);
      },
      utils::is_on_gpu(nodes));
}

}  // namespace ops
}  // namespace graphbolt
