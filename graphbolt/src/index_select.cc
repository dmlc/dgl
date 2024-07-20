/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include "./index_select.h"

#include <graphbolt/cuda_ops.h>
#include <graphbolt/fused_csc_sampling_graph.h>

#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index) {
  if (utils::is_on_gpu(index) && input.is_pinned()) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "UVAIndexSelect",
        { return UVAIndexSelectImpl(input, index); });
  }
  auto output_shape = input.sizes().vec();
  output_shape[0] = index.numel();
  auto result = torch::empty(
      output_shape,
      index.options().dtype(input.dtype()).pinned_memory(index.is_pinned()));
  return torch::index_select_out(result, input, 0, index);
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
        "dimension mismatch between input.size(i) and src.size(i), ",
        input.size(i), " != ", src.size(i), ".");
  }
  return async([=] {
    const auto row_bytes = src.slice(0, 0, 1).numel() * src.element_size();
    const auto src_ptr = reinterpret_cast<std::byte*>(src.data_ptr());
    auto input_ptr = reinterpret_cast<std::byte*>(input.data_ptr());
    AT_DISPATCH_INDEX_TYPES(
        index.scalar_type(), "ScatterAsync::index::scalar_type()", ([&] {
          const auto index_ptr = index.data_ptr<index_t>();
          torch::parallel_for(
              0, index.size(0), 64, [&](int64_t begin, int64_t end) {
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
  // @todo: The CPU supports only integer dtypes for indices tensor.
  TORCH_CHECK(
      c10::isIntegralType(indices.scalar_type(), false),
      "IndexSelectCSC is not implemented to slice noninteger types yet.");
  sampling::FusedCSCSamplingGraph g(indptr, indices);
  const auto res = g.InSubgraph(nodes);
  return std::make_tuple(res->indptr, res->indices);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> IndexSelectCSCBatched(
    torch::Tensor indptr, std::vector<torch::Tensor> indices_list,
    torch::Tensor nodes, torch::optional<int64_t> output_size) {
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
              indptr, indices_list, nodes, output_size);
        });
  }
  std::vector<torch::Tensor> results;
  torch::Tensor output_indptr;
  for (auto& indices : indices_list) {
    // @todo: The CPU supports only integer dtypes for indices tensor.
    TORCH_CHECK(
        c10::isIntegralType(indices.scalar_type(), false),
        "IndexSelectCSCBatched is not implemented to slice noninteger types "
        "yet.");
    sampling::FusedCSCSamplingGraph g(indptr, indices);
    const auto res = g.InSubgraph(nodes);
    output_indptr = res->indptr;
    results.push_back(res->indices);
  }
  return std::make_tuple(output_indptr, results);
}

}  // namespace ops
}  // namespace graphbolt
