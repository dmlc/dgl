/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.cc
 * @brief Index select operators.
 */
#include <aio.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <graphbolt/cuda_ops.h>
#include <graphbolt/fused_csc_sampling_graph.h>

#include <fstream>

#include "./cnpy.h"
#include "./macro.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index) {
  if (input.is_pinned() &&
      (index.is_pinned() || index.device().type() == c10::DeviceType::CUDA)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "UVAIndexSelect",
        { return UVAIndexSelectImpl(input, index); });
  }
  return input.index({index.to(torch::kLong)});
}

torch::Tensor DiskIndexSelect(std::string path, torch::Tensor index) {
  cnpy::NpyArray arr(path);
  // arr.print_npy_header();
  // return arr.index_select_all({index.to(torch::kLong)});
  // return arr.index_select_pread({index.to(torch::kLong)});
  // return arr.index_select_pread_single({index.to(torch::kLong)});
  // return arr.index_select_aio({index.to(torch::kLong)});
  return arr.index_select_iouring({index.to(torch::kLong)});
}

torch::Tensor DiskFeatureSize(std::string path) {
  cnpy::NpyArray arr(path);
  return arr.feature_size();
}

std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes) {
  TORCH_CHECK(
      indices.sizes().size() == 1, "IndexSelectCSC only supports 1d tensors");
  if (utils::is_accessible_from_gpu(indptr) &&
      utils::is_accessible_from_gpu(indices) &&
      utils::is_accessible_from_gpu(nodes)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IndexSelectCSCImpl",
        { return IndexSelectCSCImpl(indptr, indices, nodes); });
  }
  // @todo: The CPU supports only integer dtypes for indices tensor.
  TORCH_CHECK(
      c10::isIntegralType(indices.scalar_type(), false),
      "IndexSelectCSC is not implemented to slice noninteger types yet.");
  sampling::FusedCSCSamplingGraph g(indptr, indices);
  const auto res = g.InSubgraph(nodes);
  return std::make_tuple(res->indptr, res->indices);
}

}  // namespace ops
}  // namespace graphbolt
