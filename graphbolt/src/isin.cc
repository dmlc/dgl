/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file isin.cc
 * @brief Isin op.
 */

#include <graphbolt/cuda_ops.h>
#include <graphbolt/isin.h>

#include "./macro.h"
#include "./utils.h"

namespace {
static constexpr int kSearchGrainSize = 4096;
}  // namespace

namespace graphbolt {
namespace sampling {

torch::Tensor IsInCPU(
    const torch::Tensor& elements, const torch::Tensor& test_elements) {
  torch::Tensor sorted_test_elements;
  std::tie(sorted_test_elements, std::ignore) = test_elements.sort(
      /*stable=*/false, /*dim=*/0, /*descending=*/false);
  torch::Tensor result = torch::empty_like(elements, torch::kBool);
  size_t num_test_elements = test_elements.size(0);
  size_t num_elements = elements.size(0);

  AT_DISPATCH_INTEGRAL_TYPES(
      elements.scalar_type(), "IsInOperation", ([&] {
        const scalar_t* elements_ptr = elements.data_ptr<scalar_t>();
        const scalar_t* sorted_test_elements_ptr =
            sorted_test_elements.data_ptr<scalar_t>();
        bool* result_ptr = result.data_ptr<bool>();
        torch::parallel_for(
            0, num_elements, kSearchGrainSize, [&](size_t start, size_t end) {
              for (auto i = start; i < end; i++) {
                result_ptr[i] = std::binary_search(
                    sorted_test_elements_ptr,
                    sorted_test_elements_ptr + num_test_elements,
                    elements_ptr[i]);
              }
            });
      }));
  return result;
}

torch::Tensor IsIn(
    const torch::Tensor& elements, const torch::Tensor& test_elements) {
  if (utils::is_on_gpu(elements) && utils::is_on_gpu(test_elements)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "IsInOperation",
        { return ops::IsIn(elements, test_elements); });
  } else {
    return IsInCPU(elements, test_elements);
  }
}

torch::Tensor IsNotInIndex(
    const torch::Tensor& elements, const torch::Tensor& test_elements) {
  auto mask = IsIn(elements, test_elements);
  if (utils::is_on_gpu(mask)) {
    GRAPHBOLT_DISPATCH_CUDA_ONLY_DEVICE(
        c10::DeviceType::CUDA, "NonzeroOperation",
        { return ops::Nonzero(mask, true); });
  }
  return torch::nonzero(torch::logical_not(mask)).squeeze(1);
}

c10::intrusive_ptr<Future<torch::Tensor>> IsNotInIndexAsync(
    const torch::Tensor& elements, const torch::Tensor& test_elements) {
  return async([=] { return IsNotInIndex(elements, test_elements); });
}

}  // namespace sampling
}  // namespace graphbolt
