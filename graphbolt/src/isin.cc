/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file isin.cc
 * @brief Isin op.
 */

#include <graphbolt/isin.h>

namespace {
static constexpr int kSearchGrainSize = 4096;
}  // namespace

namespace graphbolt {
namespace sampling {

torch::Tensor IsIn(
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
}  // namespace sampling
}  // namespace graphbolt
