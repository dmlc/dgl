/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file cuda/sort_impl.cu
 * @brief Sort implementation on CUDA.
 */
#include <c10/core/ScalarType.h>

#include <cub/cub.cuh>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

template <bool return_original_positions, typename scalar_t>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(const scalar_t* input_keys, int64_t num_items, int num_bits) {
  const auto options = torch::TensorOptions().device(c10::DeviceType::CUDA);
  constexpr c10::ScalarType dtype = c10::CppTypeToScalarType<scalar_t>::value;
  auto sorted_array = torch::empty(num_items, options.dtype(dtype));
  auto sorted_keys = sorted_array.data_ptr<scalar_t>();
  if (num_bits == 0) {
    num_bits = sizeof(scalar_t) * 8;
  }

  if constexpr (return_original_positions) {
    // We utilize int64_t for the values array. (torch::kLong == int64_t)
    auto original_idx = torch::arange(num_items, options.dtype(torch::kLong));
    auto sorted_idx = torch::empty_like(original_idx);
    const int64_t* input_values = original_idx.data_ptr<int64_t>();
    int64_t* sorted_values = sorted_idx.data_ptr<int64_t>();
    CUB_CALL(
        DeviceRadixSort::SortPairs, input_keys, sorted_keys, input_values,
        sorted_values, num_items, 0, num_bits);
    return std::make_pair(sorted_array, sorted_idx);
  } else {
    CUB_CALL(
        DeviceRadixSort::SortKeys, input_keys, sorted_keys, num_items, 0,
        num_bits);
    return sorted_array;
  }
}

template <bool return_original_positions>
std::conditional_t<
    return_original_positions, std::pair<torch::Tensor, torch::Tensor>,
    torch::Tensor>
Sort(torch::Tensor input, int num_bits) {
  return AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "SortImpl", ([&] {
                                      return Sort<return_original_positions>(
                                          input.data_ptr<scalar_t>(),
                                          input.size(0), num_bits);
                                    }));
}

template torch::Tensor Sort<false>(torch::Tensor input, int num_bits);
template std::pair<torch::Tensor, torch::Tensor> Sort<true>(
    torch::Tensor input, int num_bits);

}  //  namespace ops
}  //  namespace graphbolt
