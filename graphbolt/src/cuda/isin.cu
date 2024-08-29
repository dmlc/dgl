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
 * @file cuda/isin.cu
 * @brief IsIn operator implementation on CUDA.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/remove.h>

#include "./common.h"

namespace graphbolt {
namespace ops {

torch::Tensor IsIn(torch::Tensor elements, torch::Tensor test_elements) {
  auto sorted_test_elements = Sort<false>(test_elements);
  auto result = torch::empty_like(elements, torch::kBool);

  AT_DISPATCH_INTEGRAL_TYPES(
      elements.scalar_type(), "IsInOperation", ([&] {
        THRUST_CALL(
            binary_search, sorted_test_elements.data_ptr<scalar_t>(),
            sorted_test_elements.data_ptr<scalar_t>() +
                sorted_test_elements.size(0),
            elements.data_ptr<scalar_t>(),
            elements.data_ptr<scalar_t>() + elements.size(0),
            result.data_ptr<bool>());
      }));
  return result;
}

torch::Tensor Nonzero(torch::Tensor mask, bool logical_not) {
  thrust::counting_iterator<int64_t> iota(0);
  auto result = torch::empty_like(mask, torch::kInt64);
  auto mask_ptr = mask.data_ptr<bool>();
  auto result_ptr = result.data_ptr<int64_t>();
  int64_t* result_end;
  if (logical_not) {
    result_end = THRUST_CALL(
        remove_copy_if, iota, iota + mask.numel(), mask_ptr, result_ptr,
        thrust::identity<bool>());
  } else {
    result_end = THRUST_CALL(
        copy_if, iota, iota + mask.numel(), mask_ptr, result_ptr,
        thrust::identity<bool>());
  }
  return result.slice(0, 0, result_end - result_ptr);
}

}  // namespace ops
}  // namespace graphbolt
