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
 * @file cuda/cumsum.cu
 * @brief Cumsum operators implementation on CUDA.
 */
#include <cub/cub.cuh>

#include "./common.h"

namespace graphbolt {
namespace ops {

torch::Tensor ExclusiveCumSum(torch::Tensor input) {
  auto result = torch::empty_like(input);

  AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "ExclusiveCumSum", ([&] {
                               CUB_CALL(
                                   DeviceScan::ExclusiveSum,
                                   input.data_ptr<scalar_t>(),
                                   result.data_ptr<scalar_t>(), input.size(0));
                             }));
  return result;
}

}  // namespace ops
}  // namespace graphbolt
