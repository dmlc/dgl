/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/cumsum.cu
 * @brief Cumsum operators implementation on CUDA.
 */
#include <cub/cub.cuh>

#include "./common.h"

namespace graphbolt {
namespace ops {

torch::Tensor ExclusiveCumSum(torch::Tensor input) {
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  auto result = torch::empty_like(input);

  AT_DISPATCH_INTEGRAL_TYPES(
      input.scalar_type(), "ExclusiveCumSum", ([&] {
        CUB_CALL(
            cub::DeviceScan::ExclusiveSum, input.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(), input.size(0), stream);
      }));
  return result;
}

}  // namespace ops
}  // namespace graphbolt
