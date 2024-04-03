/**
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/gather.cu
 * @brief Gather operators implementation on CUDA.
 */
#include <thrust/gather.h>

#include "./common.h"

namespace graphbolt {
namespace ops {

torch::Tensor Gather(
    torch::Tensor input, torch::Tensor index,
    torch::optional<torch::ScalarType> dtype) {
  if (!dtype.has_value()) dtype = input.scalar_type();
  auto output = torch::empty(index.sizes(), index.options().dtype(*dtype));
  AT_DISPATCH_INDEX_TYPES(
      index.scalar_type(), "GatherIndexType", ([&] {
        AT_DISPATCH_INTEGRAL_TYPES(
            input.scalar_type(), "GatherInputType", ([&] {
              using input_t = scalar_t;
              AT_DISPATCH_INTEGRAL_TYPES(*dtype, "GatherOutputType", ([&] {
                using output_t = scalar_t;
                THRUST_CALL(
                    gather, index.data_ptr<index_t>(),
                    index.data_ptr<index_t>() + index.size(0),
                    input.data_ptr<input_t>(), output.data_ptr<output_t>());
              }));
            }));
      }));
  return output;
}

}  // namespace ops
}  // namespace graphbolt
