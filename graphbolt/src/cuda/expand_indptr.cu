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
 * @file cuda/expand_indptr.cu
 * @brief ExpandIndptr operator implementation on CUDA.
 */
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <limits>

#include "./common.h"

namespace graphbolt {
namespace ops {

template <typename indices_t, typename nodes_t>
struct RepeatIndex {
  const nodes_t* nodes;
  __host__ __device__ auto operator()(indices_t i) {
    return thrust::make_constant_iterator(nodes ? nodes[i] : i);
  }
};

template <typename indices_t, typename nodes_t>
struct IotaIndex {
  const nodes_t* nodes;
  __host__ __device__ auto operator()(indices_t i) {
    return thrust::make_counting_iterator(nodes ? nodes[i] : 0);
  }
};

template <typename indptr_t, typename indices_t>
struct OutputBufferIndexer {
  const indptr_t* indptr;
  indices_t* buffer;
  __host__ __device__ auto operator()(int64_t i) { return buffer + indptr[i]; }
};

template <typename indptr_t>
struct AdjacentDifference {
  const indptr_t* indptr;
  __host__ __device__ auto operator()(int64_t i) {
    return indptr[i + 1] - indptr[i];
  }
};

torch::Tensor ExpandIndptrImpl(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> nodes, torch::optional<int64_t> output_size,
    const bool is_edge_ids_variant) {
  if (!output_size.has_value()) {
    output_size = AT_DISPATCH_INTEGRAL_TYPES(
        indptr.scalar_type(), "ExpandIndptrIndptr[-1]", ([&]() -> int64_t {
          auto indptr_ptr = indptr.data_ptr<scalar_t>();
          auto output_size = cuda::CopyScalar{indptr_ptr + indptr.size(0) - 1};
          return static_cast<scalar_t>(output_size);
        }));
  }
  auto csc_rows =
      torch::empty(output_size.value(), indptr.options().dtype(dtype));

  AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "ExpandIndptrIndptr", ([&] {
        using indptr_t = scalar_t;
        auto indptr_ptr = indptr.data_ptr<indptr_t>();
        AT_DISPATCH_INTEGRAL_TYPES(
            dtype, "ExpandIndptrIndices", ([&] {
              using indices_t = scalar_t;
              auto csc_rows_ptr = csc_rows.data_ptr<indices_t>();

              auto nodes_dtype = nodes ? nodes.value().scalar_type() : dtype;
              AT_DISPATCH_INTEGRAL_TYPES(
                  nodes_dtype, "ExpandIndptrNodes", ([&] {
                    using nodes_t = scalar_t;
                    auto nodes_ptr =
                        nodes ? nodes.value().data_ptr<nodes_t>() : nullptr;

                    thrust::counting_iterator<int64_t> iota(0);
                    auto output_buffer = thrust::make_transform_iterator(
                        iota, OutputBufferIndexer<indptr_t, indices_t>{
                                  indptr_ptr, csc_rows_ptr});
                    auto buffer_sizes = thrust::make_transform_iterator(
                        iota, AdjacentDifference<indptr_t>{indptr_ptr});

                    const auto num_rows = indptr.size(0) - 1;
                    constexpr int64_t max_copy_at_once =
                        std::numeric_limits<int32_t>::max();

                    if (is_edge_ids_variant) {
                      auto input_buffer = thrust::make_transform_iterator(
                          iota, IotaIndex<indices_t, nodes_t>{nodes_ptr});
                      for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
                        CUB_CALL(
                            DeviceCopy::Batched, input_buffer + i,
                            output_buffer + i, buffer_sizes + i,
                            std::min(num_rows - i, max_copy_at_once));
                      }
                    } else {
                      auto input_buffer = thrust::make_transform_iterator(
                          iota, RepeatIndex<indices_t, nodes_t>{nodes_ptr});
                      for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
                        CUB_CALL(
                            DeviceCopy::Batched, input_buffer + i,
                            output_buffer + i, buffer_sizes + i,
                            std::min(num_rows - i, max_copy_at_once));
                      }
                    }
                  }));
            }));
      }));
  return csc_rows;
}

torch::Tensor ExpandIndptrImpl(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> nodes,
    torch::optional<int64_t> output_size) {
  return ExpandIndptrImpl(indptr, dtype, nodes, output_size, false);
}

torch::Tensor IndptrEdgeIdsImpl(
    torch::Tensor indptr, torch::ScalarType dtype,
    torch::optional<torch::Tensor> offset,
    torch::optional<int64_t> output_size) {
  return ExpandIndptrImpl(indptr, dtype, offset, output_size, true);
}

}  // namespace ops
}  // namespace graphbolt
