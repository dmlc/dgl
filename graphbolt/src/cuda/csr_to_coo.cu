/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/csr_to_coo.cu
 * @brief CSRToCOO operator implementation on CUDA.
 */
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <limits>

#include "./common.h"

namespace graphbolt {
namespace ops {

template <typename indices_t>
struct RepeatIndex {
  __host__ __device__ auto operator()(indices_t i) {
    return thrust::make_constant_iterator(i);
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

torch::Tensor CSRToCOO(torch::Tensor indptr, torch::ScalarType output_dtype) {
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  const auto num_rows = indptr.size(0) - 1;
  thrust::counting_iterator<int64_t> iota(0);

  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "CSRToCOOIndptr", ([&] {
        using indptr_t = scalar_t;
        auto indptr_ptr = indptr.data_ptr<indptr_t>();
        auto num_edges =
            cuda::CopyScalar{indptr.data_ptr<indptr_t>() + num_rows};
        auto csr_rows = torch::empty(
            static_cast<indptr_t>(num_edges),
            indptr.options().dtype(output_dtype));
        AT_DISPATCH_INTEGRAL_TYPES(
            output_dtype, "CSRToCOOIndices", ([&] {
              using indices_t = scalar_t;
              auto csc_rows_ptr = csr_rows.data_ptr<indices_t>();

              auto input_buffer = thrust::make_transform_iterator(
                  iota, RepeatIndex<indices_t>{});
              auto output_buffer = thrust::make_transform_iterator(
                  iota, OutputBufferIndexer<indptr_t, indices_t>{
                            indptr_ptr, csc_rows_ptr});
              auto buffer_sizes = thrust::make_transform_iterator(
                  iota, AdjacentDifference<indptr_t>{indptr_ptr});

              constexpr int64_t max_copy_at_once =
                  std::numeric_limits<int32_t>::max();
              for (int64_t i = 0; i < num_rows; i += max_copy_at_once) {
                std::size_t tmp_storage_size = 0;
                CUDA_CALL(cub::DeviceCopy::Batched(
                    nullptr, tmp_storage_size, input_buffer + i,
                    output_buffer + i, buffer_sizes + i,
                    std::min(num_rows - i, max_copy_at_once), stream));

                auto tmp_storage =
                    allocator.AllocateStorage<char>(tmp_storage_size);

                CUDA_CALL(cub::DeviceCopy::Batched(
                    tmp_storage.get(), tmp_storage_size, input_buffer + i,
                    output_buffer + i, buffer_sizes + i,
                    std::min(num_rows - i, max_copy_at_once), stream));
              }
            }));
        return csr_rows;
      }));
}

}  // namespace ops
}  // namespace graphbolt