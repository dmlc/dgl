/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/index_select_impl.cu
 * @brief Index select operator implementation on CUDA.
 */
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAStream.h>
#include <graphbolt/cuda_ops.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <numeric>

#include "./common.h"
#include "./utils.h"

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

torch::Tensor CSRToCOO(
    torch::Tensor indptr, torch::ScalarType indices_scalar_type) {
  auto allocator = cuda::GetAllocator();
  auto stream = c10::cuda::getDefaultCUDAStream();
  const auto num_rows = indptr.size(0) - 1;
  thrust::counting_iterator<int64_t> iota(0);

  return AT_DISPATCH_INTEGRAL_TYPES(
      indptr.scalar_type(), "CSCToCOOIndptr2", ([&] {
        using indptr_t = scalar_t;
        auto indptr_ptr = indptr.data_ptr<indptr_t>();
        indptr_t num_edges;
        CUDA_CALL(cudaMemcpyAsync(
            &num_edges, indptr.data_ptr<scalar_t>() + num_rows,
            sizeof(num_edges), cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));
        auto csr_rows = torch::empty(
            num_edges, indptr.options().dtype(indices_scalar_type));
        AT_DISPATCH_INTEGRAL_TYPES(
            indices_scalar_type, "CSCToCOOIndices", ([&] {
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

c10::intrusive_ptr<sampling::FusedSampledSubgraph>
SampleNeighborsWithoutReplacement(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes,
    const std::vector<int64_t>& fanouts, bool replace, bool layer,
    bool return_eids, torch::optional<torch::Tensor> type_per_edge,
    torch::optional<torch::Tensor> probs_or_mask) {
  // We assume that indptr, indices, nodes, type_per_edge and probs_or_mask
  // are all resident on the GPU. If not, it is better to first extract them.
  TORCH_CHECK(
      replace,
      "CUDA implementation of sampling with replacement is not implemented "
      "yet!");
  const auto num_rows = nodes.size(0);
  auto in_degree_and_sliced_indptr = SliceCSCIndptr(indptr, nodes);
  auto in_degree = std::get<0>(in_degree_and_sliced_indptr);
  auto output_indptr = in_degree.cumsum(0);
  auto coo_rows = CSRToCOO(output_indptr, indices.scalar_type());
}
}  //  namespace ops
}  //  namespace graphbolt
