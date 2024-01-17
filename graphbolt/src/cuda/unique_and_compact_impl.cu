/**
 *  Copyright (c) 2023 by Contributors
 *  Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 * @file cuda/unique_and_compact_impl.cu
 * @brief Unique and compact operator implementation on CUDA.
 */
#include <graphbolt/cuda_ops.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/logical.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

template <typename scalar_t>
struct EqualityFunc {
  const scalar_t* sorted_order;
  const scalar_t* found_locations;
  const scalar_t* searched_items;
  __host__ __device__ auto operator()(int64_t i) {
    return sorted_order[found_locations[i]] == searched_items[i];
  }
};

#define DefineCubReductionFunction(cub_reduce_fn, name)           \
  template <typename scalar_iterator_t>                           \
  auto name(const scalar_iterator_t input, int64_t size) {        \
    using scalar_t = std::remove_reference_t<decltype(input[0])>; \
    cuda::CopyScalar<scalar_t> result;                            \
    CUB_CALL(cub_reduce_fn, input, result.get(), size);           \
    return result;                                                \
  }

DefineCubReductionFunction(DeviceReduce::Max, Max);
DefineCubReductionFunction(DeviceReduce::Min, Min);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniqueAndCompact(
    const torch::Tensor src_ids, const torch::Tensor dst_ids,
    const torch::Tensor unique_dst_ids, int num_bits) {
  TORCH_CHECK(
      src_ids.scalar_type() == dst_ids.scalar_type() &&
          dst_ids.scalar_type() == unique_dst_ids.scalar_type(),
      "Dtypes of tensors passed to UniqueAndCompact need to be identical.");
  auto allocator = cuda::GetAllocator();
  auto stream = cuda::GetCurrentStream();
  return AT_DISPATCH_INTEGRAL_TYPES(
      src_ids.scalar_type(), "unique_and_compact", ([&] {
        auto src_ids_ptr = src_ids.data_ptr<scalar_t>();
        auto dst_ids_ptr = dst_ids.data_ptr<scalar_t>();
        auto unique_dst_ids_ptr = unique_dst_ids.data_ptr<scalar_t>();

        // If num_bits is not given, compute maximum vertex ids to compute
        // num_bits later to speedup the expensive sort operations.
        cuda::CopyScalar<scalar_t> max_id_src;
        cuda::CopyScalar<scalar_t> max_id_dst;
        if (num_bits == 0) {
          max_id_src = Max(src_ids_ptr, src_ids.size(0));
          max_id_dst = Max(unique_dst_ids_ptr, unique_dst_ids.size(0));
        }

        // Sort the unique_dst_ids tensor.
        auto sorted_unique_dst_ids =
            Sort<false>(unique_dst_ids_ptr, unique_dst_ids.size(0), num_bits);
        auto sorted_unique_dst_ids_ptr =
            sorted_unique_dst_ids.data_ptr<scalar_t>();

        // Mark dst nodes in the src_ids tensor.
        auto is_dst = allocator.AllocateStorage<bool>(src_ids.size(0));
        THRUST_CALL(
            binary_search, sorted_unique_dst_ids_ptr,
            sorted_unique_dst_ids_ptr + unique_dst_ids.size(0), src_ids_ptr,
            src_ids_ptr + src_ids.size(0), is_dst.get());

        // Filter the non-dst nodes in the src_ids tensor, hence only_src.
        auto only_src =
            torch::empty(src_ids.size(0), sorted_unique_dst_ids.options());
        {
          auto is_src = thrust::make_transform_iterator(
              is_dst.get(), thrust::logical_not<bool>{});
          cuda::CopyScalar<int64_t> only_src_size;
          CUB_CALL(
              DeviceSelect::Flagged, src_ids_ptr, is_src,
              only_src.data_ptr<scalar_t>(), only_src_size.get(),
              src_ids.size(0));
          stream.synchronize();
          only_src = only_src.slice(0, 0, static_cast<int64_t>(only_src_size));
        }

        // The code block above synchronizes, ensuring safe access to max_id_src
        // and max_id_dst.
        if (num_bits == 0) {
          num_bits = cuda::NumberOfBits(
              1 + std::max(
                      static_cast<scalar_t>(max_id_src),
                      static_cast<scalar_t>(max_id_dst)));
        }

        // Sort the only_src tensor so that we can unique it later.
        auto sorted_only_src = Sort<false>(
            only_src.data_ptr<scalar_t>(), only_src.size(0), num_bits);

        auto unique_only_src =
            torch::empty(only_src.size(0), src_ids.options());
        auto unique_only_src_ptr = unique_only_src.data_ptr<scalar_t>();

        {  // Compute the unique operation on the only_src tensor.
          cuda::CopyScalar<int64_t> unique_only_src_size;
          CUB_CALL(
              DeviceSelect::Unique, sorted_only_src.data_ptr<scalar_t>(),
              unique_only_src_ptr, unique_only_src_size.get(),
              only_src.size(0));
          stream.synchronize();
          unique_only_src = unique_only_src.slice(
              0, 0, static_cast<int64_t>(unique_only_src_size));
        }

        auto real_order = torch::cat({unique_dst_ids, unique_only_src});
        // Sort here so that binary search can be used to lookup new_ids.
        torch::Tensor sorted_order, new_ids;
        std::tie(sorted_order, new_ids) = Sort(real_order, num_bits);
        auto sorted_order_ptr = sorted_order.data_ptr<scalar_t>();
        auto new_ids_ptr = new_ids.data_ptr<int64_t>();
        // Holds the found locations of the src and dst ids in the sorted_order.
        // Later is used to lookup the new ids of the src_ids and dst_ids
        // tensors.
        auto new_dst_ids_loc =
            allocator.AllocateStorage<scalar_t>(dst_ids.size(0));
        THRUST_CALL(
            lower_bound, sorted_order_ptr,
            sorted_order_ptr + sorted_order.size(0), dst_ids_ptr,
            dst_ids_ptr + dst_ids.size(0), new_dst_ids_loc.get());

        cuda::CopyScalar<bool> all_exist;
        // Check if unique_dst_ids includes all dst_ids.
        if (dst_ids.size(0) > 0) {
          thrust::counting_iterator<int64_t> iota(0);
          auto equal_it = thrust::make_transform_iterator(
              iota, EqualityFunc<scalar_t>{
                        sorted_order_ptr, new_dst_ids_loc.get(), dst_ids_ptr});
          all_exist = Min(equal_it, dst_ids.size(0));
          all_exist.record();
        }

        auto new_src_ids_loc =
            allocator.AllocateStorage<scalar_t>(src_ids.size(0));
        THRUST_CALL(
            lower_bound, sorted_order_ptr,
            sorted_order_ptr + sorted_order.size(0), src_ids_ptr,
            src_ids_ptr + src_ids.size(0), new_src_ids_loc.get());

        // Finally, lookup the new compact ids of the src and dst tensors via
        // gather operations.
        auto new_src_ids = torch::empty_like(src_ids);
        THRUST_CALL(
            gather, new_src_ids_loc.get(),
            new_src_ids_loc.get() + src_ids.size(0),
            new_ids.data_ptr<int64_t>(), new_src_ids.data_ptr<scalar_t>());
        // Perform check before we gather for the dst indices.
        if (dst_ids.size(0) > 0 && !static_cast<bool>(all_exist)) {
          throw std::out_of_range("Some ids not found.");
        }
        auto new_dst_ids = torch::empty_like(dst_ids);
        THRUST_CALL(
            gather, new_dst_ids_loc.get(),
            new_dst_ids_loc.get() + dst_ids.size(0),
            new_ids.data_ptr<int64_t>(), new_dst_ids.data_ptr<scalar_t>());
        return std::make_tuple(real_order, new_src_ids, new_dst_ids);
      }));
}

}  // namespace ops
}  // namespace graphbolt
