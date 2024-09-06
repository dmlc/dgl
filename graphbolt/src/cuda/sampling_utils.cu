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
 * @file cuda/sampling_utils.cu
 * @brief Sampling utility function implementations on CUDA.
 */
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cub/cub.cuh>

#include "./common.h"
#include "./utils.h"

namespace graphbolt {
namespace ops {

// Given rows and indptr, computes:
// inrow_indptr[i] = indptr[rows[i]];
// in_degree[i] = indptr[rows[i] + 1] - indptr[rows[i]];
template <typename indptr_t, typename nodes_t>
struct SliceFunc {
  const nodes_t* rows;
  const indptr_t* indptr;
  indptr_t* in_degree;
  indptr_t* inrow_indptr;
  __host__ __device__ auto operator()(int64_t tIdx) {
    const auto out_row = rows[tIdx];
    const auto indptr_val = indptr[out_row];
    const auto degree = indptr[out_row + 1] - indptr_val;
    in_degree[tIdx] = degree;
    inrow_indptr[tIdx] = indptr_val;
  }
};

// Returns (indptr[nodes + 1] - indptr[nodes], indptr[nodes])
std::tuple<torch::Tensor, torch::Tensor> SliceCSCIndptr(
    torch::Tensor indptr, torch::optional<torch::Tensor> nodes_optional) {
  if (nodes_optional.has_value()) {
    auto nodes = nodes_optional.value();
    const int64_t num_nodes = nodes.size(0);
    // Read indptr only once in case it is pinned and access is slow.
    auto sliced_indptr =
        torch::empty(num_nodes, nodes.options().dtype(indptr.scalar_type()));
    // compute in-degrees
    auto in_degree = torch::empty(
        num_nodes + 1, nodes.options().dtype(indptr.scalar_type()));
    thrust::counting_iterator<int64_t> iota(0);
    AT_DISPATCH_INTEGRAL_TYPES(
        indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
          using indptr_t = scalar_t;
          AT_DISPATCH_INDEX_TYPES(
              nodes.scalar_type(), "IndexSelectCSCNodes", ([&] {
                using nodes_t = index_t;
                THRUST_CALL(
                    for_each, iota, iota + num_nodes,
                    SliceFunc<indptr_t, nodes_t>{
                        nodes.data_ptr<nodes_t>(), indptr.data_ptr<indptr_t>(),
                        in_degree.data_ptr<indptr_t>(),
                        sliced_indptr.data_ptr<indptr_t>()});
              }));
        }));
    return {in_degree, sliced_indptr};
  } else {
    const int64_t num_nodes = indptr.size(0) - 1;
    auto sliced_indptr = indptr.slice(0, 0, num_nodes);
    auto in_degree = torch::empty(
        num_nodes + 2, indptr.options().dtype(indptr.scalar_type()));
    AT_DISPATCH_INTEGRAL_TYPES(
        indptr.scalar_type(), "IndexSelectCSCIndptr", ([&] {
          using indptr_t = scalar_t;
          CUB_CALL(
              DeviceAdjacentDifference::SubtractLeftCopy,
              indptr.data_ptr<indptr_t>(), in_degree.data_ptr<indptr_t>(),
              num_nodes + 1, cub::Difference{});
        }));
    in_degree = in_degree.slice(0, 1);
    return {in_degree, sliced_indptr};
  }
}

template <typename indptr_t, typename etype_t>
struct EdgeTypeSearch {
  const indptr_t* sub_indptr;
  const indptr_t* sliced_indptr;
  const etype_t* etypes;
  int64_t num_fanouts;
  int64_t num_rows;
  indptr_t* new_sub_indptr;
  indptr_t* new_sliced_indptr;
  __host__ __device__ auto operator()(int64_t i) {
    const auto homo_i = i / num_fanouts;
    const auto indptr_i = sub_indptr[homo_i];
    const auto degree = sub_indptr[homo_i + 1] - indptr_i;
    const etype_t etype = i % num_fanouts;
    auto offset = cub::LowerBound(etypes + indptr_i, degree, etype);
    new_sub_indptr[i] = indptr_i + offset;
    new_sliced_indptr[i] = sliced_indptr[homo_i] + offset;
    if (i == num_rows - 1) new_sub_indptr[num_rows] = indptr_i + degree;
  }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SliceCSCIndptrHetero(
    torch::Tensor sub_indptr, torch::Tensor etypes, torch::Tensor sliced_indptr,
    int64_t num_fanouts) {
  auto num_rows = (sub_indptr.size(0) - 1) * num_fanouts;
  auto new_sub_indptr = torch::empty(num_rows + 1, sub_indptr.options());
  auto new_indegree = torch::empty(num_rows + 2, sub_indptr.options());
  auto new_sliced_indptr = torch::empty(num_rows, sliced_indptr.options());
  thrust::counting_iterator<int64_t> iota(0);
  AT_DISPATCH_INTEGRAL_TYPES(
      sub_indptr.scalar_type(), "SliceCSCIndptrHeteroIndptr", ([&] {
        using indptr_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(
            etypes.scalar_type(), "SliceCSCIndptrHeteroTypePerEdge", ([&] {
              using etype_t = scalar_t;
              THRUST_CALL(
                  for_each, iota, iota + num_rows,
                  EdgeTypeSearch<indptr_t, etype_t>{
                      sub_indptr.data_ptr<indptr_t>(),
                      sliced_indptr.data_ptr<indptr_t>(),
                      etypes.data_ptr<etype_t>(), num_fanouts, num_rows,
                      new_sub_indptr.data_ptr<indptr_t>(),
                      new_sliced_indptr.data_ptr<indptr_t>()});
            }));
        CUB_CALL(
            DeviceAdjacentDifference::SubtractLeftCopy,
            new_sub_indptr.data_ptr<indptr_t>(),
            new_indegree.data_ptr<indptr_t>(), num_rows + 1, cub::Difference{});
      }));
  // Discard the first element of the SubtractLeftCopy result and ensure that
  // new_indegree tensor has size num_rows + 1 so that its ExclusiveCumSum is
  // directly equivalent to new_sub_indptr.
  // Equivalent to new_indegree = new_indegree[1:] in Python.
  new_indegree = new_indegree.slice(0, 1);
  return {new_sub_indptr, new_indegree, new_sliced_indptr};
}

}  //  namespace ops
}  //  namespace graphbolt
