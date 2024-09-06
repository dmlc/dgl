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
 * @file cuda/expand_indptr.cuh
 * @brief ExpandIndptr helper class implementations on CUDA.
 */
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

}  // namespace ops
}  // namespace graphbolt
