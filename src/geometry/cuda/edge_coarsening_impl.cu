/*!
 *  Copyright (c) 2019 by Contributors
 * \file geometry/cuda/edge_coarsening_impl.cu
 * \brief Edge coarsening CUDA implementation
 */
#include <dgl/array.h>
#include <curand.h>
#include <dgl/random.h>
#include <cstdint>
#include "../geometry_op.h"
#include "../../runtime/cuda/cuda_common.h"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS
#define BLUE_P 0.53406
#define BLUE -1
#define RED -2
#define EMPTY_IDX -1
#define CURAND_CALL(func)           \
{                                   \
  curandStatus_t e = (func);        \
  CHECK(e == CURAND_STATUS_SUCCESS) \
    << "CURAND: Error at "          \
    << __FILE__ << ":" << __LINE__; \
}

namespace dgl {
namespace geometry {
namespace impl {

__device__ bool done_d;
__global__ void init_done_kernel() { done_d = true; }

template <typename IdType>
__global__ void colorize_kernel(const float *prop, int64_t num_elem, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] < 0) { // unmatched
      // result[idx] = (IdType)(ceilf(prop[idx] - 1 + BLUE_P) - 2);
      result[idx] = (prop[idx] > BLUE_P) ? RED : BLUE;
      done_d = false;
    }
  }
}

template <typename IdType>
__global__ void propose_kernel(const IdType *indptr, const IdType *indices,
                               int64_t num_elem, IdType *proposal, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] != BLUE) return;

    bool has_unmatched_neighbor = false;
    for (IdType i = indptr[idx]; i < indptr[idx + 1]; ++i) {
      auto v = indices[i];

      if (result[v] < 0)
        has_unmatched_neighbor = true;
      if (result[v] == RED) { // propose to the first red neighbor
        proposal[idx] = v;
        break;
      }
    }
    __syncthreads();
    if (!has_unmatched_neighbor)
      result[idx] = idx;
  }
}

template <typename FloatType, typename IdType>
__global__ void weighted_propose_kernel(const IdType *indptr, const IdType *indices,
                                        const FloatType *weights, int64_t num_elem,
                                        IdType *proposal, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] != BLUE) return;

    bool has_unmatched_neighbor = false;
    FloatType weight_max = 0.;
    IdType v_max = EMPTY_IDX;

    for (IdType i = indptr[idx]; i < indptr[idx + 1]; ++i) {
      auto v = indices[i];

      if (result[v] < 0)
        has_unmatched_neighbor = true;
      if (result[v] == RED && weights[i] >= weight_max) {
        v_max = v;
        weight_max = weights[i];
      }
    }

    proposal[idx] = v_max;
    if (!has_unmatched_neighbor)
      result[idx] = idx;
  }
}

template <typename IdType>
__global__ void respond_kernel(const IdType *indptr, const IdType *indices,
                               int64_t num_elem, IdType *proposal, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdType target = EMPTY_IDX;
  if (idx < num_elem) {
    if (result[idx] != RED) return;

    bool has_unmatched_neighbors = false;

    for (IdType i = indptr[idx]; i < indptr[idx + 1]; ++i) {
      auto v = indices[i];

      if (result[v] < 0)
        has_unmatched_neighbors = true;
      if (result[v] == BLUE && proposal[v] == idx) {
        // Match first blue neighbhor v which proposed to u.
        target = v;
        break;
      }
    }
    __syncthreads();

    if (target != EMPTY_IDX) {
      result[target] = min(idx, target);
      result[idx] = min(idx, target);
    }

    if (!has_unmatched_neighbors)
      result[idx] = idx;
  }
}

template <typename FloatType, typename IdType>
__global__ void weighted_respond_kernel(const IdType *indptr, const IdType *indices,
                                        const FloatType *weights, int64_t num_elem,
                                        IdType *proposal, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] != RED) return;

    bool has_unmatched_neighbors = false;
    IdType v_max = -1;
    FloatType weight_max = 0.;

    for (IdType i = indptr[idx]; i < indptr[idx + 1]; ++i) {
      auto v = indices[i];

      if (result[v] < 0)
        has_unmatched_neighbors = true;
      
      if (result[v] == BLUE && proposal[v] == idx && weights[i] >= weight_max) {
        v_max = v;
        weight_max = weights[i];
      }
    }
    if (v_max >= 0) {
      result[v_max] = min(idx, v_max);
      result[idx] = min(idx, v_max);
    }

    if (!has_unmatched_neighbors)
      result[idx] = idx;
  }
}

template<typename IdType>
bool Colorize(NDArray result, curandGenerator_t gen) {
  // initial done signal
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  CUDA_KERNEL_CALL(init_done_kernel, 1, 1, 0, thr_entry->stream);

  // generate color prop for each node
  const int64_t num_nodes = result->shape[0];
  float *prop;
  CUDA_CALL(cudaMalloc((void **)&prop, num_nodes * sizeof(float)));
  CURAND_CALL(curandGenerateUniform(gen, prop, num_nodes));
  cudaDeviceSynchronize(); // wait for random number generation finish since curand is async

  // call kernel
  IdType * result_data = static_cast<IdType*>(result->data);
  CUDA_KERNEL_CALL(colorize_kernel, BLOCKS(num_nodes), THREADS, 0, thr_entry->stream,
                   prop, num_nodes, result_data);
  
  bool done_h = false;
  CUDA_CALL(cudaMemcpyFromSymbol(&done_h, done_d, sizeof(done_h), 0, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaFree(prop));
  return done_h;
}

template<typename FloatType, typename IdType>
void Propose(const NDArray indptr, const NDArray indices,
             const NDArray weight, NDArray result, NDArray proposal) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t num_nodes = result->shape[0];
  IdType *indptr_data = static_cast<IdType*>(indptr->data);
  IdType *indices_data = static_cast<IdType*>(indices->data);
  IdType *result_data = static_cast<IdType*>(result->data);
  IdType *proposal_data = static_cast<IdType*>(proposal->data);

  if (aten::IsNullArray(weight)) {
    CUDA_KERNEL_CALL(propose_kernel, BLOCKS(num_nodes), THREADS, 0, thr_entry->stream,
                     indptr_data, indices_data, num_nodes, proposal_data, result_data);
  } else {
    FloatType *weight_data = static_cast<FloatType*>(weight->data);
    CUDA_KERNEL_CALL(weighted_propose_kernel, BLOCKS(num_nodes), THREADS, 0, thr_entry->stream,
                     indptr_data, indices_data, weight_data, num_nodes, proposal_data, result_data);
  }
}

template<typename FloatType, typename IdType>
void Respond(const NDArray indptr, const NDArray indices,
             const NDArray weight, NDArray result, NDArray proposal) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t num_nodes = result->shape[0];
  IdType *indptr_data = static_cast<IdType*>(indptr->data);
  IdType *indices_data = static_cast<IdType*>(indices->data);
  IdType *result_data = static_cast<IdType*>(result->data);
  IdType *proposal_data = static_cast<IdType*>(proposal->data);

  if (aten::IsNullArray(weight)) {
    CUDA_KERNEL_CALL(respond_kernel, BLOCKS(num_nodes), THREADS, 0, thr_entry->stream,
                     indptr_data, indices_data, num_nodes, proposal_data, result_data);
  } else {
    FloatType *weight_data = static_cast<FloatType*>(weight->data);
    CUDA_KERNEL_CALL(weighted_respond_kernel, BLOCKS(num_nodes), THREADS, 0, thr_entry->stream,
                     indptr_data, indices_data, weight_data, num_nodes, proposal_data, result_data);
  }
}

template <DLDeviceType XPU, typename FloatType, typename IdType>
void EdgeCoarsening(const NDArray indptr, const NDArray indices,
                    const NDArray weight, NDArray result) {
  // get random generator
  curandGenerator_t gen;
  uint64_t seed = dgl::RandomEngine::ThreadLocal()->RandInt(UINT64_MAX);
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));

  // create proposal tensor
  const int64_t num_nodes = result->shape[0];
  IdArray proposal = aten::Full(-1, num_nodes, sizeof(IdType) * 8, result->ctx);

  while (!Colorize<IdType>(result, gen)) {
    Propose<FloatType, IdType>(indptr, indices, weight, result, proposal);
    Respond<FloatType, IdType>(indptr, indices, weight, result, proposal);
  }
}

template void EdgeCoarsening<kDLGPU, float, int32_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, NDArray result);
template void EdgeCoarsening<kDLGPU, float, int64_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, NDArray result);
template void EdgeCoarsening<kDLGPU, double, int32_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, NDArray result);
template void EdgeCoarsening<kDLGPU, double, int64_t>(
    const NDArray indptr, const NDArray indices,
    const NDArray weight, NDArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl
