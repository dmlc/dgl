/**
 *  Copyright (c) 2019 by Contributors
 * @file geometry/cuda/edge_coarsening_impl.cu
 * @brief Edge coarsening CUDA implementation
 */
#include <curand_kernel.h>
#include <dgl/array.h>
#include <dgl/random.h>
#include <dmlc/thread_local.h>

#include <cstdint>

#include "../../array/cuda/utils.h"
#include "../../runtime/cuda/cuda_common.h"
#include "../geometry_op.h"

#define BLOCKS(N, T) (N + T - 1) / T

namespace dgl {
namespace geometry {
namespace impl {

constexpr float BLUE_P = 0.53406;
constexpr int BLUE = -1;
constexpr int RED = -2;
constexpr int EMPTY_IDX = -1;

__device__ bool done_d;
__global__ void init_done_kernel() { done_d = true; }

__global__ void generate_uniform_kernel(
    float *ret_values, size_t num, uint64_t seed) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < num) {
    curandState state;
    curand_init(seed, id, 0, &state);
    ret_values[id] = curand_uniform(&state);
  }
}

template <typename IdType>
__global__ void colorize_kernel(
    const float *prop, int64_t num_elem, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] < 0) {  // if unmatched
      result[idx] = (prop[idx] > BLUE_P) ? RED : BLUE;
      done_d = false;
    }
  }
}

template <typename FloatType, typename IdType>
__global__ void weighted_propose_kernel(
    const IdType *indptr, const IdType *indices, const FloatType *weights,
    int64_t num_elem, IdType *proposal, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] != BLUE) return;

    bool has_unmatched_neighbor = false;
    FloatType weight_max = 0.;
    IdType v_max = EMPTY_IDX;

    for (IdType i = indptr[idx]; i < indptr[idx + 1]; ++i) {
      auto v = indices[i];

      if (result[v] < 0) has_unmatched_neighbor = true;
      if (result[v] == RED && weights[i] >= weight_max) {
        v_max = v;
        weight_max = weights[i];
      }
    }

    proposal[idx] = v_max;
    if (!has_unmatched_neighbor) result[idx] = idx;
  }
}

template <typename FloatType, typename IdType>
__global__ void weighted_respond_kernel(
    const IdType *indptr, const IdType *indices, const FloatType *weights,
    int64_t num_elem, IdType *proposal, IdType *result) {
  const IdType idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elem) {
    if (result[idx] != RED) return;

    bool has_unmatched_neighbors = false;
    IdType v_max = -1;
    FloatType weight_max = 0.;

    for (IdType i = indptr[idx]; i < indptr[idx + 1]; ++i) {
      auto v = indices[i];

      if (result[v] < 0) {
        has_unmatched_neighbors = true;
      }
      if (result[v] == BLUE && proposal[v] == idx && weights[i] >= weight_max) {
        v_max = v;
        weight_max = weights[i];
      }
    }
    if (v_max >= 0) {
      result[v_max] = min(idx, v_max);
      result[idx] = min(idx, v_max);
    }

    if (!has_unmatched_neighbors) result[idx] = idx;
  }
}

/** @brief The colorize procedure. This procedure randomly marks unmarked
 * nodes with BLUE(-1) and RED(-2) and checks whether the node matching
 * process has finished.
 */
template <typename IdType>
bool Colorize(IdType *result_data, int64_t num_nodes, float *const prop) {
  // initial done signal
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  CUDA_KERNEL_CALL(init_done_kernel, 1, 1, 0, stream);

  // generate color prop for each node
  uint64_t seed = dgl::RandomEngine::ThreadLocal()->RandInt(UINT64_MAX);
  auto num_threads = cuda::FindNumThreads(num_nodes);
  auto num_blocks = cuda::FindNumBlocks<'x'>(BLOCKS(num_nodes, num_threads));
  CUDA_KERNEL_CALL(
      generate_uniform_kernel, num_blocks, num_threads, 0, stream, prop,
      num_nodes, seed);

  // call kernel
  CUDA_KERNEL_CALL(
      colorize_kernel, num_blocks, num_threads, 0, stream, prop, num_nodes,
      result_data);
  bool done_h = false;
  CUDA_CALL(cudaMemcpyFromSymbol(
      &done_h, done_d, sizeof(done_h), 0, cudaMemcpyDeviceToHost));
  return done_h;
}

/** @brief Weighted neighbor matching procedure (GPU version).
 * This implementation is from `A GPU Algorithm for Greedy Graph Matching
 * <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`__
 *
 * This algorithm has three parts: colorize, propose and respond.
 * In colorize procedure, each unmarked node will be marked as BLUE or
 * RED randomly. If all nodes are marked, finish and return.
 * In propose procedure, each BLUE node will propose to the RED
 * neighbor with the largest weight (or randomly choose one if without weight).
 * If all its neighbors are marked, mark this node with its id.
 * In respond procedure, each RED node will respond to the BLUE neighbor
 * that has proposed to it and has the largest weight. If all neighbors
 * are marked, mark this node with its id. Else match this (BLUE, RED) node
 * pair and mark them with the smaller id between them.
 */
template <DGLDeviceType XPU, typename FloatType, typename IdType>
void WeightedNeighborMatching(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  const auto &ctx = result->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  device->SetDevice(ctx);

  // create proposal tensor
  const int64_t num_nodes = result->shape[0];
  IdArray proposal = aten::Full(-1, num_nodes, sizeof(IdType) * 8, ctx);

  // get data ptrs
  IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  IdType *result_data = static_cast<IdType *>(result->data);
  IdType *proposal_data = static_cast<IdType *>(proposal->data);
  FloatType *weight_data = static_cast<FloatType *>(weight->data);

  // allocate workspace for prop used in Colorize()
  float *prop = static_cast<float *>(
      device->AllocWorkspace(ctx, num_nodes * sizeof(float)));

  auto num_threads = cuda::FindNumThreads(num_nodes);
  auto num_blocks = cuda::FindNumBlocks<'x'>(BLOCKS(num_nodes, num_threads));
  while (!Colorize<IdType>(result_data, num_nodes, prop)) {
    CUDA_KERNEL_CALL(
        weighted_propose_kernel, num_blocks, num_threads, 0, stream,
        indptr_data, indices_data, weight_data, num_nodes, proposal_data,
        result_data);
    CUDA_KERNEL_CALL(
        weighted_respond_kernel, num_blocks, num_threads, 0, stream,
        indptr_data, indices_data, weight_data, num_nodes, proposal_data,
        result_data);
  }
  device->FreeWorkspace(ctx, prop);
}
template void WeightedNeighborMatching<kDGLCUDA, float, int32_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);
template void WeightedNeighborMatching<kDGLCUDA, float, int64_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);
template void WeightedNeighborMatching<kDGLCUDA, double, int32_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);
template void WeightedNeighborMatching<kDGLCUDA, double, int64_t>(
    const aten::CSRMatrix &csr, const NDArray weight, IdArray result);

/** @brief Unweighted neighbor matching procedure (GPU version).
 * Instead of directly sample neighbors, we assign each neighbor
 * with a random weight. We use random weight for 2 reasons:
 *  1. Random sample for each node in GPU is expensive. Although
 *     we can perform a global group-wise (neighborhood of each
 *     node as a group) random permutation as in CPU version,
 *     it still cost too much compared to directly using random weights.
 *  2. Graph is sparse, thus neighborhood of each node is small,
 *     which is suitable for GPU implementation.
 */
template <DGLDeviceType XPU, typename IdType>
void NeighborMatching(const aten::CSRMatrix &csr, IdArray result) {
  const int64_t num_edges = csr.indices->shape[0];
  const auto &ctx = result->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  device->SetDevice(ctx);

  // generate random weights
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  NDArray weight = NDArray::Empty(
      {num_edges}, DGLDataType{kDGLFloat, sizeof(float) * 8, 1}, ctx);
  float *weight_data = static_cast<float *>(weight->data);
  uint64_t seed = dgl::RandomEngine::ThreadLocal()->RandInt(UINT64_MAX);
  auto num_threads = cuda::FindNumThreads(num_edges);
  auto num_blocks = cuda::FindNumBlocks<'x'>(BLOCKS(num_edges, num_threads));
  CUDA_KERNEL_CALL(
      generate_uniform_kernel, num_blocks, num_threads, 0, stream, weight_data,
      num_edges, seed);

  WeightedNeighborMatching<XPU, float, IdType>(csr, weight, result);
}
template void NeighborMatching<kDGLCUDA, int32_t>(
    const aten::CSRMatrix &csr, IdArray result);
template void NeighborMatching<kDGLCUDA, int64_t>(
    const aten::CSRMatrix &csr, IdArray result);

}  // namespace impl
}  // namespace geometry
}  // namespace dgl
