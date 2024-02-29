/**
 *  Copyright (c) 2021-2022 by Contributors
 * @file graph/sampling/randomwalk_gpu.cu
 * @brief CUDA random walk sampleing
 */

#include <curand_kernel.h>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <cub/cub.cuh>
#include <tuple>
#include <utility>
#include <vector>

#include "../../../runtime/cuda/cuda_common.h"
#include "frequency_hashmap.cuh"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

template <typename IdType>
struct GraphKernelData {
  const IdType *in_ptr;
  const IdType *in_cols;
  const IdType *data;
};

template <typename IdType, typename FloatType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _RandomWalkKernel(
    const uint64_t rand_seed, const IdType *seed_data, const int64_t num_seeds,
    const IdType *metapath_data, const uint64_t max_num_steps,
    const GraphKernelData<IdType> *graphs, const FloatType *restart_prob_data,
    const int64_t restart_prob_size, const int64_t max_nodes,
    IdType *out_traces_data, IdType *out_eids_data) {
  assert(BLOCK_SIZE == blockDim.x);
  int64_t idx = blockIdx.x * TILE_SIZE + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_seeds);
  int64_t trace_length = (max_num_steps + 1);
  curandState rng;
  // reference:
  //     https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
  curand_init(rand_seed + idx, 0, 0, &rng);

  while (idx < last_idx) {
    IdType curr = seed_data[idx];
    assert(curr < max_nodes);
    IdType *traces_data_ptr = &out_traces_data[idx * trace_length];
    IdType *eids_data_ptr = &out_eids_data[idx * max_num_steps];
    *(traces_data_ptr++) = curr;
    int64_t step_idx;
    for (step_idx = 0; step_idx < max_num_steps; ++step_idx) {
      IdType metapath_id = metapath_data[step_idx];
      const GraphKernelData<IdType> &graph = graphs[metapath_id];
      const int64_t in_row_start = graph.in_ptr[curr];
      const int64_t deg = graph.in_ptr[curr + 1] - graph.in_ptr[curr];
      if (deg == 0) {  // the degree is zero
        break;
      }
      const int64_t num = curand(&rng) % deg;
      IdType pick = graph.in_cols[in_row_start + num];
      IdType eid =
          (graph.data ? graph.data[in_row_start + num] : in_row_start + num);
      *traces_data_ptr = pick;
      *eids_data_ptr = eid;
      if ((restart_prob_size > 1) &&
          (curand_uniform(&rng) < restart_prob_data[step_idx])) {
        break;
      } else if (
          (restart_prob_size == 1) &&
          (curand_uniform(&rng) < restart_prob_data[0])) {
        break;
      }
      ++traces_data_ptr;
      ++eids_data_ptr;
      curr = pick;
    }
    for (; step_idx < max_num_steps; ++step_idx) {
      *(traces_data_ptr++) = -1;
      *(eids_data_ptr++) = -1;
    }
    idx += BLOCK_SIZE;
  }
}

template <typename IdType, typename FloatType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _RandomWalkBiasedKernel(
    const uint64_t rand_seed, const IdType *seed_data, const int64_t num_seeds,
    const IdType *metapath_data, const uint64_t max_num_steps,
    const GraphKernelData<IdType> *graphs, const FloatType **probs,
    const FloatType **prob_sums, const FloatType *restart_prob_data,
    const int64_t restart_prob_size, const int64_t max_nodes,
    IdType *out_traces_data, IdType *out_eids_data) {
  assert(BLOCK_SIZE == blockDim.x);
  int64_t idx = blockIdx.x * TILE_SIZE + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_seeds);
  int64_t trace_length = (max_num_steps + 1);
  curandState rng;
  // reference:
  //     https://docs.nvidia.com/cuda/curand/device-api-overview.html#performance-notes
  curand_init(rand_seed + idx, 0, 0, &rng);

  while (idx < last_idx) {
    IdType curr = seed_data[idx];
    assert(curr < max_nodes);
    IdType *traces_data_ptr = &out_traces_data[idx * trace_length];
    IdType *eids_data_ptr = &out_eids_data[idx * max_num_steps];
    *(traces_data_ptr++) = curr;
    int64_t step_idx;
    for (step_idx = 0; step_idx < max_num_steps; ++step_idx) {
      IdType metapath_id = metapath_data[step_idx];
      const GraphKernelData<IdType> &graph = graphs[metapath_id];
      const int64_t in_row_start = graph.in_ptr[curr];
      const int64_t deg = graph.in_ptr[curr + 1] - graph.in_ptr[curr];
      if (deg == 0) {  // the degree is zero
        break;
      }

      // randomly select by weight
      const FloatType *prob_sum = prob_sums[metapath_id];
      const FloatType *prob = probs[metapath_id];
      int64_t num;
      if (prob == nullptr) {
        num = curand(&rng) % deg;
      } else {
        auto rnd_sum_w = prob_sum[curr] * curand_uniform(&rng);
        FloatType sum_w{0.};
        for (num = 0; num < deg; ++num) {
          sum_w += prob[in_row_start + num];
          if (sum_w >= rnd_sum_w) break;
        }
      }

      IdType pick = graph.in_cols[in_row_start + num];
      IdType eid =
          (graph.data ? graph.data[in_row_start + num] : in_row_start + num);
      *traces_data_ptr = pick;
      *eids_data_ptr = eid;
      if ((restart_prob_size > 1) &&
          (curand_uniform(&rng) < restart_prob_data[step_idx])) {
        break;
      } else if (
          (restart_prob_size == 1) &&
          (curand_uniform(&rng) < restart_prob_data[0])) {
        break;
      }
      ++traces_data_ptr;
      ++eids_data_ptr;
      curr = pick;
    }
    for (; step_idx < max_num_steps; ++step_idx) {
      *(traces_data_ptr++) = -1;
      *(eids_data_ptr++) = -1;
    }
    idx += BLOCK_SIZE;
  }
}

}  // namespace

// random walk for uniform choice
template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkUniform(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    FloatArray restart_prob) {
  const int64_t max_num_steps = metapath->shape[0];
  const IdType *metapath_data = static_cast<IdType *>(metapath->data);
  const int64_t begin_ntype =
      hg->meta_graph()->FindEdge(metapath_data[0]).first;
  const int64_t max_nodes = hg->NumVertices(begin_ntype);
  int64_t num_etypes = hg->NumEdgeTypes();
  auto ctx = seeds->ctx;

  const IdType *seed_data = static_cast<const IdType *>(seeds->data);
  CHECK(seeds->ndim == 1) << "seeds shape is not one dimension.";
  const int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = max_num_steps + 1;
  IdArray traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, ctx);
  IdArray eids = IdArray::Empty({num_seeds, max_num_steps}, seeds->dtype, ctx);
  IdType *traces_data = traces.Ptr<IdType>();
  IdType *eids_data = eids.Ptr<IdType>();

  std::vector<GraphKernelData<IdType>> h_graphs(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const CSRMatrix &csr = hg->GetCSRMatrix(etype);
    h_graphs[etype].in_ptr = static_cast<const IdType *>(csr.indptr->data);
    h_graphs[etype].in_cols = static_cast<const IdType *>(csr.indices->data);
    h_graphs[etype].data =
        (CSRHasData(csr) ? static_cast<const IdType *>(csr.data->data)
                         : nullptr);
  }
  // use cuda stream from local thread
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto device = DeviceAPI::Get(ctx);
  auto d_graphs = static_cast<GraphKernelData<IdType> *>(device->AllocWorkspace(
      ctx, (num_etypes) * sizeof(GraphKernelData<IdType>)));
  // copy graph metadata pointers to GPU
  device->CopyDataFromTo(
      h_graphs.data(), 0, d_graphs, 0,
      (num_etypes) * sizeof(GraphKernelData<IdType>), DGLContext{kDGLCPU, 0},
      ctx, hg->GetCSRMatrix(0).indptr->dtype);
  // copy metapath to GPU
  auto d_metapath = metapath.CopyTo(ctx);
  const IdType *d_metapath_data = static_cast<IdType *>(d_metapath->data);

  constexpr int BLOCK_SIZE = 256;
  constexpr int TILE_SIZE = BLOCK_SIZE * 4;
  dim3 block(256);
  dim3 grid((num_seeds + TILE_SIZE - 1) / TILE_SIZE);
  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  ATEN_FLOAT_TYPE_SWITCH(
      restart_prob->dtype, FloatType, "random walk GPU kernel", {
        CHECK(restart_prob->ctx.device_type == kDGLCUDA)
            << "restart prob should be in GPU.";
        CHECK(restart_prob->ndim == 1) << "restart prob dimension should be 1.";
        const FloatType *restart_prob_data = restart_prob.Ptr<FloatType>();
        const int64_t restart_prob_size = restart_prob->shape[0];
        CUDA_KERNEL_CALL(
            (_RandomWalkKernel<IdType, FloatType, BLOCK_SIZE, TILE_SIZE>), grid,
            block, 0, stream, random_seed, seed_data, num_seeds,
            d_metapath_data, max_num_steps, d_graphs, restart_prob_data,
            restart_prob_size, max_nodes, traces_data, eids_data);
      });

  device->FreeWorkspace(ctx, d_graphs);
  return std::make_pair(traces, eids);
}

/**
 * @brief Random walk for biased choice. We use inverse transform sampling to
 * choose the next step.
 */
template <DGLDeviceType XPU, typename FloatType, typename IdType>
std::pair<IdArray, IdArray> RandomWalkBiased(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob) {
  const int64_t max_num_steps = metapath->shape[0];
  const IdType *metapath_data = static_cast<IdType *>(metapath->data);
  const int64_t begin_ntype =
      hg->meta_graph()->FindEdge(metapath_data[0]).first;
  const int64_t max_nodes = hg->NumVertices(begin_ntype);
  int64_t num_etypes = hg->NumEdgeTypes();
  auto ctx = seeds->ctx;

  const IdType *seed_data = static_cast<const IdType *>(seeds->data);
  CHECK(seeds->ndim == 1) << "seeds shape is not one dimension.";
  const int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = max_num_steps + 1;
  IdArray traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, ctx);
  IdArray eids = IdArray::Empty({num_seeds, max_num_steps}, seeds->dtype, ctx);
  IdType *traces_data = traces.Ptr<IdType>();
  IdType *eids_data = eids.Ptr<IdType>();

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto device = DeviceAPI::Get(ctx);
  // new probs and prob sums pointers
  assert(num_etypes == static_cast<int64_t>(prob.size()));
  std::unique_ptr<FloatType *[]> probs(new FloatType *[prob.size()]);
  std::unique_ptr<FloatType *[]> prob_sums(new FloatType *[prob.size()]);
  std::vector<FloatArray> prob_sums_arr;
  prob_sums_arr.reserve(prob.size());

  // graphs
  std::vector<GraphKernelData<IdType>> h_graphs(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const CSRMatrix &csr = hg->GetCSRMatrix(etype);
    h_graphs[etype].in_ptr = static_cast<const IdType *>(csr.indptr->data);
    h_graphs[etype].in_cols = static_cast<const IdType *>(csr.indices->data);
    h_graphs[etype].data =
        (CSRHasData(csr) ? static_cast<const IdType *>(csr.data->data)
                         : nullptr);

    int64_t num_segments = csr.indptr->shape[0] - 1;
    // will handle empty probs in the kernel
    if (IsNullArray(prob[etype])) {
      probs[etype] = nullptr;
      prob_sums[etype] = nullptr;
      continue;
    }
    probs[etype] = prob[etype].Ptr<FloatType>();
    prob_sums_arr.push_back(
        FloatArray::Empty({num_segments}, prob[etype]->dtype, ctx));
    prob_sums[etype] = prob_sums_arr[etype].Ptr<FloatType>();

    // calculate the sum of the neighbor weights
    const IdType *d_offsets = static_cast<const IdType *>(csr.indptr->data);
    size_t temp_storage_size = 0;
    CUDA_CALL(cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_size, probs[etype], prob_sums[etype],
        num_segments, d_offsets, d_offsets + 1, stream));
    void *temp_storage = device->AllocWorkspace(ctx, temp_storage_size);
    CUDA_CALL(cub::DeviceSegmentedReduce::Sum(
        temp_storage, temp_storage_size, probs[etype], prob_sums[etype],
        num_segments, d_offsets, d_offsets + 1, stream));
    device->FreeWorkspace(ctx, temp_storage);
  }

  // copy graph metadata pointers to GPU
  auto d_graphs = static_cast<GraphKernelData<IdType> *>(device->AllocWorkspace(
      ctx, (num_etypes) * sizeof(GraphKernelData<IdType>)));
  device->CopyDataFromTo(
      h_graphs.data(), 0, d_graphs, 0,
      (num_etypes) * sizeof(GraphKernelData<IdType>), DGLContext{kDGLCPU, 0},
      ctx, hg->GetCSRMatrix(0).indptr->dtype);
  // copy probs pointers to GPU
  const FloatType **probs_dev = static_cast<const FloatType **>(
      device->AllocWorkspace(ctx, num_etypes * sizeof(FloatType *)));
  device->CopyDataFromTo(
      probs.get(), 0, probs_dev, 0, (num_etypes) * sizeof(FloatType *),
      DGLContext{kDGLCPU, 0}, ctx, prob[0]->dtype);
  // copy probs_sum pointers to GPU
  const FloatType **prob_sums_dev = static_cast<const FloatType **>(
      device->AllocWorkspace(ctx, num_etypes * sizeof(FloatType *)));
  device->CopyDataFromTo(
      prob_sums.get(), 0, prob_sums_dev, 0, (num_etypes) * sizeof(FloatType *),
      DGLContext{kDGLCPU, 0}, ctx, prob[0]->dtype);
  // copy metapath to GPU
  auto d_metapath = metapath.CopyTo(ctx);
  const IdType *d_metapath_data = static_cast<IdType *>(d_metapath->data);

  constexpr int BLOCK_SIZE = 256;
  constexpr int TILE_SIZE = BLOCK_SIZE * 4;
  dim3 block(256);
  dim3 grid((num_seeds + TILE_SIZE - 1) / TILE_SIZE);
  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  CHECK(restart_prob->ctx.device_type == kDGLCUDA)
      << "restart prob should be in GPU.";
  CHECK(restart_prob->ndim == 1) << "restart prob dimension should be 1.";
  const FloatType *restart_prob_data = restart_prob.Ptr<FloatType>();
  const int64_t restart_prob_size = restart_prob->shape[0];
  CUDA_KERNEL_CALL(
      (_RandomWalkBiasedKernel<IdType, FloatType, BLOCK_SIZE, TILE_SIZE>), grid,
      block, 0, stream, random_seed, seed_data, num_seeds, d_metapath_data,
      max_num_steps, d_graphs, probs_dev, prob_sums_dev, restart_prob_data,
      restart_prob_size, max_nodes, traces_data, eids_data);

  device->FreeWorkspace(ctx, d_graphs);
  device->FreeWorkspace(ctx, probs_dev);
  device->FreeWorkspace(ctx, prob_sums_dev);
  return std::make_pair(traces, eids);
}

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalk(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob) {
  bool isUniform = true;
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      isUniform = false;
      break;
    }
  }

  auto restart_prob =
      NDArray::Empty({0}, DGLDataType{kDGLFloat, 32, 1}, DGLContext{XPU, 0});
  if (!isUniform) {
    std::pair<IdArray, IdArray> ret;
    ATEN_FLOAT_TYPE_SWITCH(prob[0]->dtype, FloatType, "probability", {
      ret = RandomWalkBiased<XPU, FloatType, IdType>(
          hg, seeds, metapath, prob, restart_prob);
    });
    return ret;
  } else {
    return RandomWalkUniform<XPU, IdType>(hg, seeds, metapath, restart_prob);
  }
}

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob) {
  bool isUniform = true;
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      isUniform = false;
      break;
    }
  }

  auto device_ctx = seeds->ctx;
  auto restart_prob_array =
      NDArray::Empty({1}, DGLDataType{kDGLFloat, 64, 1}, device_ctx);
  auto device = dgl::runtime::DeviceAPI::Get(device_ctx);

  // use cuda stream from local thread
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  device->CopyDataFromTo(
      &restart_prob, 0, restart_prob_array.Ptr<double>(), 0, sizeof(double),
      DGLContext{kDGLCPU, 0}, device_ctx, restart_prob_array->dtype);
  device->StreamSync(device_ctx, stream);

  if (!isUniform) {
    std::pair<IdArray, IdArray> ret;
    ATEN_FLOAT_TYPE_SWITCH(prob[0]->dtype, FloatType, "probability", {
      ret = RandomWalkBiased<XPU, FloatType, IdType>(
          hg, seeds, metapath, prob, restart_prob_array);
    });
    return ret;
  } else {
    return RandomWalkUniform<XPU, IdType>(
        hg, seeds, metapath, restart_prob_array);
  }
}

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob) {
  bool isUniform = true;
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      isUniform = false;
      break;
    }
  }

  if (!isUniform) {
    std::pair<IdArray, IdArray> ret;
    ATEN_FLOAT_TYPE_SWITCH(prob[0]->dtype, FloatType, "probability", {
      ret = RandomWalkBiased<XPU, FloatType, IdType>(
          hg, seeds, metapath, prob, restart_prob);
    });
    return ret;
  } else {
    return RandomWalkUniform<XPU, IdType>(hg, seeds, metapath, restart_prob);
  }
}

template <DGLDeviceType XPU, typename IdxType>
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k) {
  CHECK(src->ctx.device_type == kDGLCUDA) << "IdArray needs be on GPU!";
  const IdxType *src_data = src.Ptr<IdxType>();
  const IdxType *dst_data = dst.Ptr<IdxType>();
  const int64_t num_dst_nodes = (dst->shape[0] / num_samples_per_node);
  auto ctx = src->ctx;
  // use cuda stream from local thread
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto frequency_hashmap = FrequencyHashmap<IdxType>(
      num_dst_nodes, num_samples_per_node, ctx, stream);
  auto ret = frequency_hashmap.Topk(
      src_data, dst_data, src->dtype, src->shape[0], num_samples_per_node, k);
  return ret;
}

template std::pair<IdArray, IdArray> RandomWalk<kDGLCUDA, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template std::pair<IdArray, IdArray> RandomWalk<kDGLCUDA, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob);

template std::pair<IdArray, IdArray> RandomWalkWithRestart<kDGLCUDA, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob);
template std::pair<IdArray, IdArray> RandomWalkWithRestart<kDGLCUDA, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, double restart_prob);

template std::pair<IdArray, IdArray>
RandomWalkWithStepwiseRestart<kDGLCUDA, int32_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob);
template std::pair<IdArray, IdArray>
RandomWalkWithStepwiseRestart<kDGLCUDA, int64_t>(
    const HeteroGraphPtr hg, const IdArray seeds, const TypeArray metapath,
    const std::vector<FloatArray> &prob, FloatArray restart_prob);

template std::tuple<IdArray, IdArray, IdArray>
SelectPinSageNeighbors<kDGLCUDA, int32_t>(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);
template std::tuple<IdArray, IdArray, IdArray>
SelectPinSageNeighbors<kDGLCUDA, int64_t>(
    const IdArray src, const IdArray dst, const int64_t num_samples_per_node,
    const int64_t k);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
