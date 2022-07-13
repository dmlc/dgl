/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/sampling/randomwalk_gpu.cu
 * \brief DGL sampler 
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/device_api.h>
#include <dgl/random.h>
#include <curand_kernel.h>
#include <vector>
#include <utility>
#include <tuple>

#include "../../../runtime/cuda/cuda_common.h"
#include "frequency_hashmap.cuh"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {

template<typename IdType>
struct GraphKernelData {
  const IdType *in_ptr;
  const IdType *in_cols;
  const IdType *data;
};

template<typename IdType, typename FloatType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _RandomWalkKernel(
    const uint64_t rand_seed, const IdType *seed_data, const int64_t num_seeds,
    const IdType* metapath_data, const uint64_t max_num_steps,
    const GraphKernelData<IdType>* graphs,
    const FloatType* restart_prob_data,
    const int64_t restart_prob_size,
    const int64_t max_nodes,
    IdType *out_traces_data,
    IdType *out_eids_data) {
  assert(BLOCK_SIZE == blockDim.x);
  int64_t idx = blockIdx.x * TILE_SIZE + threadIdx.x;
  int64_t last_idx = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_seeds);
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
      IdType eid = (graph.data? graph.data[in_row_start + num] : in_row_start + num);
      *traces_data_ptr = pick;
      *eids_data_ptr = eid;
      if ((restart_prob_size > 1) && (curand_uniform(&rng) < restart_prob_data[step_idx])) {
        break;
      } else if ((restart_prob_size == 1) && (curand_uniform(&rng) < restart_prob_data[0])) {
        break;
      }
      ++traces_data_ptr; ++eids_data_ptr;
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
template<DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkUniform(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    FloatArray restart_prob) {
  const int64_t max_num_steps = metapath->shape[0];
  const IdType *metapath_data = static_cast<IdType *>(metapath->data);
  const int64_t begin_ntype = hg->meta_graph()->FindEdge(metapath_data[0]).first;
  const int64_t max_nodes = hg->NumVertices(begin_ntype);
  int64_t num_etypes = hg->NumEdgeTypes();
  auto ctx = seeds->ctx;

  const IdType *seed_data = static_cast<const IdType*>(seeds->data);
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
    h_graphs[etype].in_ptr  = static_cast<const IdType*>(csr.indptr->data);
    h_graphs[etype].in_cols = static_cast<const IdType*>(csr.indices->data);
    h_graphs[etype].data = (CSRHasData(csr) ? static_cast<const IdType*>(csr.data->data) : nullptr);
  }
  // use default stream
  cudaStream_t stream = 0;
  auto device = DeviceAPI::Get(ctx);
  auto d_graphs = static_cast<GraphKernelData<IdType>*>(
      device->AllocWorkspace(ctx, (num_etypes) * sizeof(GraphKernelData<IdType>)));
  // copy graph metadata pointers to GPU
  device->CopyDataFromTo(h_graphs.data(), 0, d_graphs, 0,
      (num_etypes) * sizeof(GraphKernelData<IdType>),
      DGLContext{kDLCPU, 0},
      ctx,
      hg->GetCSRMatrix(0).indptr->dtype,
      stream);
  // copy metapath to GPU
  auto d_metapath = metapath.CopyTo(ctx);
  const IdType *d_metapath_data = static_cast<IdType *>(d_metapath->data);

  constexpr int BLOCK_SIZE = 256;
  constexpr int TILE_SIZE = BLOCK_SIZE * 4;
  dim3 block(256);
  dim3 grid((num_seeds + TILE_SIZE - 1) / TILE_SIZE);
  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  ATEN_FLOAT_TYPE_SWITCH(restart_prob->dtype, FloatType, "random walk GPU kernel", {
    CHECK(restart_prob->ctx.device_type == kDLGPU) << "restart prob should be in GPU.";
    CHECK(restart_prob->ndim == 1) << "restart prob dimension should be 1.";
    const FloatType *restart_prob_data = restart_prob.Ptr<FloatType>();
    const int64_t restart_prob_size = restart_prob->shape[0];
    CUDA_KERNEL_CALL(
      (_RandomWalkKernel<IdType, FloatType, BLOCK_SIZE, TILE_SIZE>),
      grid, block, 0, stream,
      random_seed,
      seed_data,
      num_seeds,
      d_metapath_data,
      max_num_steps,
      d_graphs,
      restart_prob_data,
      restart_prob_size,
      max_nodes,
      traces_data,
      eids_data);
  });

  device->FreeWorkspace(ctx, d_graphs);
  return std::make_pair(traces, eids);
}

template<DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob) {

  // not support no-uniform choice now
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      LOG(FATAL) << "Non-uniform choice is not supported in GPU.";
    }
  }

  auto restart_prob = NDArray::Empty(
      {0}, DLDataType{kDLFloat, 32, 1}, DGLContext{XPU, 0});
  return RandomWalkUniform<XPU, IdType>(hg, seeds, metapath, restart_prob);
}

template<DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkWithRestart(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {

  // not support no-uniform choice now
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      LOG(FATAL) << "Non-uniform choice is not supported in GPU.";
    }
  }
  auto device_ctx = seeds->ctx;
  auto restart_prob_array = NDArray::Empty(
      {1}, DLDataType{kDLFloat, 64, 1}, device_ctx);
  auto device = dgl::runtime::DeviceAPI::Get(device_ctx);

  // use default stream
  cudaStream_t stream = 0;
  device->CopyDataFromTo(
      &restart_prob, 0, restart_prob_array.Ptr<double>(), 0,
      sizeof(double),
      DGLContext{kDLCPU, 0}, device_ctx,
      restart_prob_array->dtype, stream);
  device->StreamSync(device_ctx, stream);

  return RandomWalkUniform<XPU, IdType>(hg, seeds, metapath, restart_prob_array);
}

template<DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob) {

  // not support no-uniform choice now
  for (const auto &etype_prob : prob) {
    if (!IsNullArray(etype_prob)) {
      LOG(FATAL) << "Non-uniform choice is not supported in GPU.";
    }
  }

  return RandomWalkUniform<XPU, IdType>(hg, seeds, metapath, restart_prob);
}

template<DLDeviceType XPU, typename IdxType>
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors(
    const IdArray src,
    const IdArray dst,
    const int64_t num_samples_per_node,
    const int64_t k) {
  CHECK(src->ctx.device_type == kDLGPU) <<
    "IdArray needs be on GPU!";
  const IdxType* src_data = src.Ptr<IdxType>();
  const IdxType* dst_data = dst.Ptr<IdxType>();
  const int64_t num_dst_nodes = (dst->shape[0] / num_samples_per_node);
  auto ctx = src->ctx;
  // use default stream
  cudaStream_t stream = 0;
  auto frequency_hashmap = FrequencyHashmap<IdxType>(num_dst_nodes,
      num_samples_per_node, ctx, stream);
  auto ret = frequency_hashmap.Topk(src_data, dst_data, src->dtype,
      src->shape[0], num_samples_per_node, k);
  return ret;
}

template
std::pair<IdArray, IdArray> RandomWalk<kDLGPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);
template
std::pair<IdArray, IdArray> RandomWalk<kDLGPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob);

template
std::pair<IdArray, IdArray> RandomWalkWithRestart<kDLGPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);
template
std::pair<IdArray, IdArray> RandomWalkWithRestart<kDLGPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);

template
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart<kDLGPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob);
template
std::pair<IdArray, IdArray> RandomWalkWithStepwiseRestart<kDLGPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    FloatArray restart_prob);

template
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors<kDLGPU, int32_t>(
    const IdArray src,
    const IdArray dst,
    const int64_t num_samples_per_node,
    const int64_t k);
template
std::tuple<IdArray, IdArray, IdArray> SelectPinSageNeighbors<kDLGPU, int64_t>(
    const IdArray src,
    const IdArray dst,
    const int64_t num_samples_per_node,
    const int64_t k);


};  // namespace impl

};  // namespace sampling

};  // namespace dgl
