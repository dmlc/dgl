/**
 *  Copyright (c) 2021 by Contributors
 * @file graph/sampling/frequency_hashmap.cu
 * @brief frequency hashmap - used to select top-k frequency edges of each node
 */

#include <algorithm>
#include <cub/cub.cuh>  // NOLINT
#include <tuple>
#include <utility>

#include "../../../array/cuda/atomic.cuh"
#include "../../../runtime/cuda/cuda_common.h"
#include "frequency_hashmap.cuh"

namespace dgl {

namespace sampling {

namespace impl {

namespace {

int64_t _table_size(const int64_t num, const int64_t scale) {
  /**
   * Calculate the number of buckets in the hashtable. To guarantee we can
   * fill the hashtable in the worst case, we must use a number of buckets which
   * is a power of two.
   * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
   */
  const int64_t next_pow2 = 1 << static_cast<int64_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

template <typename IdxType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _init_edge_table(void *edge_hashmap, int64_t edges_len) {
  using EdgeItem = typename DeviceEdgeHashmap<IdxType>::EdgeItem;
  auto edge_hashmap_t = static_cast<EdgeItem *>(edge_hashmap);
  int64_t start_idx = (blockIdx.x * TILE_SIZE) + threadIdx.x;
  int64_t last_idx = start_idx + TILE_SIZE;
#pragma unroll(4)
  for (int64_t idx = start_idx; idx < last_idx; idx += BLOCK_SIZE) {
    if (idx < edges_len) {
      EdgeItem *edge = (edge_hashmap_t + idx);
      edge->src = static_cast<IdxType>(-1);
      edge->cnt = static_cast<IdxType>(0);
    }
  }
}

template <typename IdxType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _count_frequency(
    const IdxType *src_data, const int64_t num_edges,
    const int64_t num_edges_per_node, IdxType *edge_blocks_prefix,
    bool *is_first_position, DeviceEdgeHashmap<IdxType> device_edge_hashmap) {
  int64_t start_idx = (blockIdx.x * TILE_SIZE) + threadIdx.x;
  int64_t last_idx = start_idx + TILE_SIZE;

  IdxType count = 0;
  for (int64_t idx = start_idx; idx < last_idx; idx += BLOCK_SIZE) {
    if (idx < num_edges) {
      IdxType src = src_data[idx];
      if (src == static_cast<IdxType>(-1)) {
        continue;
      }
      IdxType dst_idx = (idx / num_edges_per_node);
      if (device_edge_hashmap.InsertEdge(src, dst_idx) == 0) {
        is_first_position[idx] = true;
        ++count;
      }
    }
  }

  using BlockReduce = typename cub::BlockReduce<IdxType, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);
  if (threadIdx.x == 0) {
    edge_blocks_prefix[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      edge_blocks_prefix[gridDim.x] = 0;
    }
  }
}

/**
 * This structure is used with cub's block-level prefixscan in order to
 * keep a running sum as items are iteratively processed.
 */
template <typename T>
struct BlockPrefixCallbackOp {
  T _running_total;

  __device__ BlockPrefixCallbackOp(const T running_total)
      : _running_total(running_total) {}

  __device__ T operator()(const T block_aggregate) {
    const T old_prefix = _running_total;
    _running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename IdxType, typename Idx64Type, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _compact_frequency(
    const IdxType *src_data, const IdxType *dst_data, const int64_t num_edges,
    const int64_t num_edges_per_node, const IdxType *edge_blocks_prefix,
    const bool *is_first_position, IdxType *num_unique_each_node,
    IdxType *unique_src_edges, Idx64Type *unique_frequency,
    DeviceEdgeHashmap<IdxType> device_edge_hashmap) {
  int64_t start_idx = (blockIdx.x * TILE_SIZE) + threadIdx.x;
  int64_t last_idx = start_idx + TILE_SIZE;
  const IdxType block_offset = edge_blocks_prefix[blockIdx.x];

  using BlockScan = typename cub::BlockScan<IdxType, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_space;
  BlockPrefixCallbackOp<IdxType> prefix_op(0);

  for (int64_t idx = start_idx; idx < last_idx; idx += BLOCK_SIZE) {
    IdxType flag = 0;
    if (idx < num_edges) {
      IdxType src = src_data[idx];
      IdxType dst_idx = (idx / num_edges_per_node);
      if (idx % num_edges_per_node == 0) {
        num_unique_each_node[dst_idx] =
            device_edge_hashmap.GetDstCount(dst_idx);
      }
      if (is_first_position[idx] == true) {
        flag = 1;
      }
      BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
      __syncthreads();
      if (is_first_position[idx] == true) {
        const IdxType pos = (block_offset + flag);
        unique_src_edges[pos] = src;
        if (sizeof(IdxType) != sizeof(Idx64Type) &&
            sizeof(IdxType) == 4) {  // if IdxType is a 32-bit data
          unique_frequency[pos] =
              ((static_cast<Idx64Type>(num_edges / num_edges_per_node - dst_idx)
                << 32) |
               device_edge_hashmap.GetEdgeCount(src, dst_idx));
        } else {
          unique_frequency[pos] =
              device_edge_hashmap.GetEdgeCount(src, dst_idx);
        }
      }
    }
  }
}

template <typename IdxType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _get_pick_num(
    IdxType *num_unique_each_node, const int64_t num_pick,
    const int64_t num_dst_nodes) {
  int64_t start_idx = (blockIdx.x * TILE_SIZE) + threadIdx.x;
  int64_t last_idx = start_idx + TILE_SIZE;
#pragma unroll(4)
  for (int64_t idx = start_idx; idx < last_idx; idx += BLOCK_SIZE) {
    if (idx < num_dst_nodes) {
      IdxType &num_unique = num_unique_each_node[idx];
      num_unique = min(num_unique, static_cast<IdxType>(num_pick));
    }
  }
}

template <typename IdxType, typename Idx64Type, int BLOCK_SIZE, int TILE_SIZE>
__global__ void _pick_data(
    const Idx64Type *unique_frequency, const IdxType *unique_src_edges,
    const IdxType *unique_input_offsets, const IdxType *dst_data,
    const int64_t num_edges_per_node, const int64_t num_dst_nodes,
    const int64_t num_edges, const IdxType *unique_output_offsets,
    IdxType *output_src, IdxType *output_dst, IdxType *output_frequency) {
  int64_t start_idx = (blockIdx.x * TILE_SIZE) + threadIdx.x;
  int64_t last_idx = start_idx + TILE_SIZE;

  for (int64_t idx = start_idx; idx < last_idx; idx += BLOCK_SIZE) {
    if (idx < num_dst_nodes) {
      const int64_t dst_pos = (idx * num_edges_per_node);
      assert(dst_pos < num_edges);
      const IdxType dst = dst_data[dst_pos];
      const IdxType last_output_offset = unique_output_offsets[idx + 1];
      assert(
          (last_output_offset - unique_output_offsets[idx]) <=
          (unique_input_offsets[idx + 1] - unique_input_offsets[idx]));
      for (IdxType output_idx = unique_output_offsets[idx],
                   input_idx = unique_input_offsets[idx];
           output_idx < last_output_offset; ++output_idx, ++input_idx) {
        output_src[output_idx] = unique_src_edges[input_idx];
        output_dst[output_idx] = dst;
        output_frequency[output_idx] =
            static_cast<IdxType>(unique_frequency[input_idx]);
      }
    }
  }
}

}  // namespace

// return the old cnt of this edge
template <typename IdxType>
inline __device__ IdxType DeviceEdgeHashmap<IdxType>::InsertEdge(
    const IdxType &src, const IdxType &dst_idx) {
  IdxType start_off = dst_idx * _num_items_each_dst;
  IdxType pos = EdgeHash(src);
  IdxType delta = 1;
  IdxType old_cnt = static_cast<IdxType>(-1);
  while (true) {
    IdxType old_src = dgl::aten::cuda::AtomicCAS(
        &_edge_hashmap[start_off + pos].src, static_cast<IdxType>(-1), src);
    if (old_src == static_cast<IdxType>(-1) || old_src == src) {
      // first insert
      old_cnt = dgl::aten::cuda::AtomicAdd(
          &_edge_hashmap[start_off + pos].cnt, static_cast<IdxType>(1));
      if (old_src == static_cast<IdxType>(-1)) {
        assert(dst_idx < _num_dst);
        dgl::aten::cuda::AtomicAdd(
            &_dst_unique_edges[dst_idx], static_cast<IdxType>(1));
      }
      break;
    }
    pos = EdgeHash(pos + delta);
    delta += 1;
  }
  return old_cnt;
}

template <typename IdxType>
inline __device__ IdxType
DeviceEdgeHashmap<IdxType>::GetDstCount(const IdxType &dst_idx) {
  return _dst_unique_edges[dst_idx];
}

template <typename IdxType>
inline __device__ IdxType DeviceEdgeHashmap<IdxType>::GetEdgeCount(
    const IdxType &src, const IdxType &dst_idx) {
  IdxType start_off = dst_idx * _num_items_each_dst;
  IdxType pos = EdgeHash(src);
  IdxType delta = 1;
  while (_edge_hashmap[start_off + pos].src != src) {
    pos = EdgeHash(pos + delta);
    delta += 1;
  }
  return _edge_hashmap[start_off + pos].cnt;
}

template <typename IdxType>
FrequencyHashmap<IdxType>::FrequencyHashmap(
    int64_t num_dst, int64_t num_items_each_dst, DGLContext ctx,
    cudaStream_t stream, int64_t edge_table_scale) {
  _ctx = ctx;
  _stream = stream;
  num_items_each_dst = _table_size(num_items_each_dst, edge_table_scale);
  auto device = dgl::runtime::DeviceAPI::Get(_ctx);
  auto dst_unique_edges = static_cast<IdxType *>(
      device->AllocWorkspace(_ctx, (num_dst) * sizeof(IdxType)));
  auto edge_hashmap = static_cast<EdgeItem *>(device->AllocWorkspace(
      _ctx, (num_dst * num_items_each_dst) * sizeof(EdgeItem)));
  constexpr int BLOCK_SIZE = 256;
  constexpr int TILE_SIZE = BLOCK_SIZE * 8;
  dim3 block(BLOCK_SIZE);
  dim3 grid((num_dst * num_items_each_dst + TILE_SIZE - 1) / TILE_SIZE);
  CUDA_CALL(cudaMemset(dst_unique_edges, 0, (num_dst) * sizeof(IdxType)));
  CUDA_KERNEL_CALL(
      (_init_edge_table<IdxType, BLOCK_SIZE, TILE_SIZE>), grid, block, 0,
      _stream, edge_hashmap, (num_dst * num_items_each_dst));
  _device_edge_hashmap = new DeviceEdgeHashmap<IdxType>(
      num_dst, num_items_each_dst, dst_unique_edges, edge_hashmap);
  _dst_unique_edges = dst_unique_edges;
  _edge_hashmap = edge_hashmap;
}

template <typename IdxType>
FrequencyHashmap<IdxType>::~FrequencyHashmap() {
  auto device = dgl::runtime::DeviceAPI::Get(_ctx);
  delete _device_edge_hashmap;
  _device_edge_hashmap = nullptr;
  device->FreeWorkspace(_ctx, _dst_unique_edges);
  _dst_unique_edges = nullptr;
  device->FreeWorkspace(_ctx, _edge_hashmap);
  _edge_hashmap = nullptr;
}

template <typename IdxType>
std::tuple<IdArray, IdArray, IdArray> FrequencyHashmap<IdxType>::Topk(
    const IdxType *src_data, const IdxType *dst_data, DGLDataType dtype,
    const int64_t num_edges, const int64_t num_edges_per_node,
    const int64_t num_pick) {
  using Idx64Type = int64_t;
  const int64_t num_dst_nodes = (num_edges / num_edges_per_node);
  constexpr int BLOCK_SIZE = 256;
  // XXX: a experienced value, best performance in GV100
  constexpr int TILE_SIZE = BLOCK_SIZE * 32;
  const dim3 block(BLOCK_SIZE);
  const dim3 edges_grid((num_edges + TILE_SIZE - 1) / TILE_SIZE);
  auto device = dgl::runtime::DeviceAPI::Get(_ctx);
  const IdxType num_edge_blocks = static_cast<IdxType>(edges_grid.x);
  IdxType num_unique_edges = 0;

  // to mark if this position of edges is the first inserting position for
  // _edge_hashmap
  bool *is_first_position = static_cast<bool *>(
      device->AllocWorkspace(_ctx, sizeof(bool) * (num_edges)));
  CUDA_CALL(cudaMemset(is_first_position, 0, sizeof(bool) * (num_edges)));
  // double space to use ExclusiveSum
  auto edge_blocks_prefix_data = static_cast<IdxType *>(device->AllocWorkspace(
      _ctx, 2 * sizeof(IdxType) * (num_edge_blocks + 1)));
  IdxType *edge_blocks_prefix = edge_blocks_prefix_data;
  IdxType *edge_blocks_prefix_alternate =
      (edge_blocks_prefix_data + (num_edge_blocks + 1));
  // triple space to use ExclusiveSum and unique_output_offsets
  auto num_unique_each_node_data = static_cast<IdxType *>(
      device->AllocWorkspace(_ctx, 3 * sizeof(IdxType) * (num_dst_nodes + 1)));
  IdxType *num_unique_each_node = num_unique_each_node_data;
  IdxType *num_unique_each_node_alternate =
      (num_unique_each_node_data + (num_dst_nodes + 1));
  IdxType *unique_output_offsets =
      (num_unique_each_node_data + 2 * (num_dst_nodes + 1));

  // 1. Scan the all edges and count the unique edges and unique edges for each
  // dst node
  CUDA_KERNEL_CALL(
      (_count_frequency<IdxType, BLOCK_SIZE, TILE_SIZE>), edges_grid, block, 0,
      _stream, src_data, num_edges, num_edges_per_node, edge_blocks_prefix,
      is_first_position, *_device_edge_hashmap);

  // 2. Compact the unique edges frequency
  // 2.1 ExclusiveSum the edge_blocks_prefix
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, edge_blocks_prefix,
      edge_blocks_prefix_alternate, num_edge_blocks + 1, _stream));
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, edge_blocks_prefix,
      edge_blocks_prefix_alternate, num_edge_blocks + 1, _stream));
  device->FreeWorkspace(_ctx, d_temp_storage);
  std::swap(edge_blocks_prefix, edge_blocks_prefix_alternate);
  device->CopyDataFromTo(
      &edge_blocks_prefix[num_edge_blocks], 0, &num_unique_edges, 0,
      sizeof(num_unique_edges), _ctx, DGLContext{kDGLCPU, 0}, dtype);
  device->StreamSync(_ctx, _stream);
  // 2.2 Allocate the data of unique edges and frequency
  // double space to use SegmentedRadixSort
  auto unique_src_edges_data = static_cast<IdxType *>(
      device->AllocWorkspace(_ctx, 2 * sizeof(IdxType) * (num_unique_edges)));
  IdxType *unique_src_edges = unique_src_edges_data;
  IdxType *unique_src_edges_alternate =
      unique_src_edges_data + num_unique_edges;
  // double space to use SegmentedRadixSort
  auto unique_frequency_data = static_cast<Idx64Type *>(
      device->AllocWorkspace(_ctx, 2 * sizeof(Idx64Type) * (num_unique_edges)));
  Idx64Type *unique_frequency = unique_frequency_data;
  Idx64Type *unique_frequency_alternate =
      unique_frequency_data + num_unique_edges;
  // 2.3 Compact the unique edges and their frequency
  CUDA_KERNEL_CALL(
      (_compact_frequency<IdxType, Idx64Type, BLOCK_SIZE, TILE_SIZE>),
      edges_grid, block, 0, _stream, src_data, dst_data, num_edges,
      num_edges_per_node, edge_blocks_prefix, is_first_position,
      num_unique_each_node, unique_src_edges, unique_frequency,
      *_device_edge_hashmap);

  // 3. SegmentedRadixSort the unique edges and unique_frequency
  // 3.1 ExclusiveSum the num_unique_each_node
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, num_unique_each_node,
      num_unique_each_node_alternate, num_dst_nodes + 1, _stream));
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, num_unique_each_node,
      num_unique_each_node_alternate, num_dst_nodes + 1, _stream));
  device->FreeWorkspace(_ctx, d_temp_storage);
  // 3.2 SegmentedRadixSort the unique_src_edges and unique_frequency
  // Create a set of DoubleBuffers to wrap pairs of device pointers
  cub::DoubleBuffer<Idx64Type> d_unique_frequency(
      unique_frequency, unique_frequency_alternate);
  cub::DoubleBuffer<IdxType> d_unique_src_edges(
      unique_src_edges, unique_src_edges_alternate);
  // Determine temporary device storage requirements
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  // the DeviceRadixSort is faster than DeviceSegmentedRadixSort,
  // especially when num_dst_nodes is large (about ~10000)
  if (dtype.bits == 32) {
    CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_unique_frequency,
        d_unique_src_edges, num_unique_edges, 0, sizeof(Idx64Type) * 8,
        _stream));
  } else {
    CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_unique_frequency,
        d_unique_src_edges, num_unique_edges, num_dst_nodes,
        num_unique_each_node_alternate, num_unique_each_node_alternate + 1, 0,
        sizeof(Idx64Type) * 8, _stream));
  }
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
  if (dtype.bits == 32) {
    CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_unique_frequency,
        d_unique_src_edges, num_unique_edges, 0, sizeof(Idx64Type) * 8,
        _stream));
  } else {
    CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp_storage, temp_storage_bytes, d_unique_frequency,
        d_unique_src_edges, num_unique_edges, num_dst_nodes,
        num_unique_each_node_alternate, num_unique_each_node_alternate + 1, 0,
        sizeof(Idx64Type) * 8, _stream));
  }
  device->FreeWorkspace(_ctx, d_temp_storage);

  // 4. Get the final pick number for each dst node
  // 4.1 Reset the min(num_pick, num_unique_each_node) to num_unique_each_node
  constexpr int NODE_TILE_SIZE = BLOCK_SIZE * 2;
  const dim3 nodes_grid((num_dst_nodes + NODE_TILE_SIZE - 1) / NODE_TILE_SIZE);
  CUDA_KERNEL_CALL(
      (_get_pick_num<IdxType, BLOCK_SIZE, NODE_TILE_SIZE>), nodes_grid, block,
      0, _stream, num_unique_each_node, num_pick, num_dst_nodes);
  // 4.2 ExclusiveSum the new num_unique_each_node as unique_output_offsets
  // use unique_output_offsets;
  d_temp_storage = nullptr;
  temp_storage_bytes = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, num_unique_each_node,
      unique_output_offsets, num_dst_nodes + 1, _stream));
  d_temp_storage = device->AllocWorkspace(_ctx, temp_storage_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, num_unique_each_node,
      unique_output_offsets, num_dst_nodes + 1, _stream));
  device->FreeWorkspace(_ctx, d_temp_storage);

  // 5. Pick the data to result
  IdxType num_output = 0;
  device->CopyDataFromTo(
      &unique_output_offsets[num_dst_nodes], 0, &num_output, 0,
      sizeof(num_output), _ctx, DGLContext{kDGLCPU, 0}, dtype);
  device->StreamSync(_ctx, _stream);

  IdArray res_src =
      IdArray::Empty({static_cast<int64_t>(num_output)}, dtype, _ctx);
  IdArray res_dst =
      IdArray::Empty({static_cast<int64_t>(num_output)}, dtype, _ctx);
  IdArray res_cnt =
      IdArray::Empty({static_cast<int64_t>(num_output)}, dtype, _ctx);
  CUDA_KERNEL_CALL(
      (_pick_data<IdxType, Idx64Type, BLOCK_SIZE, NODE_TILE_SIZE>), nodes_grid,
      block, 0, _stream, d_unique_frequency.Current(),
      d_unique_src_edges.Current(), num_unique_each_node_alternate, dst_data,
      num_edges_per_node, num_dst_nodes, num_edges, unique_output_offsets,
      res_src.Ptr<IdxType>(), res_dst.Ptr<IdxType>(), res_cnt.Ptr<IdxType>());

  device->FreeWorkspace(_ctx, is_first_position);
  device->FreeWorkspace(_ctx, edge_blocks_prefix_data);
  device->FreeWorkspace(_ctx, num_unique_each_node_data);
  device->FreeWorkspace(_ctx, unique_src_edges_data);
  device->FreeWorkspace(_ctx, unique_frequency_data);

  return std::make_tuple(res_src, res_dst, res_cnt);
}

template class FrequencyHashmap<int64_t>;

template class FrequencyHashmap<int32_t>;

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
