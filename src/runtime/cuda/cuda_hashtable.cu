/*!
 *  Copyright (c) 2021 by Contributors
 * \file runtime/cuda/cuda_device_common.cuh
 * \brief Device level functions for within cuda kernels.
 */

#include <cub/cub.cuh>
#include <cassert>

#include "cuda_hashtable.cuh"


namespace dgl {
namespace runtime {
namespace cuda {

namespace {

constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;

}  // namespace


template<typename IdType>
inline __device__ bool OrderedHashTable<IdType>::AttemptInsertAt(
    const size_t pos,
    const IdType id,
    const size_t index) {
  const IdType key = AtomicCAS(&table_[pos].key, EmptyKey, id);
  if (key == EmptyKey || key == id) {
    // we either set a match key, or found a matching key, so then place the
    // minimum index in position. Match the type of atomicMin, so ignore
    // linting
    atomicMin(reinterpret_cast<unsigned long long*>(&table_[pos].index), // NOLINT
        static_cast<unsigned long long>(index)); // NOLINT
    return true;
  } else {
    // we need to search elsewhere
    return false;
  }
}

template<typename IdType>
inline __device__ OrderedHashTable<IdType>::Iterator OrderedHashTable<IdType>::Insert(
    const IdType id,
    const size_t index) {
  size_t pos = Hash(id);

  // linearly scan for an empty slot or matching entry
  IdType delta = 1;
  while (!AttemptInsertAt(pos, id, index)) {
    pos = Hash(pos+delta);
    delta +=1;
  }

  return table_+pos;
}


/**
* \brief This generates a hash map where the keys are the global node numbers,
* and the values are indexes, and inputs may have duplciates.
*
* \tparam IdType The type of of id.
* \tparam BLOCK_SIZE The size of the thread block.
* \tparam TILE_SIZE The number of entries each thread block will process.
* \param nodes The nodes to insert.
* \param num_nodes The number of nodes to insert.
* \param table The hash table.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_duplicates(
    const IdType * const nodes,
    const int64_t num_nodes,
    OrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      table.Insert(nodes[index], index);
    }
  }
}

/**
* \brief This generates a hash map where the keys are the global node numbers,
* and the values are indexes, and all inputs are unique.
*
* \tparam IdType The type of of id.
* \tparam BLOCK_SIZE The size of the thread block.
* \tparam TILE_SIZE The number of entries each thread block will process.
* \param nodes The unique nodes to insert.
* \param num_nodes The number of nodes to insert.
* \param table The hash table.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(
    const IdType * const nodes,
    const int64_t num_nodes,
    OrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Iterator = typename OrderedHashTable<IdType>::Iterator;

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const Iterator pos = table.Insert(nodes[index], index);

      // since we are only inserting unique nodes, we know their local id
      // will be equal to their index
      pos->local = static_cast<IdType>(index);
    }
  }
}

/**
* \brief This counts the number of nodes inserted per thread block.
*
* \tparam IdType The type of of id.
* \tparam BLOCK_SIZE The size of the thread block.
* \tparam TILE_SIZE The number of entries each thread block will process.
* \param nodes The nodes ot insert.
* \param num_nodes The number of nodes to insert.
* \param hash_size The size of the hash table.
* \param pairs The hash table.
* \param num_unique_nodes The number of nodes inserted into the hash table per thread
* block.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(
    const IdType * nodes,
    const size_t num_nodes,
    const OrderedHashTable<IdType> table,
    IdType * const num_unique_nodes) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  IdType count = 0;

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const Mapping& mapping = *table.Search(nodes[index]);
      if (mapping.index == index) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique_nodes[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique_nodes[gridDim.x] = 0;
    }
  }
}


/**
* \brief Update the local numbering of elements in the hashmap.
*
* \tparam IdType The type of id.
* \tparam BLOCK_SIZE The size of the thread blocks.
* \tparam TILE_SIZE The number of elements each thread block works on.
* \param nodes The set of non-unique nodes to update from.
* \param num_nodes The number of non-unique nodes.
* \param hash_size The size of the hash table.
* \param mappings The hash table.
* \param num_nodes_prefix The number of unique nodes preceding each thread
* block.
* \param global_nodes The set of unique nodes (output).
* \param global_node_count The number of unique nodes (output).
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(
    const IdType * const nodes,
    const size_t num_nodes,
    OrderedHashTable<IdType> table,
    const IdType * const num_nodes_prefix,
    IdType * const global_nodes,
    int64_t * const global_node_count) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename OrderedHashTable<IdType>::Mapping;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_nodes_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i*BLOCK_SIZE + blockIdx.x*TILE_SIZE;

    FlagType flag;
    Mapping * kv;
    if (index < num_nodes) {
      kv = table.Search(nodes[index]);
      flag = kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = offset+flag;
      kv->local = pos;
      global_nodes[pos] = nodes[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *global_node_count = num_nodes_prefix[gridDim.x];
  }
}

template<typename IdType>
void OrderedHashTable<IdType>::FillWithDuplicates(
    const IdType * const input,
    const size_t num_input,
    IdType * const unique,
    int64_t * const num_unique,
    DGLContext ctx,
    cudaStream_t stream) {

  auto device = runtime::DeviceAPI::Get(ctx);

  const int64_t num_tiles = (num_input+TILE_SIZE-1)/TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(BLOCK_SIZE);

  generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
      input,
      num_input,
      *this);
  CUDA_CALL(cudaGetLastError());

  IdType * item_prefix = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*(num_input+1)));

  count_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
      input,
      num_input,
      *this,
      item_prefix);
  CUDA_CALL(cudaGetLastError());

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr,
      workspace_bytes,
      static_cast<IdType*>(nullptr),
      static_cast<IdType*>(nullptr),
      grid.x+1));
  void * workspace = device->AllocWorkspace(ctx, workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace,
      workspace_bytes,
      item_prefix,
      item_prefix,
      grid.x+1, stream));
  device->FreeWorkspace(ctx, workspace);

  compact_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
      input,
      num_input,
      *this,
      item_prefix,
      unique,
      num_unique);
  CUDA_CALL(cudaGetLastError());
  device->FreeWorkspace(ctx, item_prefix);
}

template<typename IdType>
void OrderedHashTable<IdType>::FillWithUnique(
    const IdType * const input,
    const size_t num_input,
    DGLContext /* ctx */,
    cudaStream_t stream) {

  const int64_t num_tiles = (num_input+TILE_SIZE-1)/TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(BLOCK_SIZE);

  generate_hashmap_unique<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
      input,
      num_input,
      *this);
  CUDA_CALL(cudaGetLastError());
}

template class OrderedHashTable<int32_t>;
template class OrderedHashTable<int64_t>;

}  // namespace cuda
}  // namespace runtime
}  // namespace dgl
