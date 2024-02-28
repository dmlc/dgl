/**
 *  Copyright (c) 2021 by Contributors
 * @file runtime/cuda/cuda_device_common.cuh
 * @brief Device level functions for within cuda kernels.
 */

#include <cassert>
#include <cub/cub.cuh>  // NOLINT

#include "../../array/cuda/atomic.cuh"
#include "cuda_common.h"
#include "cuda_hashtable.cuh"

using namespace dgl::aten::cuda;

namespace dgl {
namespace runtime {
namespace cuda {

namespace {

constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;

/**
 * @brief This is the mutable version of the DeviceOrderedHashTable, for use in
 * inserting elements into the hashtable.
 *
 * @tparam IdType The type of ID to store in the hashtable.
 */
template <typename IdType>
class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable<IdType> {
 public:
  typedef typename DeviceOrderedHashTable<IdType>::Mapping* Iterator;
  static constexpr IdType kEmptyKey = DeviceOrderedHashTable<IdType>::kEmptyKey;

  /**
   * @brief Create a new mutable hashtable for use on the device.
   *
   * @param hostTable The original hash table on the host.
   */
  explicit MutableDeviceOrderedHashTable(
      OrderedHashTable<IdType>* const hostTable)
      : DeviceOrderedHashTable<IdType>(hostTable->DeviceHandle()) {}

  /**
   * @brief Find the mutable mapping of a given key within the hash table.
   *
   * WARNING: The key must exist within the hashtable. Searching for a key not
   * in the hashtable is undefined behavior.
   *
   * @param id The key to search for.
   *
   * @return The mapping.
   */
  inline __device__ Iterator Search(const IdType id) {
    const IdType pos = SearchForPosition(id);

    return GetMutable(pos);
  }

  /**
   * @brief Attempt to insert into the hash table at a specific location.
   *
   * @param pos The position to insert at.
   * @param id The ID to insert into the hash table.
   * @param index The original index of the item being inserted.
   *
   * @return True, if the insertion was successful.
   */
  inline __device__ bool AttemptInsertAt(
      const size_t pos, const IdType id, const size_t index) {
    const IdType key = AtomicCAS(&GetMutable(pos)->key, kEmptyKey, id);
    if (key == kEmptyKey || key == id) {
      // we either set a match key, or found a matching key, so then place the
      // minimum index in position. Match the type of atomicMin, so ignore
      // linting
      atomicMin(
          reinterpret_cast<unsigned long long*>(  // NOLINT
              &GetMutable(pos)->index),
          static_cast<unsigned long long>(index));  // NOLINT
      return true;
    } else {
      // we need to search elsewhere
      return false;
    }
  }

  /**
   * @brief Insert key-index pair into the hashtable.
   *
   * @param id The ID to insert.
   * @param index The index at which the ID occured.
   *
   * @return An iterator to inserted mapping.
   */
  inline __device__ Iterator Insert(const IdType id, const size_t index) {
    size_t pos = Hash(id);

    // linearly scan for an empty slot or matching entry
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }

    return GetMutable(pos);
  }

 private:
  /**
   * @brief Get a mutable iterator to the given bucket in the hashtable.
   *
   * @param pos The given bucket.
   *
   * @return The iterator.
   */
  inline __device__ Iterator GetMutable(const size_t pos) {
    assert(pos < this->size_);
    // The parent class Device is read-only, but we ensure this can only be
    // constructed from a mutable version of OrderedHashTable, making this
    // a safe cast to perform.
    return const_cast<Iterator>(this->table_ + pos);
  }
};

/**
 * @brief Calculate the number of buckets in the hashtable. To guarantee we can
 * fill the hashtable in the worst case, we must use a number of buckets which
 * is a power of two.
 * https://en.wikipedia.org/wiki/Quadratic_probing#Limitations
 *
 * @param num The number of items to insert (should be an upper bound on the
 * number of unique keys).
 * @param scale The power of two larger the number of buckets should be than the
 * unique keys.
 *
 * @return The number of buckets the table should contain.
 */
size_t TableSize(const size_t num, const int scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

/**
 * @brief This structure is used with cub's block-level prefixscan in order to
 * keep a running sum as items are iteratively processed.
 *
 * @tparam IdType The type to perform the prefixsum on.
 */
template <typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(const IdType running_total)
      : running_total_(running_total) {}

  __device__ IdType operator()(const IdType block_aggregate) {
    const IdType old_prefix = running_total_;
    running_total_ += block_aggregate;
    return old_prefix;
  }
};

}  // namespace

/**
 * @brief This generates a hash map where the keys are the global item numbers,
 * and the values are indexes, and inputs may have duplciates.
 *
 * @tparam IdType The type of of id.
 * @tparam BLOCK_SIZE The size of the thread block.
 * @tparam TILE_SIZE The number of entries each thread block will process.
 * @param items The items to insert.
 * @param num_items The number of items to insert.
 * @param table The hash table.
 */
template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_duplicates(
    const IdType* const items, const int64_t num_items,
    MutableDeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      table.Insert(items[index], index);
    }
  }
}

/**
 * @brief This generates a hash map where the keys are the global item numbers,
 * and the values are indexes, and all inputs are unique.
 *
 * @tparam IdType The type of of id.
 * @tparam BLOCK_SIZE The size of the thread block.
 * @tparam TILE_SIZE The number of entries each thread block will process.
 * @param items The unique items to insert.
 * @param num_items The number of items to insert.
 * @param table The hash table.
 */
template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(
    const IdType* const items, const int64_t num_items,
    MutableDeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Iterator = typename MutableDeviceOrderedHashTable<IdType>::Iterator;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Iterator pos = table.Insert(items[index], index);

      // since we are only inserting unique items, we know their local id
      // will be equal to their index
      pos->local = static_cast<IdType>(index);
    }
  }
}

/**
 * @brief This counts the number of nodes inserted per thread block.
 *
 * @tparam IdType The type of of id.
 * @tparam BLOCK_SIZE The size of the thread block.
 * @tparam TILE_SIZE The number of entries each thread block will process.
 * @param input The nodes to insert.
 * @param num_input The number of nodes to insert.
 * @param table The hash table.
 * @param num_unique The number of nodes inserted into the hash table per thread
 * block.
 */
template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(
    const IdType* items, const size_t num_items,
    DeviceOrderedHashTable<IdType> table, IdType* const num_unique) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Mapping& mapping = *table.Search(items[index]);
      if (mapping.index == index) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

/**
 * @brief Update the local numbering of elements in the hashmap.
 *
 * @tparam IdType The type of id.
 * @tparam BLOCK_SIZE The size of the thread blocks.
 * @tparam TILE_SIZE The number of elements each thread block works on.
 * @param items The set of non-unique items to update from.
 * @param num_items The number of non-unique items.
 * @param table The hash table.
 * @param num_items_prefix The number of unique items preceding each thread
 * block.
 * @param unique_items The set of unique items (output).
 * @param num_unique_items The number of unique items (output).
 */
template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(
    const IdType* const items, const size_t num_items,
    MutableDeviceOrderedHashTable<IdType> table,
    const IdType* const num_items_prefix, IdType* const unique_items,
    int64_t* const num_unique_items) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_items_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;

    FlagType flag;
    Mapping* kv;
    if (index < num_items) {
      kv = table.Search(items[index]);
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
      const IdType pos = offset + flag;
      kv->local = pos;
      unique_items[pos] = items[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *num_unique_items = num_items_prefix[gridDim.x];
  }
}

// DeviceOrderedHashTable implementation

template <typename IdType>
DeviceOrderedHashTable<IdType>::DeviceOrderedHashTable(
    const Mapping* const table, const size_t size)
    : table_(table), size_(size) {}

template <typename IdType>
DeviceOrderedHashTable<IdType> OrderedHashTable<IdType>::DeviceHandle() const {
  return DeviceOrderedHashTable<IdType>(table_, size_);
}

// OrderedHashTable implementation

template <typename IdType>
OrderedHashTable<IdType>::OrderedHashTable(
    const size_t size, DGLContext ctx, cudaStream_t stream, const int scale)
    : table_(nullptr), size_(TableSize(size, scale)), ctx_(ctx) {
  // make sure we will at least as many buckets as items.
  CHECK_GT(scale, 0);

  auto device = runtime::DeviceAPI::Get(ctx_);
  table_ = static_cast<Mapping*>(
      device->AllocWorkspace(ctx_, sizeof(Mapping) * size_));

  CUDA_CALL(cudaMemsetAsync(
      table_, DeviceOrderedHashTable<IdType>::kEmptyKey,
      sizeof(Mapping) * size_, stream));
}

template <typename IdType>
OrderedHashTable<IdType>::~OrderedHashTable() {
  auto device = runtime::DeviceAPI::Get(ctx_);
  device->FreeWorkspace(ctx_, table_);
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithDuplicates(
    const IdType* const input, const size_t num_input, IdType* const unique,
    int64_t* const num_unique, cudaStream_t stream) {
  auto device = runtime::DeviceAPI::Get(ctx_);

  const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(BLOCK_SIZE);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);

  CUDA_KERNEL_CALL(
      (generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE>), grid, block,
      0, stream, input, num_input, device_table);

  IdType* item_prefix = static_cast<IdType*>(
      device->AllocWorkspace(ctx_, sizeof(IdType) * (num_input + 1)));

  CUDA_KERNEL_CALL(
      (count_hashmap<IdType, BLOCK_SIZE, TILE_SIZE>), grid, block, 0, stream,
      input, num_input, device_table, item_prefix);

  size_t workspace_bytes;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, workspace_bytes, static_cast<IdType*>(nullptr),
      static_cast<IdType*>(nullptr), grid.x + 1, stream));
  void* workspace = device->AllocWorkspace(ctx_, workspace_bytes);

  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1,
      stream));
  device->FreeWorkspace(ctx_, workspace);

  CUDA_KERNEL_CALL(
      (compact_hashmap<IdType, BLOCK_SIZE, TILE_SIZE>), grid, block, 0, stream,
      input, num_input, device_table, item_prefix, unique, num_unique);
  device->FreeWorkspace(ctx_, item_prefix);
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithUnique(
    const IdType* const input, const size_t num_input, cudaStream_t stream) {
  const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(BLOCK_SIZE);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);

  CUDA_KERNEL_CALL(
      (generate_hashmap_unique<IdType, BLOCK_SIZE, TILE_SIZE>), grid, block, 0,
      stream, input, num_input, device_table);
}

template class OrderedHashTable<int32_t>;
template class OrderedHashTable<int64_t>;

}  // namespace cuda
}  // namespace runtime
}  // namespace dgl
