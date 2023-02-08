/**
 *  Copyright (c) 2023 by Contributors
 * @file array/cpu/id_hash_map.cc
 * @brief Class about id hash map
 */

#include "id_hash_map.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif  // _MSC_VER

#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

#include <cmath>
#include <numeric>

using namespace dgl::runtime;

namespace {
static constexpr int64_t kEmptyKey = -1;
static constexpr int kGrainSize = 256;

// The formula is established from experience which is used
// to get the hashmap size from the input array size.
inline size_t GetMapSize(size_t num) {
  size_t capacity = 1;
  return capacity << static_cast<size_t>(1 + std::log2(num * 3));
}
}  // namespace

namespace dgl {
namespace aten {

template <typename IdType>
IdType IdHashMap<IdType>::CompareAndSwap(
    IdType* ptr, IdType old_val, IdType new_val) {
#ifdef _MSC_VER
  if (sizeof(IdType) == 4) {
    return _InterlockedCompareExchange(
        reinterpret_cast<LONG*>(ptr), new_val, old_val);
  } else if (sizeof(IdType) == 8) {
    return _InterlockedCompareExchange64(
        reinterpret_cast<LONGLONG*>(ptr), new_val, old_val);
  } else {
    LOG(FATAL) << "ID can only be int32 or int64";
  }
#elif __GNUC__  // _MSC_VER
  return __sync_val_compare_and_swap(ptr, old_val, new_val);
#else           // _MSC_VER
#error "CompareAndSwap is not supported on this platform."
#endif  // _MSC_VER
}

template <typename IdType>
IdHashMap<IdType>::IdHashMap() : mask_(0) {
  // Used to deallocate the memory in hash_map_ with device api
  // when the pointer is freed.
  auto deleter = [](Mapping* mappings) {
    if (mappings != nullptr) {
      std::cout << "Delete rcalled!\n";
      DGLContext ctx = DGLContext{kDGLCPU, 0};
      auto device = DeviceAPI::Get(ctx);
      device->FreeWorkspace(ctx, mappings);
    }
  };
  hash_map_ = {nullptr, deleter};
}

template <typename IdType>
IdArray IdHashMap<IdType>::Init(const IdArray& ids) {
  CHECK_EQ(ids.defined(), true);
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t num_ids = static_cast<size_t>(ids->shape[0]);
  // Make sure `ids` is not 0 dim.
  CHECK_GT(num_ids, 0);
  size_t capacity = GetMapSize(num_ids);
  mask_ = static_cast<IdType>(capacity - 1);

  auto ctx = DGLContext{kDGLCPU, 0};
  hash_map_.reset(static_cast<Mapping*>(
      DeviceAPI::Get(ctx)->AllocWorkspace(ctx, sizeof(Mapping) * capacity)));
  memset(hash_map_.get(), -1, sizeof(Mapping) * capacity);

  // This code block is to fill the ids into hash_map_.
  IdArray unique_ids = NewIdArray(num_ids, ctx, sizeof(IdType) * 8);
  // An auxiliary array indicates whether the corresponding elements
  // are inserted into hash map or not. Use `int16_t` instead of `bool` as
  // vector<bool> is unsafe when updating different elements from different
  // threads. See https://en.cppreference.com/w/cpp/container#Thread_safety.
  std::vector<int16_t> valid(num_ids);
  auto thread_num = compute_num_threads(0, num_ids, kGrainSize);
  std::vector<size_t> block_offset(thread_num + 1, 0);
  IdType* unique_ids_data = unique_ids.Ptr<IdType>();

  // Insert all elements in this loop.
  parallel_for(0, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    size_t count = 0;
    for (int64_t i = s; i < e; i++) {
      Insert(ids_data[i], &valid, i);
      count += valid[i];
    }
    block_offset[omp_get_thread_num() + 1] = count;
  });

  // Get ExclusiveSum of each block.
  std::partial_sum(
      block_offset.begin() + 1, block_offset.end(), block_offset.begin() + 1);
  unique_ids->shape[0] = block_offset.back();

  // Get unique array from ids and set value for hash map.
  parallel_for(0, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    auto tid = omp_get_thread_num();
    auto pos = block_offset[tid];
    for (int64_t i = s; i < e; i++) {
      if (valid[i]) {
        unique_ids_data[pos] = ids_data[i];
        Set(ids_data[i], pos);
        pos = pos + 1;
      }
    }
  });
  return unique_ids;
}

template <typename IdType>
IdArray IdHashMap<IdType>::MapIds(const IdArray& ids) const {
  CHECK_EQ(ids.defined(), true);
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t num_ids = static_cast<size_t>(ids->shape[0]);
  CHECK_GT(num_ids, 0);

  DGLContext ctx = DGLContext{kDGLCPU, 0};
  IdArray new_ids = NewIdArray(num_ids, ctx, sizeof(IdType) * 8);
  IdType* values_data = new_ids.Ptr<IdType>();

  parallel_for(0, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      values_data[i] = MapId(ids_data[i]);
    }
  });
  return new_ids;
}

template <typename IdType>
inline void IdHashMap<IdType>::Next(IdType* pos, IdType* delta) const {
  // Use Quadric probing.
  *pos = (*pos + (*delta) * (*delta)) & mask_;
  *delta = *delta + 1;
}

template <typename IdType>
IdType IdHashMap<IdType>::MapId(IdType id) const {
  IdType pos = (id & mask_), delta = 1;
  IdType empty_key = static_cast<IdType>(kEmptyKey);
  while (hash_map_[pos].key != empty_key && hash_map_[pos].key != id) {
    Next(&pos, &delta);
  }
  return hash_map_[pos].value;
}

template <typename IdType>
void IdHashMap<IdType>::Insert(
    IdType id, std::vector<int16_t>* valid, size_t index) {
  IdType pos = (id & mask_), delta = 1;
  while (!AttemptInsertAt(pos, id, valid, index)) {
    Next(&pos, &delta);
  }
}

template <typename IdType>
void IdHashMap<IdType>::Set(IdType key, IdType value) {
  IdType pos = (key & mask_), delta = 1;
  while (hash_map_[pos].key != key) {
    Next(&pos, &delta);
  }

  hash_map_[pos].value = value;
}

template <typename IdType>
bool IdHashMap<IdType>::AttemptInsertAt(
    int64_t pos, IdType key, std::vector<int16_t>* valid, size_t index) {
  IdType empty_key = static_cast<IdType>(kEmptyKey);
  IdType old_val = CompareAndSwap(&(hash_map_[pos].key), empty_key, key);

  if (old_val != empty_key && old_val != key) {
    return false;
  } else {
    if (old_val == empty_key) (*valid)[index] = true;
    return true;
  }
}

template class IdHashMap<int32_t>;
template class IdHashMap<int64_t>;

}  // namespace aten
}  // namespace dgl
