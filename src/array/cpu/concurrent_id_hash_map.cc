/**
 *  Copyright (c) 2023 by Contributors
 * @file array/cpu/concurrent_id_hash_map.cc
 * @brief Class about id hash map
 */

#include "concurrent_id_hash_map.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif  // _MSC_VER

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
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
IdType ConcurrentIdHashMap<IdType>::CompareAndSwap(
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
ConcurrentIdHashMap<IdType>::ConcurrentIdHashMap() : mask_(0) {
  // Used to deallocate the memory in hash_map_ with device api
  // when the pointer is freed.
  auto deleter = [](Mapping* mappings) {
    if (mappings != nullptr) {
      DGLContext ctx = DGLContext{kDGLCPU, 0};
      auto device = DeviceAPI::Get(ctx);
      device->FreeWorkspace(ctx, mappings);
    }
  };
  hash_map_ = {nullptr, deleter};
}

template <typename IdType>
IdArray ConcurrentIdHashMap<IdType>::Init(
    const IdArray& ids, size_t num_seeds) {
  CHECK_EQ(ids.defined(), true);
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t num_ids = static_cast<size_t>(ids->shape[0]);
  // Make sure `ids` is not 0 dim.
  CHECK_GE(num_seeds, 0);
  CHECK_GE(num_ids, num_seeds);
  size_t capacity = GetMapSize(num_ids);
  mask_ = static_cast<IdType>(capacity - 1);

  auto ctx = DGLContext{kDGLCPU, 0};
  auto device = DeviceAPI::Get(ctx);
  hash_map_.reset(static_cast<Mapping*>(
      device->AllocWorkspace(ctx, sizeof(Mapping) * capacity)));
  memset(hash_map_.get(), -1, sizeof(Mapping) * capacity);

  // This code block is to fill the ids into hash_map_.
  IdArray unique_ids = NewIdArray(num_ids, ctx, sizeof(IdType) * 8);
  IdType* unique_ids_data = unique_ids.Ptr<IdType>();
  // Fill in the first `num_seeds` ids.
  parallel_for(0, num_seeds, kGrainSize, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      InsertAndSet(ids_data[i], static_cast<IdType>(i));
    }
  });
  // Place the first `num_seeds` ids.
  device->CopyDataFromTo(
      ids_data, 0, unique_ids_data, 0, sizeof(IdType) * num_seeds, ctx, ctx,
      ids->dtype);

  // An auxiliary array indicates whether the corresponding elements
  // are inserted into hash map or not. Use `int16_t` instead of `bool` as
  // vector<bool> is unsafe when updating different elements from different
  // threads. See https://en.cppreference.com/w/cpp/container#Thread_safety.
  std::vector<int16_t> valid(num_ids);
  auto thread_num = compute_num_threads(0, num_ids, kGrainSize);
  std::vector<size_t> block_offset(thread_num + 1, 0);
  // Insert all elements in this loop.
  parallel_for(num_seeds, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    size_t count = 0;
    for (int64_t i = s; i < e; i++) {
      valid[i] = Insert(ids_data[i]);
      count += valid[i];
    }
    block_offset[omp_get_thread_num() + 1] = count;
  });

  // Get ExclusiveSum of each block.
  std::partial_sum(
      block_offset.begin() + 1, block_offset.end(), block_offset.begin() + 1);
  unique_ids->shape[0] = num_seeds + block_offset.back();

  // Get unique array from ids and set value for hash map.
  parallel_for(num_seeds, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    auto tid = omp_get_thread_num();
    auto pos = block_offset[tid] + num_seeds;
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
IdArray ConcurrentIdHashMap<IdType>::MapIds(const IdArray& ids) const {
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
inline void ConcurrentIdHashMap<IdType>::Next(
    IdType* pos, IdType* delta) const {
  // Use Quadric probing.
  *pos = (*pos + (*delta) * (*delta)) & mask_;
  *delta = *delta + 1;
}

template <typename IdType>
inline IdType ConcurrentIdHashMap<IdType>::MapId(IdType id) const {
  IdType pos = (id & mask_), delta = 1;
  IdType empty_key = static_cast<IdType>(kEmptyKey);
  while (hash_map_[pos].key != empty_key && hash_map_[pos].key != id) {
    Next(&pos, &delta);
  }
  return hash_map_[pos].value;
}

template <typename IdType>
bool ConcurrentIdHashMap<IdType>::Insert(IdType id) {
  IdType pos = (id & mask_), delta = 1;
  InsertState state = AttemptInsertAt(pos, id);
  while (state == InsertState::OCCUPIED) {
    Next(&pos, &delta);
    state = AttemptInsertAt(pos, id);
  }

  return state == InsertState::INSERTED;
}

template <typename IdType>
inline void ConcurrentIdHashMap<IdType>::Set(IdType key, IdType value) {
  IdType pos = (key & mask_), delta = 1;
  while (hash_map_[pos].key != key) {
    Next(&pos, &delta);
  }

  hash_map_[pos].value = value;
}

template <typename IdType>
inline void ConcurrentIdHashMap<IdType>::InsertAndSet(IdType id, IdType value) {
  IdType pos = (id & mask_), delta = 1;
  while (AttemptInsertAt(pos, id) == InsertState::OCCUPIED) {
    Next(&pos, &delta);
  }

  hash_map_[pos].value = value;
}

template <typename IdType>
inline typename ConcurrentIdHashMap<IdType>::InsertState
ConcurrentIdHashMap<IdType>::AttemptInsertAt(int64_t pos, IdType key) {
  IdType empty_key = static_cast<IdType>(kEmptyKey);
  IdType old_val = CompareAndSwap(&(hash_map_[pos].key), empty_key, key);
  if (old_val == empty_key) {
    return InsertState::INSERTED;
  } else if (old_val == key) {
    return InsertState::EXISTED;
  } else {
    return InsertState::OCCUPIED;
  }
}

template class ConcurrentIdHashMap<int32_t>;
template class ConcurrentIdHashMap<int64_t>;

template <typename IdType>
bool BoolCompareAndSwap(IdType* ptr) {
#ifdef _MSC_VER
  if (sizeof(IdType) == 4) {
    return _InterlockedCompareExchange(reinterpret_cast<LONG*>(ptr), 0, -1) ==
           -1;
  } else if (sizeof(IdType) == 8) {
    return _InterlockedCompareExchange64(
               reinterpret_cast<LONGLONG*>(ptr), 0, -1) == -1;
  } else {
    LOG(FATAL) << "ID can only be int32 or int64";
  }
#elif __GNUC__  // _MSC_VER
  return __sync_bool_compare_and_swap(ptr, -1, 0);
#else           // _MSC_VER
#error "CompareAndSwap is not supported on this platform."
#endif  // _MSC_VER
}

template bool BoolCompareAndSwap<int32_t>(int32_t*);
template bool BoolCompareAndSwap<int64_t>(int64_t*);

}  // namespace aten
}  // namespace dgl
