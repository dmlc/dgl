/**
 *  Copyright (c) 2023 by Contributors
 * @file concurrent_id_hash_map.cc
 * @brief Class about id hash map.
 */

#include "concurrent_id_hash_map.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif  // _MSC_VER

#include <cmath>
#include <numeric>

namespace {
static constexpr int64_t kEmptyKey = -1;
static constexpr int kGrainSize = 256;

// The formula is established from experience which is used to get the hashmap
// size from the input array size.
inline size_t GetMapSize(size_t num) {
  size_t capacity = 1;
  return capacity << static_cast<size_t>(1 + std::log2(num * 3));
}
}  // namespace

namespace graphbolt {
namespace sampling {

template <typename IdType>
IdType ConcurrentIdHashMap<IdType>::CompareAndSwap(
    IdType* ptr, IdType old_val, IdType new_val) {
#ifdef _MSC_VER
  if (sizeof(IdType) == 4) {
    return _InterlockedCompareExchange(
        reinterpret_cast<long*>(ptr), new_val, old_val);
  } else if (sizeof(IdType) == 8) {
    return _InterlockedCompareExchange64(
        reinterpret_cast<long long*>(ptr), new_val, old_val);
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
ConcurrentIdHashMap<IdType>::ConcurrentIdHashMap() : mask_(0) {}

template <typename IdType>
torch::Tensor ConcurrentIdHashMap<IdType>::Init(
    const torch::Tensor& ids, size_t num_seeds) {
  const IdType* ids_data = ids.data_ptr<IdType>();
  const size_t num_ids = static_cast<size_t>(ids.size(0));
  size_t capacity = GetMapSize(num_ids);
  mask_ = static_cast<IdType>(capacity - 1);

  hash_map_ =
      torch::full({static_cast<int64_t>(capacity * 2)}, -1, ids.options());

  // This code block is to fill the ids into hash_map_.
  auto unique_ids = torch::empty_like(ids);
  IdType* unique_ids_data = unique_ids.data_ptr<IdType>();
  // Insert all ids into the hash map.
  torch::parallel_for(0, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      InsertAndSetMin(ids_data[i], static_cast<IdType>(i));
    }
  });
  // Place the first `num_seeds` ids.
  unique_ids.slice(0, 0, num_seeds) = ids.slice(0, 0, num_seeds);

  // An auxiliary array indicates whether the corresponding elements
  // are inserted into hash map or not. Use `int16_t` instead of `bool` as
  // vector<bool> is unsafe when updating different elements from different
  // threads. See https://en.cppreference.com/w/cpp/container#Thread_safety.
  std::vector<int16_t> valid(num_ids);

  const int64_t num_threads = torch::get_num_threads();
  std::vector<size_t> block_offset(num_threads + 1, 0);

  // Count the valid numbers in each thread.
  torch::parallel_for(
      num_seeds, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
        size_t count = 0;
        for (int64_t i = s; i < e; i++) {
          if (MapId(ids_data[i]) == i) {
            count++;
            valid[i] = 1;
          }
        }
        auto thread_id = torch::get_thread_num();
        block_offset[thread_id + 1] = count;
      });

  // Get ExclusiveSum of each block.
  std::partial_sum(
      block_offset.begin() + 1, block_offset.end(), block_offset.begin() + 1);
  unique_ids = unique_ids.slice(0, 0, num_seeds + block_offset.back());

  // Get unique array from ids and set value for hash map.
  torch::parallel_for(
      num_seeds, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
        auto thread_id = torch::get_thread_num();
        auto pos = block_offset[thread_id] + num_seeds;
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
torch::Tensor ConcurrentIdHashMap<IdType>::MapIds(
    const torch::Tensor& ids) const {
  const IdType* ids_data = ids.data_ptr<IdType>();

  torch::Tensor new_ids = torch::empty_like(ids);
  auto num_ids = new_ids.size(0);
  IdType* values_data = new_ids.data_ptr<IdType>();

  torch::parallel_for(0, num_ids, kGrainSize, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      values_data[i] = MapId(ids_data[i]);
    }
  });
  return new_ids;
}

template <typename IdType>
constexpr IdType getKeyIndex(IdType pos) {
  return 2 * pos;
}

template <typename IdType>
constexpr IdType getValueIndex(IdType pos) {
  return 2 * pos + 1;
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
  IdType* hash_map_data = hash_map_.data_ptr<IdType>();
  IdType key = hash_map_data[getKeyIndex(pos)];
  while (key != empty_key && key != id) {
    Next(&pos, &delta);
    key = hash_map_data[getKeyIndex(pos)];
  }
  if (key == empty_key) {
    throw std::out_of_range("Id not found: " + std::to_string(id));
  }
  return hash_map_data[getValueIndex(pos)];
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
  IdType* hash_map_data = hash_map_.data_ptr<IdType>();
  while (hash_map_data[getKeyIndex(pos)] != key) {
    Next(&pos, &delta);
  }

  hash_map_data[getValueIndex(pos)] = value;
}

template <typename IdType>
inline void ConcurrentIdHashMap<IdType>::InsertAndSet(IdType id, IdType value) {
  IdType pos = (id & mask_), delta = 1;
  while (AttemptInsertAt(pos, id) == InsertState::OCCUPIED) {
    Next(&pos, &delta);
  }

  hash_map_.data_ptr<IdType>()[getValueIndex(pos)] = value;
}

template <typename IdType>
void ConcurrentIdHashMap<IdType>::InsertAndSetMin(IdType id, IdType value) {
  IdType pos = (id & mask_), delta = 1;
  IdType* hash_map_data = hash_map_.data_ptr<IdType>();
  InsertState state = AttemptInsertAt(pos, id);
  while (state == InsertState::OCCUPIED) {
    Next(&pos, &delta);
    state = AttemptInsertAt(pos, id);
  }

  IdType empty_key = static_cast<IdType>(kEmptyKey);
  IdType val_pos = getValueIndex(pos);
  IdType old_val = empty_key;
  while (old_val == empty_key || old_val > value) {
    IdType replaced_val =
        CompareAndSwap(&(hash_map_data[val_pos]), old_val, value);
    if (old_val == replaced_val) break;
    old_val = replaced_val;
  }
}

template <typename IdType>
inline typename ConcurrentIdHashMap<IdType>::InsertState
ConcurrentIdHashMap<IdType>::AttemptInsertAt(int64_t pos, IdType key) {
  IdType empty_key = static_cast<IdType>(kEmptyKey);
  IdType* hash_map_data = hash_map_.data_ptr<IdType>();
  IdType old_val =
      CompareAndSwap(&(hash_map_data[getKeyIndex(pos)]), empty_key, key);
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
template class ConcurrentIdHashMap<int16_t>;
template class ConcurrentIdHashMap<int8_t>;
template class ConcurrentIdHashMap<uint8_t>;

}  // namespace sampling
}  // namespace graphbolt
