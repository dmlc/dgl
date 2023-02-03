/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/cpu_id_hash_map.cc
 * @brief Class about CPU id hash map
 */

#include "cpu_id_hash_map.h"

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/parallel_for.h>

#include <cmath>
#include <numeric>

using namespace dgl::runtime;

namespace {
  static constexpr int64_t kEmptyKey = -1;
  static constexpr int kGrainSize = 256;
}  // namespace


namespace dgl {
namespace aten {

template <typename IdType>
IdType CpuIdHashMap<IdType>::CompareAndSwap(
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
CpuIdHashMap<IdType>::CpuIdHashMap() : hmap_(nullptr), mask_(0) {}

template <typename IdType>
IdArray CpuIdHashMap<IdType>::Init(const IdArray ids) {
  CHECK_EQ(ids.defined(), true);
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t num = static_cast<size_t>(ids->shape[0]);
  CHECK_GT(num, 0);
  size_t capcacity = 1;
  capcacity = capcacity << static_cast<size_t>(1 + std::log2(num * 3));
  mask_ = static_cast<IdType>(capcacity - 1);

  DGLContext ctx = DGLContext{kDGLCPU, 0};
  auto device = DeviceAPI::Get(ctx);
  hmap_ = static_cast<Mapping*>(
      device->AllocWorkspace(ctx, sizeof(Mapping) * capcacity));
  memset(hmap_, -1, sizeof(Mapping) * capcacity);

  IdArray unique_ids = NewIdArray(num, ctx, sizeof(IdType) * 8);
  return FillInIds(num, ids_data, unique_ids);
}

template <typename IdType>
IdArray CpuIdHashMap<IdType>::Map(const IdArray ids) const {
  CHECK_EQ(ids.defined(), true);
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t len = static_cast<size_t>(ids->shape[0]);
  CHECK_GT(len, 0);

  DGLContext ctx = DGLContext{kDGLCPU, 0};
  IdArray new_ids = NewIdArray(len, ctx, sizeof(IdType) * 8);
  IdType* values_data = new_ids.Ptr<IdType>();

  parallel_for(0, len, kGrainSize, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      values_data[i] = MapId(ids_data[i]);
    }
  });
  return new_ids;
}

template <typename IdType>
CpuIdHashMap<IdType>::~CpuIdHashMap() {
  if (hmap_ != nullptr) {
    DGLContext ctx = DGLContext{kDGLCPU, 0};
    auto device = DeviceAPI::Get(ctx);
    device->FreeWorkspace(ctx, hmap_);
  }
}

template <typename IdType>
IdArray CpuIdHashMap<IdType>::FillInIds(
    size_t num_ids, const IdType* ids_data, IdArray unique_ids) {
  // Use `int16_t` instead of `bool` here. As vector<bool> is an exception
  // for whom updating different elements from different threads is unsafe.
  // see https://en.cppreference.com/w/cpp/container#Thread_safety.
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
inline void CpuIdHashMap<IdType>::Next(IdType* pos, IdType* delta) const {
  *pos = (*pos + (*delta) * (*delta)) & mask_;
  *delta = *delta + 1;
}

template <typename IdType>
IdType CpuIdHashMap<IdType>::MapId(IdType id) const {
  IdType pos = (id & mask_);
  IdType delta = 1;
  IdType empty_key = static_cast<IdType>(kEmptyKey);
  while (hmap_[pos].key != empty_key && hmap_[pos].key != id) {
    Next(&pos, &delta);
  }
  return hmap_[pos].value;
}

template <typename IdType>
void CpuIdHashMap<IdType>::Insert(
    IdType id, std::vector<int16_t>* valid, size_t index) {
  IdType pos = (id & mask_);
  IdType delta = 1;
  while (!AttemptInsertAt(pos, id, valid, index)) {
    Next(&pos, &delta);
  }
}

template <typename IdType>
void CpuIdHashMap<IdType>::Set(IdType key, IdType value) {
  IdType pos = (key & mask_);
  IdType delta = 1;
  while (hmap_[pos].key != key) {
    Next(&pos, &delta);
  }

  hmap_[pos].value = value;
}

template <typename IdType>
bool CpuIdHashMap<IdType>::AttemptInsertAt(
    int64_t pos, IdType key, std::vector<int16_t>* valid, size_t index) {
  IdType empty_key = static_cast<IdType>(kEmptyKey);;
  IdType old_val = CompareAndSwap(&(hmap_[pos].key), empty_key, key);

  if (old_val == empty_key) {
    (*valid)[index] = true;
    return true;
  } else {
    if (old_val == key)
      return true;
    else
      return false;
  }
}

template class CpuIdHashMap<int32_t>;
template class CpuIdHashMap<int64_t>;

}  // namespace aten
}  // namespace dgl
