/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/cpu_id_hash_map.cc
 * @brief Class about CPU id hash map
 */

#include "cpu_id_hash_map.h"

#include <dgl/runtime/parallel_for.h>
#include <dgl/runtime/device_api.h>

#include <cmath>

using namespace dgl::runtime;

namespace dgl {
namespace aten {

template <typename IdType>
CpuIdHashMap<IdType>::CpuIdHashMap() : _hmap(nullptr), _mask(0) {
}

template <typename IdType>
size_t CpuIdHashMap<IdType>::Init(IdArray ids, IdArray unique_ids) {
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t num = ids->shape[0];

  CHECK_GT(num, 0);

  size_t capcacity = 1;
  capcacity = capcacity << static_cast<size_t>(1 + std::log2(num * 3));
  _mask =  static_cast<IdType>(capcacity - 1);

  DGLContext ctx = DGLContext{kDGLCPU, 0};
  auto device = DeviceAPI::Get(ctx);
  _hmap = static_cast<Mapping*>(
      device->AllocWorkspace(ctx, sizeof(Mapping) * capcacity));
  memset(_hmap, -1, sizeof(Mapping) * capcacity);

  return fillInIds(num, ids_data, unique_ids);
}

template <typename IdType>
void CpuIdHashMap<IdType>::Map(IdArray ids,
    IdType default_val, IdArray new_ids) const {
  const IdType* ids_data = ids.Ptr<IdType>();
  const size_t len = static_cast<size_t>(ids->shape[0]);
  IdType* values_data = new_ids.Ptr<IdType>();

  parallel_for(0, len, grain_size, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
        values_data[i] = mapId(ids_data[i], default_val);
    }
  });
}

template <typename IdType>
CpuIdHashMap<IdType>::~CpuIdHashMap() {
  if (_hmap != nullptr) {
      DGLContext ctx = DGLContext{kDGLCPU, 0};
      auto device = DeviceAPI::Get(ctx);
      device->FreeWorkspace(ctx, _hmap);
  }
}

template <typename IdType>
size_t CpuIdHashMap<IdType>::fillInIds(size_t num_ids,
    const IdType* ids_data, IdArray unique_ids) {
  // Use `int16_t` instead of `bool` here. As vector<bool> is an exception
  // for whom updating different elements from different threads is unsafe.
  // see https://en.cppreference.com/w/cpp/container#Thread_safety.
  std::vector<int16_t> valid(num_ids);
  auto thread_num = compute_num_threads(0, num_ids, grain_size);
  std::vector<size_t> block_offset(thread_num + 1, 0);
  IdType* unique_ids_data = unique_ids.Ptr<IdType>();

  // Insert all elements in this loop.
  parallel_for(0, num_ids, grain_size, [&](int64_t s, int64_t e) {
      size_t count = 0;
      for (int64_t i = s; i < e; i++) {
          insert(ids_data[i], &valid, i);
          if (valid[i]) {
              count++;
          }
      }

      size_t tid = omp_get_thread_num();
      block_offset[tid + 1] = count;
  });

  // Get ExclusiveSum of each block.
  for (size_t i = 1; i <= thread_num; i++) {
      block_offset[i] += block_offset[i - 1];
  }

  // Get unique array from ids and set value for hash map.
  parallel_for(0, num_ids, grain_size, [&](int64_t s, int64_t e) {
      auto tid = omp_get_thread_num();
      auto pos = block_offset[tid];
      for (int64_t i = s; i < e; i++) {
          if (valid[i]) {
              unique_ids_data[pos] = ids_data[i];
              set(ids_data[i], pos);
              pos = pos + 1;
          }
      }
  });

  return block_offset[thread_num];
}

template <typename IdType>
IdType CpuIdHashMap<IdType>::mapId(IdType id, IdType default_val) const {
  IdType pos = (id & _mask);
  IdType delta = 1;
  while (_hmap[pos].key != k_empty_key && _hmap[pos].key != id) {
      next(&pos, &delta);
  }

  if (_hmap[pos].key == k_empty_key) {
      return default_val;
  } else {
      return _hmap[pos].value;
  }
}

template <typename IdType>
void CpuIdHashMap<IdType>::next(IdType* pos, IdType* delta) const {
  *pos = (*pos + (*delta) * (*delta)) & _mask;
  *delta = *delta + 1;
}

template <typename IdType>
void CpuIdHashMap<IdType>::insert(IdType id,
    std::vector<int16_t>* valid,
    size_t index) {
  IdType pos = (id & _mask), delta = 1;
  while (!attemptInsertAt(pos, id, valid, index)) {
      next(&pos, &delta);
  }
}

template <typename IdType>
void CpuIdHashMap<IdType>::set(IdType key, IdType value) {
  IdType pos = (key & _mask), delta = 1;
  while (_hmap[pos].key != key) {
      next(&pos, &delta);
  }

  _hmap[pos].value = value;
}

template <typename IdType>
bool CpuIdHashMap<IdType>::attemptInsertAt(int64_t pos, IdType key,
    std::vector<int16_t>* valid, size_t index) {
  IdType empty_key = k_empty_key;
  IdType old_val = k_empty_key;
  COMPARE_AND_SWAP(&(_hmap[pos].key), empty_key, key, &old_val);

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
