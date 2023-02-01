/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/cpu_id_hash_map.cc
 * @brief Class about CPU id hash map
 */

#include <cmath>

#include <dgl/runtime/parallel_for.h>
#include <dgl/runtime/device_api.h>

#include "cpu_id_hash_map.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

template <typename IdType>
CpuIdHashMap<IdType>::CpuIdHashMap(DGLContext ctx) : _hmap(nullptr), _ctx(ctx) {
}

template <typename IdType>
size_t CpuIdHashMap<IdType>::Init(IdArray ids, IdArray unique_ids) {
    const IdType* ids_data = ids.Ptr<IdType>();
    const size_t num = ids->shape[0];

    CHECK_GT(num, 0);

    size_t capcacity = 1 << static_cast<size_t>(1 + std::log2(num * 3));
    _mask =  static_cast<IdType>(capcacity - 1);

    auto device = DeviceAPI::Get(_ctx);
    _hmap = static_cast<Mapping*>(
        device->AllocWorkspace(_ctx, sizeof(Mapping) * capcacity));
    memset(_hmap, -1, sizeof(Mapping) * capcacity);

    return fillInIds(num, ids_data, unique_ids);
}

template <typename IdType>
size_t CpuIdHashMap<IdType>::fillInIds(size_t num_ids, const IdType* ids_data, IdArray unique_ids) {
    // Use `short` instead of `bool` here. As vector<bool> is an exception that updating
    // different elements from different threads is unsafe for it.
    // see https://en.cppreference.com/w/cpp/container#Thread_safety.
    std::vector<short> valid(num_ids);
    auto thread_num = compute_num_threads(0, num_ids, grain_size);
    std::vector<size_t> block_offset(thread_num + 1, 0);
    IdType* unique_ids_data = unique_ids.Ptr<IdType>();

    // Insert all elements in this loop. 
    parallel_for(0, num_ids, grain_size, [&](int64_t s, int64_t e) {
        size_t count = 0;
        for (int64_t i = s; i < e; i++) {
            insert_cas(ids_data[i], valid, i);
            if (valid[i]) {
                count++;
            }
        }

        auto tid = omp_get_thread_num();
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
                set_value(ids_data[i], pos);
                pos = pos + 1;
            }
        }
    });
    
    return block_offset[thread_num];
}

template <typename IdType>
CpuIdHashMap<IdType>::~CpuIdHashMap() {
    if (_hmap != nullptr) {
        auto device = DeviceAPI::Get(_ctx);
        device->FreeWorkspace(_ctx, _hmap);
    }
}

template <typename IdType>
void CpuIdHashMap<IdType>::Map(IdArray ids,
    IdType default_val, IdArray new_ids) const {
    const IdType* ids_data = ids.Ptr<IdType>();
    const size_t len = static_cast<size_t>(ids->shape[0]);
    IdType* values_data = new_ids.Ptr<IdType>();

    parallel_for(0, len, grain_size, [&](int64_t s, int64_t e) {
        for (int64_t i = s; i < e; i++) {
            values_data[i] = map(ids_data[i], default_val);
        }
    });
}

template <typename IdType>
IdType CpuIdHashMap<IdType>::map(IdType id, IdType default_val) const {
    IdType pos = (id & _mask);
    IdType delta = 1;
    while (_hmap[pos].key != kEmptyKey && _hmap[pos].key != id) {
        next(pos, delta);
    }

    if (_hmap[pos].key == kEmptyKey) {
        return default_val;
    } else {
        return _hmap[pos].value;
    }
}

template <typename IdType>
void CpuIdHashMap<IdType>::next(IdType& pos, IdType&delta) const {
    pos = ((pos + delta*delta) & _mask);
    delta += 1;
}

template <typename IdType>
void CpuIdHashMap<IdType>::insert_cas(IdType id, std::vector<short>& valid, size_t index) {
    IdType pos = (id & _mask), delta = 1;
    while (!attempt_insert_at(pos, id, valid, index)) {
        next(pos, delta);
    }
}
  
template <typename IdType>
void CpuIdHashMap<IdType>::set_value(IdType k, IdType v) {
    IdType pos = (k & _mask), delta = 1;
    while (_hmap[pos].key != k) {
        next(pos, delta);
    }

    _hmap[pos].value = v;
}

template <typename IdType>
bool CpuIdHashMap<IdType>::attempt_insert_at(int64_t pos,
    IdType key, std::vector<short>& valid, size_t index) {
    IdType empty_key = kEmptyKey;
    
    IdType old_val = CAS(&(_hmap[pos].key), empty_key, key);

    if (old_val == empty_key) {
        valid[index] = true;
        return true;
    }
    else {
        if(old_val == key)
            return true;
        else
            return false; 
    }
}

template class CpuIdHashMap<int32_t>;
template class CpuIdHashMap<int64_t>;

}  // namespace aten
}  // namespace dgl
