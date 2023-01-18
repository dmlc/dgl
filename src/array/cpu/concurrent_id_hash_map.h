/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/concurrent_id_hash_map.h
 * @brief Utility classes and functions for DGL arrays.
 */

#ifndef DGL_ARRAY_CONCURRENT_HASH_MAP_H_
#define DGL_ARRAY_CONCURRENT_HASH_MAP_H_

#include <vector>
#include <atomic>
#include <omp.h>
#include <algorithm>
#include <dgl/aten/types.h>

#define PICK_THREAD_NUM 16

namespace dgl {
namespace aten {

template <typename IdType>
class ConcurrentIdHashMap {
 public:
    struct Mapping {
        std::atomic<IdType> key;
        IdType value;
    };

    ConcurrentIdHashMap() : hmap(NULL) {}

    void Init(size_t n) {
        size_t new_size;
        for (new_size = 1; new_size < 3 * n; new_size *= 2) {}
        mask =  static_cast<IdType>(new_size - 1);
        
        DGLContext ctx;
        ctx.device_type = kDGLCPU;
        auto device = runtime::DeviceAPI::Get(ctx);
        hmap = static_cast<Mapping*>(
            device->AllocWorkspace(ctx, sizeof(Mapping) * new_size));

        // parallel memset can accelerate it
        memset((void*)hmap, -1, sizeof(Mapping)*new_size);
    }

    size_t Update(IdArray ids,  IdArray& unique_ids) {
        const IdType* ids_data = ids.Ptr<IdType>();
        const size_t size = ids->shape[0];
        std::vector<bool> valid(size);
        
        size_t thread_num = std::min(PICK_THREAD_NUM, omp_get_max_threads());
        if (size < thread_num) {
            thread_num = 1;
        }

        std::vector<int64_t> block_offset(thread_num + 1, 0);
        size_t block_size = (size + thread_num - 1) / thread_num;
        IdType* unique_ids_data = unique_ids.Ptr<IdType>();

        // Insert all elements in this loop. 
        #pragma omp parallel
        {
        #pragma omp for
            for(size_t i=0; i < thread_num; i++) {
                    size_t start = i *block_size;
                    size_t end = (i + 1) *block_size;
                    end = std::min(end, size);
                    size_t count = 0;
                    for (size_t index = start; index < end; ++index) {
                        insert_cas(ids_data[index], valid, index);
                        if (valid[index]) {
                            count++;
                        }
                    }
                    block_offset[i + 1] = count;
            }
        }

        // Get ExclusiveSum of each block.
        for (size_t i = 1; i <= thread_num; i++) {
            block_offset[i] += block_offset[i - 1]; 
        }

        // Get the unique array of ids and set value of hash map.
        #pragma omp parallel
        {
        #pragma omp for
            for(size_t i=0; i < thread_num; i++) {
                size_t start = i *block_size;
                size_t end = (i + 1) *block_size;
                end = std::min(end, size);
                size_t pos = block_offset[i];
                for (size_t index = start; index < end; ++index) {
                    if (valid[index]) {
                        unique_ids_data[pos] = ids_data[index];
                        set_value(ids_data[index], pos);
                        pos = pos + 1;
                    }
                }
            }
        }
        
        return block_offset[thread_num];
    }

    // Return true if the given id is contained in this hashmap.
    bool Contains(IdType id) const {
        IdType j;
        for (j = (id&mask); hmap[j].key != -1 && hmap[j].key != id; j = ((j + 1) & mask)) {}
        return hmap[j].key != -1;
    }

    // Return the new id of the given id. If the given id is not contained
    // in the hash map, returns the default_val instead.
    IdType Map(IdType id, IdType default_val) const {
        IdType pos = (id & mask);
        IdType delta = 10;
        while (hmap[pos].key != -1 && hmap[pos].key != id) {
            pos = ((pos + delta*delta) & mask);
            delta += 1;
        }

        if (hmap[pos].key == -1) {
            return default_val;
        } else {
            return hmap[pos].value;
        }
    }

    void Map(IdArray ids, IdType default_val, IdArray& new_ids) const {
        const IdType* ids_data = ids.Ptr<IdType>();
        const size_t len = static_cast<size_t>(ids->shape[0]);
        IdType* values_data = new_ids.Ptr<IdType>();

        size_t thread_num = omp_get_max_threads();
        int block_size = (len + thread_num - 1) / thread_num;
        #pragma omp parallel
        {
        #pragma omp for
            for (size_t i = 0; i < thread_num; ++i) {
                size_t start = i *block_size;
                size_t end = (i + 1) *block_size;
                end = std::min(end, len);

                for (size_t index = start; index < end; ++index) {
                    values_data[index] = Map(ids_data[index], default_val);
                }
            }
        }
    }

    ~ConcurrentIdHashMap() {
        if (hmap != NULL) {
            DGLContext ctx;
            ctx.device_type = kDGLCPU;
            auto device = runtime::DeviceAPI::Get(ctx);
            device->FreeWorkspace(ctx, hmap);
        }
    }

    void insert_cas(IdType id, std::vector<bool>& valid, size_t index) {
        IdType pos = (id & mask);
        IdType delta = 10;
        while (!attempt_insert_at(pos, id, valid, index)) {
            pos = ((pos + delta*delta) & mask);
            delta += 1;
        }
    }
  
    // Key must exist.
    void set_value(IdType k, IdType v) {
        IdType pos = (k & mask), delta = 10;
        while (hmap[pos].key != k) {
            pos = ((pos + delta*delta) & mask);
            delta+=1;
        }
        
        hmap[pos].value = v;
    }

    bool attempt_insert_at(int64_t pos, IdType key, std::vector<bool>& valid, size_t index) {
        IdType a = kEmptyKey;
        bool success = hmap[pos].key.compare_exchange_weak(a, key);
        if (success) {
            valid[index] = true;
            return true;
        }
        else {
            if(hmap[pos].key == key)
                return true;
            else
                return false; 
        }
    }

 protected:
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
    Mapping* hmap;
    IdType mask;
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CONCURRENT_HASH_MAP_H_