/**
 *  Copyright (c) 2019 by Contributors
 * @file dgl/concurrent_id_hash_map.h
 * @brief Utility classes and functions for DGL arrays.
 */

#ifndef DGL_ARRAY_CPU_ID_HASH_MAP_H_
#define DGL_ARRAY_CPU_ID_HASH_MAP_H_

#include <vector>
#include <atomic>
#include <omp.h>
#include <algorithm>

#include <dgl/aten/types.h>

namespace dgl {
namespace aten {

template <typename IdType>
class CpuIdHashMap {
 public:
    struct Mapping {
        std::atomic<IdType> key;
        IdType value;
    };

    static constexpr IdType kEmptyKey = static_cast<IdType>(0);
    static constexpr int grain_size = 1024;
    
    explicit CpuIdHashMap(DGLContext ctx);

    CpuIdHashMap(const CpuIdHashMap& other) = delete;
    CpuIdHashMap& operator=(const CpuIdHashMap& other) = delete;

    size_t Init(IdArray ids, IdArray unique_ids);

    size_t FillInIds(size_t num_ids, const IdType* ids_data, IdArray unique_ids);

    void Map(IdArray ids, IdType default_val, IdArray new_ids) const;

    ~CpuIdHashMap();
   
    // Return the new id of the given id. If the given id is not contained
    // in the hash map, returns the default_val instead.
    IdType map(IdType id, IdType default_val) const;
    
    void insert_cas(IdType id, std::vector<bool>& valid, size_t index);
  
    // Key must exist.
    void set_value(IdType k, IdType v);

    bool attempt_insert_at(int64_t pos, IdType key, std::vector<bool>& valid, size_t index);

 private:
    DGLContext _ctx;
    Mapping* _hmap;
    IdType _mask;
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ID_HASH_MAP_H_
