/*!
 *  Copyright (c) 2021 by Contributors
 * \file runtime/cuda/cuda_device_common.cuh 
 * \brief Device level functions for within cuda kernels. 
 */

#ifndef DGL_RUNTIME_CUDA_CUDA_HASHTABLE_CUH_
#define DGL_RUNTIME_CUDA_CUDA_HASHTABLE_CUH_

#include <cuda_runtime.h>

namespace dgl {
namespace runtime {
namespace cuda {

template<typename IdType>
struct Mapping {
  IdType key;
  IdType local;
  int64_t index;
};


template<typename IdType>
struct EmptyKey {
  constexpr static const IdType value = static_cast<IdType>(-1);
};


// GPU Code ///////////////////////////////////////////////////////////////////

template<typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(
      const IdType running_total) :
    running_total_(running_total) {
  }

  __device__ IdType operator()(const IdType block_aggregate) {
      const IdType old_prefix = running_total_;
      running_total_ += block_aggregate;
      return old_prefix;
  }
};


template<typename IdType>
inline __device__ bool attempt_insert_at(
    const size_t pos,
    const IdType id,
    const size_t index,
    Mapping<IdType> * const table) {
  const IdType key = AtomicCAS(&table[pos].key, EmptyKey<IdType>::value, id);
  if (key == EmptyKey<IdType>::value || key == id) {
    // we either set a match key, or found a matching key, so then place the
    // minimum index in position. Match the type of atomicMin, so ignore
    // linting
    atomicMin(reinterpret_cast<unsigned long long*>(&table[pos].index), // NOLINT
        static_cast<unsigned long long>(index)); // NOLINT
    return true;
  } else {
    // we need to search elsewhere
    return false;
  }
}


template<typename IdType>
inline __device__ size_t insert_hashmap(
    const IdType id,
    const size_t index,
    Mapping<IdType> * const table,
    const size_t table_size) {
  size_t pos = id % table_size;

  // linearly scan for an empty slot or matching entry
  IdType delta = 1;
  while (!attempt_insert_at(pos, id, index, table)) {
    pos = (pos+delta) % table_size;
    delta +=1;
  }

  return pos;
}


template<typename IdType>
inline __device__ IdType search_hashmap_for_pos(
    const IdType id,
    const Mapping<IdType> * const table,
    const IdType table_size) {
  IdType pos = id % table_size;

  // linearly scan for matching entry
  IdType delta = 1;
  while (table[pos].key != id) {
    assert(table[pos].key != EmptyKey<IdType>::value);
    pos = (pos+delta) % table_size;
    delta +=1;
  }
  assert(pos < table_size);

  return pos;
}


template<typename IdType>
inline __device__ const Mapping<IdType> * search_hashmap(
    const IdType id,
    const Mapping<IdType> * const table,
    const IdType table_size) {
  const IdType pos = search_hashmap_for_pos(id, table, table_size);

  return table+pos;
}


template<typename IdType>
inline __device__ Mapping<IdType> * search_hashmap(
    const IdType id,
    Mapping<IdType> * const table,
    const IdType table_size) {
  const IdType pos = search_hashmap_for_pos(id, table, table_size);

  return table+pos;
}



}  // cuda
}  // runtime
}  // dgl

#endif
