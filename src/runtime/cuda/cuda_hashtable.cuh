/*!
 *  Copyright (c) 2021 by Contributors
 * \file runtime/cuda/cuda_device_common.cuh 
 * \brief Device level functions for within cuda kernels. 
 */

#ifndef DGL_RUNTIME_CUDA_CUDA_HASHTABLE_CUH_
#define DGL_RUNTIME_CUDA_CUDA_HASHTABLE_CUH_

#include <cuda_runtime.h>

#include "cuda_device_common.cuh"

namespace dgl {
namespace runtime {
namespace cuda {

template<typename IdType>
struct EmptyKey {
  constexpr static const IdType value = static_cast<IdType>(-1);
};

template<typename IdType>
class HashTable
{
  public:
    struct Mapping {
      IdType key;
      IdType local;
      int64_t index;
    };

    // Must be uniform bytes for memset to work
    static constexpr IdType EmptyKey = static_cast<IdType>(-1);

    static size_t RequiredWorkspace(
        const size_t size)
    {
      return sizeof(Mapping)*size;
    }

    HashTable(
        size_t size,
        void * workspace,
        size_t workspace_size,
        cudaStream_t stream) :
      table_(static_cast<Mapping*>(workspace)),
      size_(size)
    {
      if (RequiredWorkspace(size) > workspace_size) {
        throw std::runtime_error("Insufficient workspace: requires " +
            std::to_string(RequiredWorkspace(size)) + " but only provided " +
            std::to_string(workspace_size));
      }

      CUDA_CALL(cudaMemsetAsync(
        table_,
        EmptyKey,
        sizeof(Mapping)*size_,
        stream));
    }

    inline __device__ Mapping& operator[](const size_t idx) {
      return table_[idx];
    }

    inline __device__ const Mapping& operator[](const size_t idx) const {
      return table_[idx];
    }

    inline __device__ __host__ size_t size() const {
      return size_;
    }

    inline __device__ size_t hash(
        const IdType key) const
    {
      return key % size_;
    }

    inline __device__ bool attempt_insert_at(
        const size_t pos,
        const IdType id,
        const size_t index) {
      const IdType key = AtomicCAS(&table_[pos].key, EmptyKey, id);
      if (key == EmptyKey || key == id) {
        // we either set a match key, or found a matching key, so then place the
        // minimum index in position. Match the type of atomicMin, so ignore
        // linting
        atomicMin(reinterpret_cast<unsigned long long*>(&table_[pos].index), // NOLINT
            static_cast<unsigned long long>(index)); // NOLINT
        return true;
      } else {
        // we need to search elsewhere
        return false;
      }
    }

    inline __device__ size_t insert(
        const IdType id,
        const size_t index) {
      size_t pos = hash(id);

      // linearly scan for an empty slot or matching entry
      IdType delta = 1;
      while (!attempt_insert_at(pos, id, index)) {
        pos = hash(pos+delta);
        delta +=1;
      }

      return pos;
    }

    inline __device__ IdType search_for_pos(
        const IdType id) const {
      IdType pos = hash(id);

      // linearly scan for matching entry
      IdType delta = 1;
      while (table_[pos].key != id) {
        assert(table_[pos].key != EmptyKey);
        pos = hash(pos+delta);
        delta +=1;
      }
      assert(pos < size_);

      return pos;
    }

    inline __device__ const typename HashTable<IdType>::Mapping * search(
        const IdType id) const {
      const IdType pos = search_for_pos(id);

      return &table_[pos];
    }

    inline __device__ typename HashTable<IdType>::Mapping * search(
        const IdType id) {
      const IdType pos = search_for_pos(id);

      return &table_[pos];
    }

  private:
    Mapping * table_;
    size_t size_;
};

}  // cuda
}  // runtime
}  // dgl

#endif
