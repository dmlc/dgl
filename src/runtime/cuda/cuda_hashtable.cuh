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
class OrderedHashTable
{
  public:
    struct Mapping {
      IdType key;
      IdType local;
      int64_t index;
    };

    typedef Mapping* Iterator;

    // Must be uniform bytes for memset to work
    static constexpr IdType EmptyKey = static_cast<IdType>(-1);

    static size_t RequiredWorkspace(
        const size_t size)
    {
      return sizeof(Mapping)*size;
    }

    OrderedHashTable(
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

    inline __device__ Iterator Insert(
        const IdType id,
        const size_t index) {
      size_t pos = Hash(id);

      // linearly scan for an empty slot or matching entry
      IdType delta = 1;
      while (!AttemptInsertAt(pos, id, index)) {
        pos = Hash(pos+delta);
        delta +=1;
      }

      return table_+pos;
    }

    inline __device__ const typename OrderedHashTable<IdType>::Mapping * Search(
        const IdType id) const {
      const IdType pos = SearchForPosition(id);

      return &table_[pos];
    }

    inline __device__ typename OrderedHashTable<IdType>::Mapping * Search(
        const IdType id) {
      const IdType pos = SearchForPosition(id);

      return &table_[pos];
    }

  private:
    Mapping * table_;
    size_t size_;

    inline __device__ bool AttemptInsertAt(
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

    inline __device__ IdType SearchForPosition(
        const IdType id) const {
      IdType pos = Hash(id);

      // linearly scan for matching entry
      IdType delta = 1;
      while (table_[pos].key != id) {
        assert(table_[pos].key != EmptyKey);
        pos = Hash(pos+delta);
        delta +=1;
      }
      assert(pos < size_);

      return pos;
    }

    inline __device__ size_t Hash(
        const IdType key) const
    {
      return key % size_;
    }
};

}  // cuda
}  // runtime
}  // dgl

#endif
