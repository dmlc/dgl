/*!
 *  Copyright (c) 2021 by Contributors
 * \file runtime/cuda/cuda_device_common.cuh 
 * \brief Device level functions for within cuda kernels. 
 */

#ifndef DGL_RUNTIME_CUDA_CUDA_HASHTABLE_CUH_
#define DGL_RUNTIME_CUDA_CUDA_HASHTABLE_CUH_

#include <dgl/runtime/c_runtime_api.h>

#include "cuda_runtime.h"
#include "cuda_common.h"
#include "cuda_device_common.cuh"

namespace dgl {
namespace runtime {
namespace cuda {

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



/**
* \brief An on GPU hash table implementation, which preserves the order of
* inserted elements. It uses quadratic probing for collisions. The size/number
* buckets the hash table is created with, should be much greater than the
* number of unique items to be inserted. Using less than the number of 
*/
template<typename IdType>
class OrderedHashTable
{
  public:
    struct Mapping {
      IdType key;
      IdType local;
      int64_t index;
    };

    static constexpr int DEFAULT_SCALE = 3;

    typedef Mapping* Iterator;

    // Must be uniform bytes for memset to work
    static constexpr IdType EmptyKey = static_cast<IdType>(-1);

    /**
    * \brief Get the amount of workspace the hashtable needs to function, given
    * the number of positions.
    *
    * \param size The number of items to insert into the hash table.
    * \param scale The power of two times larger the number of buckets should
    * be than the number of items. Increasing should decrease the number of
    * collisions in the hashtable, at the expense of memory consumption.
    *
    * \return The required number of bytes for the hashtable.
    */
    static size_t RequiredWorkspace(
        const size_t size,
        const int scale = DEFAULT_SCALE)
    {
      return sizeof(Mapping)*TableSize(size, scale);
    }

    /**
    * \brief Create a new ordered hash table.
    *
    * \param size The number of items to insert into the hashtable.
    * \param workspace The allocated workspace on the GPU.
    * \param workspace_size The size of the provided workspace.
    * \param scale The power of two times larger the number of buckets should
    * be than the number of items.
    * \param stream The stream to use.
    */
    OrderedHashTable(
        const size_t size,
        void * workspace,
        const size_t workspace_size,
        cudaStream_t stream,
        const int scale = DEFAULT_SCALE) :
      table_(static_cast<Mapping*>(workspace)),
      size_(TableSize(size, scale))
    {
      if (RequiredWorkspace(size, scale) > workspace_size) {
        throw std::runtime_error("Insufficient workspace: requires " +
            std::to_string(RequiredWorkspace(size, scale)) + " but only provided " +
            std::to_string(workspace_size));
      }

      CUDA_CALL(cudaMemsetAsync(
        table_,
        EmptyKey,
        sizeof(Mapping)*size_,
        stream));
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

    void FillWithDuplicates(
        const IdType * const input,
        const size_t num_input,
        IdType * const unique,
        int64_t * const num_unique,
        DGLContext ctx,
        cudaStream_t stream); 

    void FillWithUnique(
        const IdType * const input,
        const size_t num_input,
        DGLContext ctx,
        cudaStream_t stream); 

    inline __device__ Iterator Insert(
        const IdType id,
        const size_t index);

  private:
    Mapping * table_;
    size_t size_;

    /**
    * \brief Calculate the size of the table for the given number of items.
    *
    * \param num The number of items.
    * \param scale The power of two times large the number of buckets should be
    * than the number of items.
    *
    * \return The number of buckets in the table.
    */
    static size_t TableSize(
        const size_t num,
        const int scale)
    {
      const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
      return next_pow2<<scale;
    }


    /**
    * \brief Attempt to insert into the hash table at a specific location.
    *
    * \param pos The position to insert at.
    * \param id The ID to insert into the hash table.
    * \param index The original index of the item being inserted.
    *
    * \return True, if the insertion was successful.
    */
    inline __device__ bool AttemptInsertAt(
        const size_t pos,
        const IdType id,
        const size_t index);

    /**
    * \brief Search for an item in the hash table which is known to exist.
    *
    * WARNING: If the ID searched for does not exist within the hashtable, this
    * function will never return.
    *
    * \param id The ID of the item to search for.
    *
    * \return The the position of the item in the hashtable.
    */
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

    /**
    * \brief Hash an ID to a to a position in the hash table.
    *
    * \param id The ID to hash.
    *
    * \return The hash.
    */
    inline __device__ size_t Hash(
        const IdType id) const
    {
      return id % size_;
    }
};

}  // cuda
}  // runtime
}  // dgl

#endif
