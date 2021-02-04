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



/*!
 * \brief A hashtable for mapping items to the first index at which they
 * appear in the provided data array.
 *
 * For any ID array A, one can view it as a mapping from the index `i`
 * (continuous integer range from zero) to its element `A[i]`. This hashtable
 * serves as a reverse mapping, i.e., from element `A[i]` to its index `i`. It
 * uses quadratic probing for collisions.
 *
 * \tparam IdType The type of the IDs.
 */
template<typename IdType>
class OrderedHashTable {
  public:
    struct Mapping {
      IdType key;
      IdType local;
      int64_t index;
    };

    static constexpr int DEFAULT_SCALE = 3;

    typedef Mapping* Iterator;
    typedef const Mapping* ConstIterator;

    // Must be uniform bytes for memset to work
    static constexpr IdType EmptyKey = static_cast<IdType>(-1);

    /**
    * \brief Create a new ordered hash table.
    *
    * \param size The number of items to insert into the hashtable.
    * \param ctx The device context to store the hashtable on.
    * \param scale The power of two times larger the number of buckets should
    * be than the number of items.
    * \param stream The stream to use for initializing the hashtable.
    */
    OrderedHashTable(
        const size_t size,
        DGLContext ctx,
        cudaStream_t stream,
        const int scale = DEFAULT_SCALE);

    /**
    * @brief Cleanup after the hashtable.
    */
    ~OrderedHashTable();

    /**
    * @brief Find the non-mutable mapping of a given key within the hash table.
    *
    * WARNING: The key must exist within the hashtable. Searching for a key not
    * in the hashtable is undefined behavior.
    *
    * @param id The key to search for.
    *
    * @return An iterator to the mapping.
    */
    inline __device__ ConstIterator Search(
        const IdType id) const {
      const IdType pos = SearchForPosition(id);

      return &table_[pos];
    }

    /**
    * @brief Find the mutable mapping of a given key within the hash table.
    *
    * WARNING: The key must exist within the hashtable. Searching for a key not
    * in the hashtable is undefined behavior.
    *
    * @param id The key to search for.
    *
    * @return The mapping.
    */
    inline __device__ Iterator Search(
        const IdType id) {
      const IdType pos = SearchForPosition(id);

      return &table_[pos];
    }

    /**
    * @brief Fill the hashtable with the array containing possibly duplicate
    * IDs.
    *
    * @param input The array of IDs to insert.
    * @param num_input The number of IDs to insert.
    * @param unique The list of unique IDs inserted.
    * @param num_unique The number of unique IDs inserted.
    * @param stream The stream to perform operations on.
    */
    void FillWithDuplicates(
        const IdType * const input,
        const size_t num_input,
        IdType * const unique,
        int64_t * const num_unique,
        cudaStream_t stream);

    /**
    * @brief Fill the hashtable with an array of unique keys.
    *
    * @param input The array of unique IDs.
    * @param num_input The number of keys.
    * @param stream The stream to perform operations on.
    */
    void FillWithUnique(
        const IdType * const input,
        const size_t num_input,
        cudaStream_t stream);

    /**
    * @brief Insert key-index pair into the hashtable.
    *
    * NOTE: This is public soley so that it can be access from within CUDA
    * kernels. It should not be used except for within the hashable's
    * implementation.
    *
    * @param id The ID to insert.
    * @param index The index at which the ID occured.
    *
    * @return An iterator to inserted mapping.
    */
    inline __device__ Iterator Insert(
        const IdType id,
        const size_t index);

  private:
    Mapping * table_;
    size_t size_;
    DGLContext ctx_;

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
        const IdType id) const {
      return id % size_;
    }
};

}  // cuda
}  // runtime
}  // dgl

#endif
