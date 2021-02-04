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

template<typename>
class OrderedHashTable;

template<typename IdType>
class DeviceOrderedHashTable {
  public:
    /**
    * @brief An entry in the hashtable.
    */
    struct Mapping {
      /**
      * @brief The ID of the item inserted.
      */
      IdType key;
      /**
      * @brief The index of the item in the unique list.
      */
      IdType local;
      /**
      * @brief The index of the item when inserted into the hashtable (e.g.,
      * the index within the array passed into FillWithDuplicates()).
      */
      int64_t index;
    };

    typedef const Mapping* ConstIterator;

    // Must be uniform bytes for memset to work
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);

    DeviceOrderedHashTable(
        const DeviceOrderedHashTable& other) = default;
    DeviceOrderedHashTable& operator=(
        const DeviceOrderedHashTable& other) = default;

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

  protected:
    const Mapping * table_;
    size_t size_;

    explicit DeviceOrderedHashTable(
        const Mapping * table,
        size_t size);

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
        assert(table_[pos].key != kEmptyKey);
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

    friend class OrderedHashTable<IdType>;
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
    static constexpr int kDefaultScale = 3;

    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;

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
        const int scale = kDefaultScale);

    /**
    * @brief Cleanup after the hashtable.
    */
    ~OrderedHashTable();

    // Disable copying 
    OrderedHashTable(
        const OrderedHashTable& other) = delete;
    OrderedHashTable& operator=(
        const OrderedHashTable& other) = delete;

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
    * @brief Get a verison of the hashtable usable from device functions.
    * 
    * @return This hashtable.
    */
    DeviceOrderedHashTable<IdType> ToDevice() const;

  private:
    Mapping * table_;
    size_t size_;
    DGLContext ctx_;

};


}  // cuda
}  // runtime
}  // dgl

#endif
