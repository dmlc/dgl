/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/cpu_id_hash_map.h
 * @brief Class about CPU id hash map
 */

#ifndef DGL_ARRAY_CPU_CPU_ID_HASH_MAP_H_
#define DGL_ARRAY_CPU_CPU_ID_HASH_MAP_H_

#include <dgl/aten/types.h>

#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#define COMPARE_AND_SWAP(ptr, oldval, newval, ret)                         \
  do {                                                        \
    if (sizeof(newval) == 32) {                               \
      *ret = _InterlockedCompareExchange(                     \
        reinterpret_cast<LONG*>(ptr), newval, oldval);        \
    } else if (sizeof(newval) == 64) {                        \
      *ret = _InterlockedCompareExchange64(                   \
        reinterpret_cast<LONGLONG*>(ptr), newval, oldval);    \
    } else {                                                  \
      LOG(FATAL) << "ID can only be int32 or int64";          \
    }                                                         \
  } while (0)
#elif __GNUC__
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 1)
#error "requires GCC 4.1 or greater"
#endif
#define COMPARE_AND_SWAP(ptr, oldval, newval, ret) \
  *ret = __sync_val_compare_and_swap(ptr, oldval, newval)
#else
#error "CAS not supported on this platform"
#endif

namespace dgl {
namespace aten {

/**
 * @brief A CPU targeted hashmap for mapping duplicate and non-consecutive ids
 * in the provided array to unique and consecutive ones. It utilizes multi-threading
 * to accelerate the insert and search speed. Currently it is only
 * designed to be used in `ToBlockCpu` for optimizing.
 * 
 * The hashmap should be used in two phases. With the first being creating the
 * hashmap, and then init it with an id array. After that, searching any old ids
 * to get the mappings according to your need. 
 * 
 * For example, for an array A with following entries:
 * [98, 98, 100, 99, 97, 99, 101, 100, 102]
 * Create the hashmap H with:
 * `H = CpuIdHashMap()` (1)
 * And Init it with:
 * `H.Init(A, U)` (2)  (U is the unique id array which used to store the unqiue
 * ids in original array).
 * Then U should be (The result is not exclusive as the element order is not
 * guaranteed to keep same with original array):
 * [98, 100, 99, 97, 101, 102]
 * And the hashmap should generate following mappings:
 *  * [
 *   {key: 98, value: 0},
 *   {key: 100, value: 1},
 *   {key: 99, value: 2},
 *   {key: 97, value: 3},
 *   {key: 101, value: 4},
 *   {key: 102, value: 5}
 * ]
 * Search the hashmap with array I=[98, 99, 102]:
 * H.Map(I, -1, R) (3)
 * R should be: 
 * [0, 2, 5]
**/
template <typename IdType>
class CpuIdHashMap {
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
     * @brief The value of the item inserted.
     */
      IdType value;
  };

  CpuIdHashMap();

  CpuIdHashMap(const CpuIdHashMap& other) = delete;
  CpuIdHashMap& operator=(const CpuIdHashMap& other) = delete;

  /**
   * @brief Init the hashmap with an array of ids.
   * Firstly allocating the memeory and init the entire space with empty key.
   * And then insert the items in `ids` concurrently to generate the
   * mappings, in passing returning the unique ids in `ids`. 
   *
   * @param ids The array of ids to be inserted as keys.
   *
   * @param unique_ids An id array for storing unique items in `ids`.
   * 
   * @return Number of unique items in input `ids`.
   */
  size_t Init(IdArray ids, IdArray unique_ids);

  /**
   * @brief Find the mappings of given keys.
   *
   * @param ids The keys to map for.
   * 
   * @param default_val Default value for missing keys.
   * 
   * @param new_ids Array for storing results.
   *
   */
  void Map(IdArray ids, IdType default_val, IdArray new_ids) const;

  ~CpuIdHashMap();

 private:
  IdType mapId(IdType id, IdType default_val) const;

  size_t fillInIds(size_t num_ids,
    const IdType* ids_data, IdArray unique_ids);

  void next(IdType* pos, IdType* delta) const;

  void insert(IdType id, std::vector<int16_t>* valid, size_t index);

  void set(IdType key, IdType value);

  bool attemptInsertAt(int64_t pos, IdType key,
    std::vector<int16_t>* valid, size_t index);

 private:
  static constexpr IdType k_empty_key = static_cast<IdType>(-1);
  static constexpr int grain_size = 1024;

  Mapping* _hmap;
  IdType _mask;
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_CPU_ID_HASH_MAP_H_
