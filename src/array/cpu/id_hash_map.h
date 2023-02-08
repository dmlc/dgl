/**
 *  Copyright (c) 2023 by Contributors
 * @file array/cpu/id_hash_map.h
 * @brief Class about id hash map
 */

#ifndef DGL_ARRAY_CPU_ID_HASH_MAP_H_
#define DGL_ARRAY_CPU_ID_HASH_MAP_H_

#include <dgl/aten/types.h>
#include <dgl/runtime/device_api.h>

#include <memory>
#include <vector>

namespace dgl {
namespace aten {

/**
 * @brief A CPU targeted hashmap for mapping duplicate and non-consecutive ids
 * in the provided array to unique and consecutive ones. It utilizes
 * multi-threading to accelerate the insert and search speed. Currently it is
 * only designed to be used in `ToBlockCpu` for optimizing, so it only support
 * key insertions once with Init function, and it does not support key deletion.
 *
 * The hash map should be prepared in two phases before using. With the first
 * being creating the hashmap, and then init it with an id array.
 *
 * For example, for an array A with following entries:
 * [98, 98, 100, 99, 97, 99, 101, 100, 102]
 * Create the hashmap H with:
 * `H = CpuIdHashMap()` (1)
 * And Init it with:
 * `U = H.Init(A)` (2)  (U is an id array used to store the unqiue
 * ids in A).
 * Then U should be (U is not exclusive as the element order is not
 * guaranteed to be steady):
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
 * R = H.Map(I) (3)
 * R should be:
 * [0, 2, 5]
 **/
template <typename IdType>
class IdHashMap {
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

  /**
   * @brief Cross platform CAS operation.
   * It is an atomic operation that compares the contents of a memory
   * location with a given value and, only if they are the same, modifies
   * the contents of that memory location to a new given value.
   *
   * @param ptr The pointer to the object to test and modify .
   * @param old_val The value expected to be found in `ptr`.
   * @param new_val The value to store in `ptr` if it is as expected.
   *
   * @return Old value pointed by the `ptr`.
   */
  static IdType CompareAndSwap(IdType* ptr, IdType old_val, IdType new_val);

  IdHashMap();

  IdHashMap(const IdHashMap& other) = delete;
  IdHashMap& operator=(const IdHashMap& other) = delete;

  /**
   * @brief Init the hashmap with an array of ids.
   * Firstly allocating the memeory and init the entire space with empty key.
   * And then insert the items in `ids` concurrently to generate the
   * mappings, in passing returning the unique ids in `ids`.
   *
   * @param ids The array of ids to be inserted as keys.
   *
   * @return Unique ids for the input `ids`.
   */
  IdArray Init(const IdArray& ids);

  /**
   * @brief Find the mappings of given keys.
   *
   * @param ids The keys to map for.
   *
   * @return Mapping results corresponding to `ids`.
   */
  IdArray MapIds(const IdArray& ids) const;

 private:
  /**
   * @brief Get the next position and delta for probing.
   *
   * @param[in,out] pos Calculate the next position with quadric probing.
   * @param[in,out] delta Calculate the next delta by adding 1.
   */
  void Next(IdType* pos, IdType* delta) const;

  /**
   * @brief Find the mapping of a given key.
   *
   * @param id The key to map for.
   *
   * @return Mapping result for the `id`.
   */
  IdType MapId(const IdType id) const;

  /**
   * @brief Insert an id into the hash map.
   *
   * @param id The id to be inserted.
   * @param valid The item at index will be set to indicate
   * whether the `id` at `index` is inserted or not.
   * @param index The index of the `id`.
   *
   */
  void Insert(IdType id, std::vector<int16_t>* valid, size_t index);

  /**
   * @brief Set the value for the key in the hash map.
   *
   * @param key The key to set for.
   * @param value The value to be set for the `key`.
   *
   * @warning Key must exist.
   */
  void Set(IdType key, IdType value);

  /**
   * @brief Attempt to insert the key into the hash map at the given position.
   * 1. If the key at `pos` is empty -> Set the key, return true and set
   * `valid[index]` to true.
   * 2. If the key at `pos` is equal to `key` -> Return true.
   * 3. If the key at `pos` is non-empty and not equal to `key` -> Return false.
   * @param pos The position in the hash map to be inserted at.
   * @param key The key to be inserted.
   * @param valid The item at index will be set to indicate
   * whether the `key` at `index` is inserted or not.
   * @param index The index of the `key`.
   *
   * @return Whether the key exists in the map now.
   */
  bool AttemptInsertAt(
      int64_t pos, IdType key, std::vector<int16_t>* valid, size_t index);

 private:
  /**
   * @brief Hash maps which is used to store all elements.
   */
  std::unique_ptr<Mapping[], std::function<void(Mapping*)>> hash_map_;

  /**
   * @brief Mask which is assisted to get the position in the table
   * for a key by performing `&` operation with it.
   */
  IdType mask_;
};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ID_HASH_MAP_H_
