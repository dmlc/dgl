/**
 *  Copyright (c) 2023 by Contributors
 * @file concurrent_id_hash_map.h
 * @brief Class about concurrent id hash map.
 */

#ifndef GRAPHBOLT_CONCURRENT_ID_HASH_MAP_H_
#define GRAPHBOLT_CONCURRENT_ID_HASH_MAP_H_

#include <torch/torch.h>

#include <functional>
#include <memory>
#include <vector>

namespace graphbolt {
namespace sampling {

/**
 * @brief A CPU targeted hashmap for mapping duplicate and non-consecutive ids
 * in the provided array to unique and consecutive ones. It utilizes
 * multi-threading to accelerate the insert and search speed. Currently it is
 * only designed to be used in `ToBlockCpu` for optimizing, so it only support
 * key insertions once with Init function, and it does not support key deletion.
 *
 * The hash map should be prepared in two phases before using. With the first
 * being creating the hashmap, and then initialize it with an id array which is
 * divided into 2 parts: [`seed ids`, `sampled ids`]. `Seed ids` refer to
 * a set ids chosen as the input for sampling process and `sampled ids` are the
 * ids new sampled from the process (note the the `seed ids` might also be
 * sampled in the process and included in the `sampled ids`). In result `seed
 * ids` are mapped to [0, num_seed_ids) and `sampled ids` to [num_seed_ids,
 * num_unique_ids). Notice that mapping order is stable for `seed ids` while not
 * for the `sampled ids`.
 *
 * For example, for an array `A` having 4 seed ids with following entries:
 * [99, 98, 100, 97, 97, 101, 101, 102, 101]
 * Create the hashmap `H` with:
 * `H = ConcurrentIdHashMap()` (1)
 * And Init it with:
 * `U = H.Init(A)` (2)  (U is an id array used to store the unqiue
 * ids in A).
 * Then `U` should be (U is not exclusive as the overall mapping is not stable):
 * [99, 98, 100, 97, 102, 101]
 * And the hashmap should generate following mappings:
 *  * [
 *   {key: 99, value: 0},
 *   {key: 98, value: 1},
 *   {key: 100, value: 2},
 *   {key: 97, value: 3},
 *   {key: 102, value: 4},
 *   {key: 101, value: 5}
 * ]
 * Search the hashmap with array `I`=[98, 99, 102]:
 * R = H.Map(I) (3)
 * R should be:
 * [1, 0, 4]
 **/
template <typename IdType>
class ConcurrentIdHashMap {
 private:
  /**
   * @brief The result state of an attempt to insert.
   */
  enum class InsertState {
    OCCUPIED,  // Indicates that the space where an insertion is being
               // attempted is already occupied by another element.
    EXISTED,  // Indicates that the element being inserted already exists in the
              // map, and thus no insertion is performed.
    INSERTED  // Indicates that the insertion was successful and a new element
              // was added to the map.
  };

 public:
  /**
   * @brief Initialize the hashmap with an array of ids. The first `num_seeds`
   * ids are unique and must be mapped to a contiguous array starting
   * from 0. The left can be duplicated and the mapping result is not stable.
   * The unique'ified ids can be accessed through calling `GetUniqueIds()`;
   *
   * @param ids The array of the ids to be inserted.
   * @param num_seeds The number of seed ids.
   */
  ConcurrentIdHashMap(const torch::Tensor& ids, size_t num_seeds);

  ConcurrentIdHashMap(const ConcurrentIdHashMap& other) = delete;
  ConcurrentIdHashMap& operator=(const ConcurrentIdHashMap& other) = delete;

  /**
   * @brief Get the unique ids for the keys given in the constructor.
   */
  const torch::Tensor& GetUniqueIds() const { return unique_ids_; }

  /**
   * @brief Find mappings of given keys.
   *
   * @param ids The keys to map for.
   *
   * @return Mapping results corresponding to `ids`.
   */
  torch::Tensor MapIds(const torch::Tensor& ids) const;

 private:
  /**
   * @brief Get the next position and delta for probing.
   *
   * @param[in,out] pos Calculate the next position with quadric probing.
   * @param[in,out] delta Calculate the next delta by adding 1.
   */
  inline void Next(IdType* pos, IdType* delta) const;

  /**
   * @brief Find the mapping of a given key.
   *
   * @param id The key to map for.
   *
   * @return Mapping result corresponding to `id`.
   */
  inline IdType MapId(const IdType id) const;

  /**
   * @brief Insert an id into the hash map.
   *
   * @param id The id to be inserted.
   *
   * @return Whether the `id` is inserted or not.
   */
  inline bool Insert(IdType id);

  /**
   * @brief Set the value for the key in the hash map.
   *
   * @param key The key to set for.
   * @param value The value to be set for the `key`.
   *
   * @warning Key must exist.
   */
  inline void Set(IdType key, IdType value);

  /**
   * @brief Insert a key into the hash map.
   *
   * @param id The key to be inserted.
   * @param value The value to be set for the `key`.
   *
   */
  inline void InsertAndSet(IdType key, IdType value);

  /**
   * @brief Insert a key into the hash map. If the key exists, set the value
   * with the smaller value.
   *
   * @param id The key to be inserted.
   * @param value The value to be set for the `key`.
   *
   */
  inline void InsertAndSetMin(IdType id, IdType value);

  /**
   * @brief Attempt to insert the key into the hash map at the given position.
   *
   * @param pos The position in the hash map to be inserted at.
   * @param key The key to be inserted.
   *
   * @return The state of the insertion.
   */
  inline InsertState AttemptInsertAt(int64_t pos, IdType key);

 private:
  /**
   * @brief Hash maps which is used to store all elements.
   */
  torch::Tensor hash_map_;

  /**
   * @brief Holds the ids that are made unique in the constructor.
   */
  torch::Tensor unique_ids_;

  /**
   * @brief Mask which is assisted to get the position in the table
   * for a key by performing `&` operation with it.
   */
  IdType mask_;
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_CONCURRENT_ID_HASH_MAP_H_
