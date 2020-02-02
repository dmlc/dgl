/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/array_utils.h
 * \brief Utility classes and functions for DGL arrays.
 */
#ifndef DGL_ARRAY_CPU_ARRAY_UTILS_H_
#define DGL_ARRAY_CPU_ARRAY_UTILS_H_

#include <dgl/array.h>
#include <vector>
#include <unordered_map>
#include <utility>

namespace dgl {

namespace aten {

/*!
 * \brief A hashmap that maps each ids in the given array to new ids starting from zero.
 */
template <typename IdType>
class IdHashMap {
 public:
  // default ctor
  IdHashMap(): filter_(kFilterSize, false) {}

  // Construct the hashmap using the given id array.
  // The id array could contain duplicates.
  explicit IdHashMap(IdArray ids): filter_(kFilterSize, false) {
    Update(ids);
  }

  // Update the hashmap with given id array.
  // The id array could contain duplicates.
  void Update(IdArray ids) {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    IdType newid = oldv2newv_.size();
    for (int64_t i = 0; i < len; ++i) {
      const IdType id = ids_data[i];
      if (!Contains(id)) {
        oldv2newv_[id] = newid++;
        filter_[id & kFilterMask] = true;
      }
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(IdType id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType Map(IdType id, IdType default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

  // Return the new id of each id in the given array.
  IdArray Map(IdArray ids, IdType default_val) const {
    const int64_t len = ids->shape[0];
    std::vector<IdType> values(len);
    for (int64_t i = 0; i < len; ++i)
      values[i] = Map(IndexSelect<IdType>(ids, i), default_val);
    return NDArray::FromVector(values, ids->dtype, ids->ctx);
  }

  // Return all the old ids collected so far, ordered by new id.
  std::vector<IdType> Values() const {
    std::vector<IdType> values(oldv2newv_.size());
    for (auto pair : oldv2newv_)
      values[pair.second] = pair.first;
    return values;
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  std::unordered_map<IdType, IdType> oldv2newv_;
};

struct PairHash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

};  // namespace aten

};  // namespace dgl

#endif  // DGL_ARRAY_CPU_ARRAY_UTILS_H_
