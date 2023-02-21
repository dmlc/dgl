#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <set>

#include "../../src/array/cpu/concurrent_id_hash_map.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;
using namespace dgl::aten;

namespace {

template <typename IdType>
size_t ConstructRandomSet(
    size_t size, IdType range, std::vector<IdType>& id_vec) {
  id_vec.resize(size);
  std::srand(std::time(nullptr));
  for (size_t i = 0; i < size; i++) {
    id_vec[i] = static_cast<IdType>(std::rand() % range);
  }

  size_t num_seeds = size / 5 + 1;
  std::sort(id_vec.begin(), id_vec.begin() + num_seeds);
  return std::unique(id_vec.begin(), id_vec.begin() + num_seeds) -
         id_vec.begin();
}

template <typename IdType, size_t size, IdType range>
void _TestIdMap() {
  std::vector<IdType> id_vec;
  auto num_seeds = ConstructRandomSet(size, range, id_vec);
  std::set<IdType> id_set(id_vec.begin(), id_vec.end());
  IdArray ids = VecToIdArray(id_vec, sizeof(IdType) * 8, CTX);
  ConcurrentIdHashMap<IdType> id_map;
  IdArray unique_ids = id_map.Init(ids, num_seeds);
  auto unique_num = static_cast<size_t>(unique_ids->shape[0]);
  IdType* unique_id_data = unique_ids.Ptr<IdType>();
  EXPECT_EQ(id_set.size(), unique_num);

  parallel_for(0, num_seeds, 64, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      EXPECT_EQ(id_vec[i], unique_id_data[i]);
    }
  });

  parallel_for(num_seeds, unique_num, 128, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; i++) {
      EXPECT_TRUE(id_set.find(unique_id_data[i]) != id_set.end());
    }
  });

  IdArray new_ids = id_map.MapIds(unique_ids);
  EXPECT_TRUE(new_ids.IsContiguous());
  ids->shape[0] = num_seeds;
  IdArray new_seed_ids = id_map.MapIds(ids);
  EXPECT_TRUE(new_seed_ids.IsContiguous());
  EXPECT_EQ(new_seed_ids.Ptr<IdType>()[0], static_cast<IdType>(0));
}

TEST(ConcurrentIdHashMapTest, TestConcurrentIdHashMap) {
  _TestIdMap<int32_t, 1, 10>();
  _TestIdMap<int64_t, 1, 10>();
  _TestIdMap<int32_t, 1000, 500000>();
  _TestIdMap<int64_t, 1000, 500000>();
  _TestIdMap<int32_t, 50000, 1000000>();
  _TestIdMap<int64_t, 50000, 1000000>();
  _TestIdMap<int32_t, 100000, 40000000>();
  _TestIdMap<int64_t, 100000, 40000000>();
}

};  // namespace
