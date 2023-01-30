#include <set>
#include <unordered_map>
#include <gtest/gtest.h>
#include <dgl/array.h>

#include "../../src/array/cpu/cpu_id_hash_map.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;
using namespace dgl::aten;

namespace {

template <typename IdType>
void ConstructRandomSet(size_t size, size_t range,
  std::vector<IdType>& id_vec) {
    id_vec.resize(size);
    std::srand(42);
    for (size_t i = 0; i < size; i++) {
        id_vec[i] = static_cast<IdType>(std::rand() % range);
    }
}

template <typename IdType>
void _TestIdMap(size_t size, size_t range) {
    std::vector<IdType> id_vec;
    ConstructRandomSet(size, range, id_vec);
    std::set<IdType> id_set(id_vec.begin(), id_vec.end());

    IdArray ids = VecToIdArray(id_vec, sizeof(IdType) * 8, CTX);
    IdArray unique_ids = NewIdArray(size, CTX, sizeof(IdType) * 8);
    CpuIdHashMap<IdType> id_map(CTX);
    auto unique_num = id_map.Init(ids, unique_ids);
    unique_ids->shape[0] = unique_num;
    IdType* unique_id_data = unique_ids.Ptr<IdType>();
    EXPECT_EQ(id_set.size(), unique_num);
    EXPECT_EQ(unique_id_data[unique_num] - 1, unique_num - 1);
    EXPECT_TRUE(unique_ids.IsContiguous());

    IdArray new_ids = NewIdArray(size, CTX, sizeof(IdType) * 8);
    IdType default_val = -1;
    id_map.Map(ids, default_val, new_ids);
    auto  new_id_vec = new_ids.ToVector<IdType>();
    std::unordered_map<IdType, IdType> new_id_map;
    std::vector<bool> isVisited(unique_num);
    for (size_t i = 0; i < size; i++) {
        EXPECT_NE(default_val, new_id_vec[i]);
        auto old_id = id_vec[i];
        // Make sure same old ids map to same new ids.
        if (new_id_map.find(old_id) == new_id_map.end()) {
            new_id_map[old_id] = new_id_vec[i];
        } else {
            EXPECT_EQ(new_id_map[old_id], new_id_vec[i]);
            isVisited[new_id_vec[i]] = 1;
        }
    }

    // // All new ids should be mapped.
    // EXPECT_TRUE(std::all_of(isVisited.begin(),
    //     isVisited.end(), [](bool flag){ return flag;}));
}

TEST(CpuIdHashMapTest, TestCpuIdHashMap) {
    _TestIdMap<int64_t>(100, 1000);
}

}; // namespace