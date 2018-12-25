/*!
 *  Copyright (c) 2018 by Contributors
 * \file cache/cache.cc
 * \brief DGL cache implementation
 */

#include <vector>
#include <dgl/graph.h>
#include <dgl/cache.h>
#include "../c_api_common.h"

namespace dgl {
namespace cache {

dgl_id_t find_id(const dgl_id_t *id_start, const dgl_id_t *id_end, dgl_id_t id) {
  const auto it = std::lower_bound(id_start, id_end, id);
  // If the vertex Id doesn't exist, the vid in the subgraph is -1.
  if (it != id_end && *it == id) {
    return it - id_start;
  } else {
    return -1;
  }
}

std::vector<IdArray> SearchCachedIds(IdArray cached_ids, const std::vector<IdArray> &queries) {
  CHECK(IsValidIdArray(cached_ids)) << "Invalid cached id array.";
  const auto cached_len = cached_ids->shape[0];
  const dgl_id_t* cached_data = static_cast<dgl_id_t*>(cached_ids->data);

  // We assume Ids in the two input arrays are sorted in the ascending order.

  std::vector<IdArray> rets(queries.size() * 4);
#pragma omp parallel for
  for (size_t i = 0; i < queries.size(); i++) {
    IdArray query = queries[i];
    CHECK(IsValidIdArray(query)) << "Invalid query id array.";
    const auto query_len = query->shape[0];
    const dgl_id_t* query_data = static_cast<dgl_id_t*>(query->data);

    // The index of the queries whose data is in the cache.
    std::vector<dgl_id_t> cached_query_idx;
    // The index of the queries whose data isn't in the cache.
    std::vector<dgl_id_t> uncached_query_idx;
    // If a queried data is in the cache, the index in the cache.
    std::vector<dgl_id_t> cache_idx;
    // If a queries data isn't in the cache, its original id.
    std::vector<dgl_id_t> global_uncached_ids;
    for (int64_t j = 0; j < query_len; j++) {
      dgl_id_t id = find_id(cached_data, cached_data + cached_len, query_data[j]);
      if (id == -1) {
        global_uncached_ids.push_back(query_data[j]);
        uncached_query_idx.push_back(j);
      } else {
        cache_idx.push_back(id);
        cached_query_idx.push_back(j);
      }
    }

    IdArray arr = IdArray::Empty({static_cast<int64_t>(cached_query_idx.size())},
                                 DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    std::copy(cached_query_idx.begin(), cached_query_idx.end(),
              static_cast<dgl_id_t*>(arr->data));
    rets[i * 4] = arr;

    arr = IdArray::Empty({static_cast<int64_t>(uncached_query_idx.size())},
                         DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    std::copy(uncached_query_idx.begin(), uncached_query_idx.end(),
              static_cast<dgl_id_t*>(arr->data));
    rets[i * 4 + 1] = arr;

    arr = IdArray::Empty({static_cast<int64_t>(cache_idx.size())},
                         DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    std::copy(cache_idx.begin(), cache_idx.end(), static_cast<dgl_id_t*>(arr->data));
    rets[i * 4 + 2] = arr;

    arr = IdArray::Empty({static_cast<int64_t>(global_uncached_ids.size())},
                         DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    std::copy(global_uncached_ids.begin(), global_uncached_ids.end(),
              static_cast<dgl_id_t*>(arr->data));
    rets[i * 4 + 3] = arr;
  }
  return rets;
}

}  // namespace cache

}  // namespace dgl
