/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/cache.h
 * \brief Operations on cache.
 */
#ifndef DGL_CACHE_H_
#define DGL_CACHE_H_

#include <vector>
#include "runtime/ndarray.h"

namespace dgl {

typedef dgl::runtime::NDArray IdArray;

namespace cache {

/*!
 * \brief search for Ids in the cached Ids.
 *
 * Ids in `cached_ids` must have been sorted. Ids in `queries` may not be sorted.
 *
 * \param cached_ids The Ids in the cache.
 * \param queries The Ids to check if they are in the cache.
 * \return the search results. For each queries vector, the function returns four arrays:
 *         cached_query_idx: The index of the queries whose data is in the cache.
 *         uncached_query_idx: The index of the queries whose data isn't in the cache.
 *         cache_idx: the index in the cache, if a queried data is in the cache.
 *         global_uncached_ids: the original Ids, if a queries data isn't in the cache.
 */
std::vector<IdArray> SearchCachedIds(IdArray cached_ids, const std::vector<IdArray> &queries);

}  // namespace cache

}  // namespace dgl

#endif  // DGL_CACHE_H_
