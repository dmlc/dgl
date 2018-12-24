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

std::vector<IdArray> SearchCachedIds(IdArray cached_ids, const std::vector<IdArray> &queries);

}  // namespace cache

}  // namespace dgl

#endif  // DGL_CACHE_H_
