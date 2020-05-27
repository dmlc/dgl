/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/utils.cc
 * \brief Kernel utilities
 */
#include <dgl/array.h>
#include <vector>
#include <string>

#include "./utils.h"
#include "./binary_reduce_common.h"

namespace dgl {
namespace kernel {
namespace utils {

int FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

int64_t ComputeXLength(runtime::NDArray feat_array) {
  int64_t ret = 1;
  for (int i = 1; i < feat_array->ndim; ++i) {
    ret *= feat_array->shape[i];
  }
  return ret;
}

int64_t NElements(const runtime::NDArray& array) {
  if (aten::IsNullArray(array)) {
    return 0;
  } else {
    int64_t ret = 1;
    for (int i = 0; i < array->ndim; ++i) {
      ret *= array->shape[i];
    }
    return ret;
  }
}

int64_t Prod(const std::vector<int64_t>& vec) {
  int64_t ret = 1;
  for (int64_t v : vec) {
    ret *= v;
  }
  return ret;
}

template <typename Idx>
inline void ComputeEdgeMapping(Idx **cur_mapping, runtime::NDArray cur, runtime::NDArray eids) {
  if (*cur_mapping == nullptr)
    *cur_mapping = static_cast<Idx*>(eids->data);
  else {
    runtime::NDArray out_map = aten::MergeIDMapping(eids, cur);
    *cur_mapping = static_cast<Idx*>(out_map->data);
  }
}


}  // namespace utils
}  // namespace kernel
}  // namespace dgl
