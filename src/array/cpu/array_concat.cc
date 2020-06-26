/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/array_concat.cc
 * \brief Array concat CPU implementation
 */
#include <dgl/array.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename DType>
NDArray Concat(const std::vector<IdArray>& arrays) {
  CHECK(arrays.size() > 1) << "Number of arrays should larger than 1";
  int64_t len = 0, offset = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    len += arrays[i]->shape[0];
  }

  NDArray ret = NDArray::Empty({len}, 
                               arrays[0]->dtype,
                               arrays[0]->ctx);
  DType* data = static_cast<DType*>(ret->data);
  for (size_t i = 0; i < arrays.size(); ++i) {
    const DType* array_data = static_cast<const DType*>(arrays[i]->data);
    std::copy(array_data,
              array_data + arrays[i]->shape[0],
              data + offset);
    offset += arrays[i]->shape[0];
  }

  return ret;
}

template NDArray Concat<kDLCPU, int32_t>(const std::vector<IdArray>&);
template NDArray Concat<kDLCPU, int64_t>(const std::vector<IdArray>&);
template NDArray Concat<kDLCPU, float>(const std::vector<IdArray>&);
template NDArray Concat<kDLCPU, double>(const std::vector<IdArray>&);

};  // namespace impl
};  // namespace aten
};  // namespace dgl