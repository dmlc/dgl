/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/array_concat.cc
 * \brief Array concat CPU implementation
 */
#include <dgl/array.h>
#include "../../cuda_utils.h"
#include "../../runtime/cuda/cuda_common.h"

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
  char* data = static_cast<char*>(ret->data);
  for (size_t i = 0; i < arrays.size(); ++i) {
    const char* array_data = static_cast<const char*>(arrays[i]->data);
    char* to = data + offset * sizeof(DType);
    size_t data_len = arrays[i]->shape[0] * sizeof(DType);
    CUDA_CALL(cudaMemcpy(to,
                         array_data,
                         data_len,
                         cudaMemcpyDeviceToDevice));
    offset += arrays[i]->shape[0];
  }

  return ret;
}

template NDArray Concat<kDLGPU, int32_t>(const std::vector<IdArray>&);
template NDArray Concat<kDLGPU, int64_t>(const std::vector<IdArray>&);
template NDArray Concat<kDLGPU, float>(const std::vector<IdArray>&);
template NDArray Concat<kDLGPU, double>(const std::vector<IdArray>&);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
