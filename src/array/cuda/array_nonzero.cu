/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/array_nonzero.cc
 * \brief Array nonzero CPU implementation
 */
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename IdType>
struct IsNonZero {
  __host__ __device__ bool operator() (const IdType val) {
    return val != 0;
  }
};

template <DLDeviceType XPU, typename IdType>
IdArray NonZero(IdArray array) {
  const int64_t len = array->shape[0];
  IdArray ret = NewIdArray(len, array->ctx, array->dtype.bits);
  thrust::device_ptr<IdType> in_data(array.Ptr<IdType>());
  thrust::device_ptr<IdType> out_data(ret.Ptr<IdType>());
  // TODO(minjie): should take control of the memory allocator.
  //   See PyTorch's implementation here: 
  //   https://github.com/pytorch/pytorch/blob/1f7557d173c8e9066ed9542ada8f4a09314a7e17/
  //     aten/src/THC/generic/THCTensorMath.cu#L104
  auto indices_end = thrust::copy_if(thrust::make_counting_iterator<IdType>(0),
                                     thrust::make_counting_iterator<IdType>(len),
                                     in_data,
                                     out_data,
                                     IsNonZero<IdType>());
  const int64_t num_nonzeros = indices_end - out_data;
  return ret.CreateView({num_nonzeros}, array->dtype, 0);
}

template IdArray NonZero<kDLGPU, int32_t>(IdArray);
template IdArray NonZero<kDLGPU, int64_t>(IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
