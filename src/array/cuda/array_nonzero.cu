/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/array_nonzero.cc
 * \brief Array nonzero CPU implementation
 */
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename IdType>
struct IsNonZero {
  __device__ bool operator() (const IdType val) {
    return val != 0;
  }
};

template <DLDeviceType XPU, typename IdType>
IdArray NonZero(IdArray array) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t len = array->shape[0];
  IdArray ret = NewIdArray(len, array->ctx, 64);
  thrust::device_ptr<IdType> in_data(array.Ptr<IdType>());
  thrust::device_ptr<int64_t> out_data(ret.Ptr<int64_t>());
  // TODO(minjie): should take control of the memory allocator.
  //   See PyTorch's implementation here:
  //   https://github.com/pytorch/pytorch/blob/1f7557d173c8e9066ed9542ada8f4a09314a7e17/
  //     aten/src/THC/generic/THCTensorMath.cu#L104
  auto startiter = thrust::make_counting_iterator<int64_t>(0);
  auto enditer = startiter + len;
  auto indices_end = thrust::copy_if(thrust::cuda::par.on(thr_entry->stream),
                                     startiter,
                                     enditer,
                                     in_data,
                                     out_data,
                                     IsNonZero<IdType>());
  const int64_t num_nonzeros = indices_end - out_data;
  return ret.CreateView({num_nonzeros}, ret->dtype, 0);
}

template IdArray NonZero<kDLGPU, int32_t>(IdArray);
template IdArray NonZero<kDLGPU, int64_t>(IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
