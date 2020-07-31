/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/array_sort.cu
 * \brief Array sort GPU implementation
 */
#include <dgl/array.h>
#include <cub/cub.cuh>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> Sort(IdArray array) {
  const auto& ctx = array->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t nitems = array->shape[0];
  IdArray orig_idx = Range(0, nitems, 64, ctx);
  IdArray sorted_array = NewIdArray(nitems, ctx, array->dtype.bits);
  IdArray sorted_idx = NewIdArray(nitems, ctx, 64);

  const IdType* keys_in = array.Ptr<IdType>();
  const int64_t* values_in = orig_idx.Ptr<int64_t>();
  IdType* keys_out = sorted_array.Ptr<IdType>();
  int64_t* values_out = sorted_idx.Ptr<int64_t>();

  // Allocate workspace
  size_t workspace_size = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, workspace_size,
      keys_in, keys_out, values_in, values_out, nitems);
  void* workspace = device->AllocWorkspace(ctx, workspace_size);

  // Compute
  cub::DeviceRadixSort::SortPairs(workspace, workspace_size,
      keys_in, keys_out, values_in, values_out, nitems);

  device->FreeWorkspace(ctx, workspace);

  return std::make_pair(sorted_array, sorted_idx);
}

template std::pair<IdArray, IdArray> Sort<kDLGPU, int32_t>(IdArray);
template std::pair<IdArray, IdArray> Sort<kDLGPU, int64_t>(IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
