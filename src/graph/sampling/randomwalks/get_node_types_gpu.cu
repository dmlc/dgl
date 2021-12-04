/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/get_node_types_gpu.cu
 * \brief DGL sampler
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/device_api.h>
#include <cuda_runtime.h>
#include <utility>
#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template<DLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg,
    const TypeArray metapath) {

  uint64_t num_etypes = metapath->shape[0];
  TypeArray result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, metapath->ctx);
  IdxType *result_data = static_cast<IdxType *>(result->data);

  auto metapath_ctx = metapath->ctx;
  auto cpu_ctx = DGLContext{kDLCPU, 0};
  auto metapath_device = DeviceAPI::Get(metapath_ctx);
  auto cpu_device = DeviceAPI::Get(cpu_ctx);
  // use default stream
  cudaStream_t stream = 0;

  auto h_metapath_data = static_cast<IdxType*>(
      cpu_device->AllocWorkspace(cpu_ctx, sizeof(IdxType) * (num_etypes)));
  auto h_result_data = static_cast<IdxType*>(
      cpu_device->AllocWorkspace(cpu_ctx, sizeof(IdxType) * (num_etypes + 1)));

  metapath_device->CopyDataFromTo(static_cast<const IdxType*>(metapath->data), 0,
                                  h_metapath_data, 0,
                                  sizeof(IdxType) * (num_etypes),
                                  metapath_ctx, cpu_ctx,
                                  metapath->dtype, stream);

  dgl_type_t curr_type = hg->GetEndpointTypes(h_metapath_data[0]).first;
  h_result_data[0] = curr_type;

  for (uint64_t i = 0; i < num_etypes; ++i) {
    auto src_dst_type = hg->GetEndpointTypes(h_metapath_data[i]);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "source of edge type #" << i <<
        " does not match destination of edge type #" << i - 1;
    }
    curr_type = dsttype;
    h_result_data[i + 1] = dsttype;
  }
  metapath_device->CopyDataFromTo(h_result_data, 0, result_data, 0,
                                  sizeof(IdxType) * (num_etypes + 1),
                                  cpu_ctx, metapath_ctx,
                                  metapath->dtype, stream);
  // release the data
  metapath_device->StreamSync(metapath_ctx, stream);
  cpu_device->FreeWorkspace(cpu_ctx, h_metapath_data);
  cpu_device->FreeWorkspace(cpu_ctx, h_result_data);
  return result;
}

template
TypeArray GetNodeTypesFromMetapath<kDLGPU, int32_t>(
    const HeteroGraphPtr hg,
    const TypeArray metapath);
template
TypeArray GetNodeTypesFromMetapath<kDLGPU, int64_t>(
    const HeteroGraphPtr hg,
    const TypeArray metapath);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
