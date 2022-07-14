/*!
 *  Copyright (c) 2021 by Contributors
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

  auto cpu_ctx = DGLContext{kDLCPU, 0};
  auto metapath_ctx = metapath->ctx;
  // use default stream
  cudaStream_t stream = 0;

  TypeArray h_result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, cpu_ctx);
  auto h_result_data = h_result.Ptr<IdxType>();

  auto h_metapath = metapath.CopyTo(cpu_ctx, stream);
  DeviceAPI::Get(metapath_ctx)->StreamSync(metapath_ctx, stream);
  const IdxType *h_metapath_data = h_metapath.Ptr<IdxType>();

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

  auto result = h_result.CopyTo(metapath->ctx, stream);
  DeviceAPI::Get(metapath_ctx)->StreamSync(metapath_ctx, stream);
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
