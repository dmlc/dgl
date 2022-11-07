/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/sampling/get_node_types_cpu.cc
 * @brief DGL sampler - CPU implementation of random walks with OpenMP
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <utility>

#include "randomwalks_impl.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template <DGLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg, const TypeArray metapath) {
  uint64_t num_etypes = metapath->shape[0];
  TypeArray result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, metapath->ctx);

  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);
  IdxType *result_data = static_cast<IdxType *>(result->data);

  dgl_type_t curr_type = hg->GetEndpointTypes(metapath_data[0]).first;
  result_data[0] = curr_type;

  for (uint64_t i = 0; i < num_etypes; ++i) {
    auto src_dst_type = hg->GetEndpointTypes(metapath_data[i]);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "source of edge type #" << i
                 << " does not match destination of edge type #" << i - 1;
      return result;
    }
    curr_type = dsttype;
    result_data[i + 1] = dsttype;
  }
  return result;
}

template TypeArray GetNodeTypesFromMetapath<kDGLCPU, int32_t>(
    const HeteroGraphPtr hg, const TypeArray metapath);
template TypeArray GetNodeTypesFromMetapath<kDGLCPU, int64_t>(
    const HeteroGraphPtr hg, const TypeArray metapath);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
