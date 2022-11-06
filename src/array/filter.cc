/**
 *  Copyright (c) 2021 by Contributors
 * @file array/filter.cc
 * @brief Object for selecting items in a set, or selecting items not in a set.
 */

#include "./filter.h"

#include <dgl/packed_func_ext.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

namespace dgl {
namespace array {

using namespace dgl::runtime;

template <DGLDeviceType XPU, typename IdType>
FilterRef CreateSetFilter(IdArray set);

DGL_REGISTER_GLOBAL("utils.filter._CAPI_DGLFilterCreateFromSet")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      IdArray array = args[0];
      auto ctx = array->ctx;
      // TODO(nv-dlasalle): Implement CPU version.
      if (ctx.device_type == kDGLCUDA) {
#ifdef DGL_USE_CUDA
        ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
          *rv = CreateSetFilter<kDGLCUDA, IdType>(array);
        });
#else
        LOG(FATAL) << "GPU support not compiled.";
#endif
      } else {
        LOG(FATAL) << "CPU support not yet implemented.";
      }
    });

DGL_REGISTER_GLOBAL("utils.filter._CAPI_DGLFilterFindIncludedIndices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      FilterRef filter = args[0];
      IdArray array = args[1];
      *rv = filter->find_included_indices(array);
    });

DGL_REGISTER_GLOBAL("utils.filter._CAPI_DGLFilterFindExcludedIndices")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      FilterRef filter = args[0];
      IdArray array = args[1];
      *rv = filter->find_excluded_indices(array);
    });

}  // namespace array
}  // namespace dgl
