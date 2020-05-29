/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include "../c_api_common.h"
#include "./fps_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

IdArray FPS(NDArray array, IdArray batch_ptr, int64_t npoints) {
  IdArray ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(batch_ptr->dtype, IdType, {
        ret = impl::FPS<XPU, DType, IdType>(array, batch_ptr, npoints, array->ctx);
      });
    });
  });
  return ret;
}

DGL_REGISTER_GLOBAL("fps._CAPI_FarthestPointSampler")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const NDArray data = args[0];
    const IdArray batch_ptr = args[1];
    const int64_t npoints = args[2];
    const auto result = FPS(data, batch_ptr, npoints);
    *rv = result;
  });

}  // namespace aten
}  // namespace dgl
