/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array_op.h
 * \brief Array operator templates
 */
#ifndef DGL_FPS_FPS_OP_H_
#define DGL_FPS_FPS_OP_H_

#include <dgl/array.h>
#include <vector>
#include <tuple>
#include <utility>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename DType, typename IdType>
IdArray FPS(NDArray array, IdArray batch_ptr, int64_t npoints, DLContext ctx);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_FPS_FPS_OP_H_
