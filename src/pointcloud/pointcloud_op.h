/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array_op.h
 * \brief Array operator templates
 */
#ifndef DGL_FPS_FPS_OP_H_
#define DGL_FPS_FPS_OP_H_

#include <dgl/array.h>

namespace dgl {
namespace aten {
namespace impl {

template <typename DType>
IdArray _FPS_CPU(NDArray array, int64_t batch_size, int64_t sample_points, DLContext ctx);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_FPS_FPS_OP_H_
