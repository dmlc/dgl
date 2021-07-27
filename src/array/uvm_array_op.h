/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array_op.h
 * \brief Array operator templates
 */
#ifndef DGL_ARRAY_UVM_ARRAY_OP_H_
#define DGL_ARRAY_UVM_ARRAY_OP_H_

#include <dgl/array.h>
#include <utility>

namespace dgl {
namespace aten {
namespace impl {

// Take CPU array and GPU index, and then index with GPU.
template <typename DType, typename IdType>
NDArray IndexSelectCPUFromGPU(NDArray array, IdArray index);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_UVM_ARRAY_OP_H_
