/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/aten/types.h
 * @brief Array and ID types
 */
#ifndef DGL_ATEN_TYPES_H_
#define DGL_ATEN_TYPES_H_

#include <cstdint>

#include "../runtime/ndarray.h"

namespace dgl {

typedef uint64_t dgl_id_t;
typedef uint64_t dgl_type_t;
/** @brief Type for dgl fomrat code, whose binary representation indices
 * which sparse format is in use and which is not.
 *
 * Suppose the binary representation is xyz, then
 * - x indicates whether csc is in use (1 for true and 0 for false).
 * - y indicates whether csr is in use.
 * - z indicates whether coo is in use.
 */
typedef uint8_t dgl_format_code_t;

using dgl::runtime::NDArray;

typedef NDArray IdArray;
typedef NDArray DegreeArray;
typedef NDArray BoolArray;
typedef NDArray IntArray;
typedef NDArray FloatArray;
typedef NDArray TypeArray;

namespace aten {

static const DGLContext CPU{kDGLCPU, 0};

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ATEN_TYPES_H_
