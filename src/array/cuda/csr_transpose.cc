/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/csr_transpose.cc
 * @brief CSR transpose (convert to CSC)
 */
#include <dgl/array.h>

#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr) {
  LOG(FATAL) << "Unreachable codes";
  return {};
}

template <>
CSRMatrix CSRTranspose<kDGLCUDA, int32_t>(CSRMatrix csr) {
  return COOToCSR(COOTranspose(CSRToCOO(csr, false)));
}

template <>
CSRMatrix CSRTranspose<kDGLCUDA, int64_t>(CSRMatrix csr) {
  return COOToCSR(COOTranspose(CSRToCOO(csr, false)));
}

template CSRMatrix CSRTranspose<kDGLCUDA, int32_t>(CSRMatrix csr);
template CSRMatrix CSRTranspose<kDGLCUDA, int64_t>(CSRMatrix csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
