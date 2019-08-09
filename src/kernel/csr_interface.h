/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/csr_interface.h
 * \brief Kernel common utilities
 */

#ifndef DGL_KERNEL_CSR_INTERFACE_H_
#define DGL_KERNEL_CSR_INTERFACE_H_

#include <dgl/array.h>
#include <dgl/runtime/c_runtime_api.h>

namespace dgl {
namespace kernel {

class CSRWrapper {
 public:
  virtual aten::CSRMatrix GetInCSRMatrix() const = 0;
  virtual aten::CSRMatrix GetOutCSRMatrix() const = 0;
  virtual DGLContext Context() const = 0;
  virtual int NumBits() const = 0;
};

};  // namespace kernel
};  // namespace dgl

#endif  // DGL_KERNEL_CSR_INTERFACE_H_
