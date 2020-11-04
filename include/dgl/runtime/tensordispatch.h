/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/runtime/tensordispatch.h
 * \brief This file defines the dispatcher of tensor operators to framework-specific
 *  implementations.
 *
 *  The dispatcher consists of a TensorDispatcher singleton in DGL C library and
 *  one separately-built shared library per supported backend.
 *
 *  Those shared libraries contain wrappers of the framework-specific operators.
 *  The wrappers have almost the same signatures as functions in aten namespace,
 *  except that they accept and return DLManagedTensors instead of NDArrays.
 *  The wrappers are defined with extern "C", meaning that the C++ compiler will
 *  not do name mangling for those functions so that DGL can conveniently locate
 *  them using dlsym(3) (or GetProcAddress in Windows).
 *
 *  The TensorDispatcher singleton maintains a mapping from an array operator to
 *  the address of the corresponding symbol in the shared library.  During
 *  initialization, the TensorDispatcher checks which backend DGL is using.
 *  It then locates and opens the corresponding shared library using dlopen(3) (or
 *  LoadLibrary in Windows), and populates the said mapping above with dlsym(3)
 *  (or GetProcAddress in Windows).
 *
 *  A tensor operator in TensorDispatcher first checks whether the corresponding symbol
 *  address is found in the mapping.  If so, it calls the function located at the
 *  symbol address instead, translating NDArrays to DLManagedTensors using
 *  NDArray::ToDLPack(), and translates the DLManagedTensors in the return values
 *  back to NDArrays using NDArray::FromDLPack().  If not, it falls back to the
 *  implementation in dgl::aten namespace.
 */

#ifndef DGL_RUNTIME_TENSORDISPATCH_H_
#define DGL_RUNTIME_TENSORDISPATCH_H_

namespace dgl {
namespace runtime {

class TensorDispatcher {
 public:
  static TensorDispatcher* Global() {
    static TensorDispatcher inst;
    return &inst;
  }

  // Allocate a tensor.
  // Calls the framework-specific tensor allocator (e.g. torch::empty) if possible.
  NDArray Empty(std::vector<int64_t> shape, DLDataType dtype, DLContext ctx) const;
  static constexpr int kEmpty = 0;

  // Clones a tensor.
  // Calls the framework-specific copy function (e.g. torch::clone) if possible.
  NDArray Clone(NDArray arr) const;
  static constexpr int kClone = 1;

 private:
  TensorDispatcher();

  const char *names_[] = {
    "TensorDispatcher__empty",
    "TensorDispatcher__clone"
  };
  void *entrypoints_[sizeof(names_) / sizeof(names_[0])];
};

};  // namespace runtime
};  // namespace dgl

#endif  // DGL_RUNTIME_TENSORDISPATCH_H_
