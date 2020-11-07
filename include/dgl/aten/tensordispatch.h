/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/tensordispatch.h
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

#ifndef DGL_ATEN_TENSORDISPATCH_H_
#define DGL_ATEN_TENSORDISPATCH_H_

#include <dlpack/dlpack.h>
#include <tensoradapter.h>
#include <vector>
#include "./types.h"

namespace dgl {
namespace aten {

class TensorDispatcher {
 public:
  static TensorDispatcher* Global() {
    static TensorDispatcher inst;
    return &inst;
  }

  // Allocate a tensor.
  // Calls the framework-specific tensor allocator (e.g. torch::empty) if possible.
  inline NDArray Empty(std::vector<int64_t> shape, DLDataType dtype, DLContext ctx) const {
    auto entry = entrypoints_[Op::kEmpty];

    /*
    if (!entrypoints_[Op::kEmpty]) {
      return NDArray::Empty(shape, dtype, ctx);
    } else {
      auto result = TA_DISPATCH(tensoradapter::TAempty, entry, shape, dtype, ctx);
      return NDArray::FromDLPack(result);
    }
    */
    CHECK(entrypoints_[Op::kEmpty]) << "torch allocator not found";
    auto result = TA_DISPATCH(tensoradapter::TAempty, entry, shape, dtype, ctx);
    return NDArray::FromDLPack(result);
  }

  inline NDArray Clone(NDArray array) const {
    auto entry = entrypoints_[Op::kClone];

    if (!entrypoints_[Op::kClone]) {
      return array.Clone();
    } else {
      return NDArray::FromDLPack(TA_DISPATCH(tensoradapter::TAclone, entry, array.ToDLPack()));
    }
  }

  inline NDArray CopyTo(NDArray array, DLContext ctx) const {
    auto entry = entrypoints_[Op::kCopyTo];

    if (!entrypoints_[Op::kCopyTo]) {
      return array.CopyTo(ctx);
    } else {
      auto tensor = array.ToDLPack();
      return NDArray::FromDLPack(TA_DISPATCH(tensoradapter::TAcopyto, entry, tensor, ctx));
    }
  }

 private:
  TensorDispatcher();

  static constexpr const char *names_[] = {
    "TAempty",
    "TAclone",
    "TAcopyto"
  };

  class Op {
   public:
    static constexpr int kEmpty = 0;
    static constexpr int kClone = 1;
    static constexpr int kCopyTo = 2;
  };

  static constexpr int num_entries_ = sizeof(names_) / sizeof(names_[0]);
  void *entrypoints_[num_entries_] = {nullptr};
};

};  // namespace aten
};  // namespace dgl

#endif  // DGL_RUNTIME_TENSORDISPATCH_H_
