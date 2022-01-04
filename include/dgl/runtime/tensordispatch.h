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

#ifndef DGL_RUNTIME_TENSORDISPATCH_H_
#define DGL_RUNTIME_TENSORDISPATCH_H_

#include <dlpack/dlpack.h>
#include <tensoradapter.h>
#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#endif  // WIN32
#include <vector>
#include "ndarray.h"

/*! \brief Casts a pointer \c entry to a function pointer with signature of \c func */
#define FUNCCAST(func, entry)   (*reinterpret_cast<decltype(&(func))>(entry))

namespace dgl {
namespace runtime {

/*!
 * \brief Dispatcher that delegates the function calls to framework-specific C++ APIs.
 *
 * This class is not thread-safe.
 */
class TensorDispatcher {
 public:
  /*! \brief Get the singleton instance. */
  static TensorDispatcher* Global() {
    static TensorDispatcher inst;
    return &inst;
  }

  /*! \brief Whether an adapter library is available */
  inline bool IsAvailable() {
    return available_;
  }

  /*! \brief Load symbols from the given tensor adapter library path */
  bool Load(const char *path_cstr);

  /*!
   * \brief Allocate an empty tensor.
   *
   * Used in NDArray::Empty().
   */
  inline NDArray Empty(std::vector<int64_t> shape, DLDataType dtype, DLContext ctx) const {
    auto entry = entrypoints_[Op::kEmpty];
    auto result = FUNCCAST(tensoradapter::TAempty, entry)(shape, dtype, ctx);
    return NDArray::FromDLPack(result);
  }

 private:
  /*! \brief ctor */
  TensorDispatcher() = default;
  /*! \brief dtor */
  ~TensorDispatcher();

  /*!
   * \brief List of symbols in the adapter library.
   *
   * Must match the functions in tensoradapter/include/tensoradapter.h.
   */
  static constexpr const char *names_[] = {
    "TAempty",
  };

  /*! \brief Index of each function to the symbol list */
  class Op {
   public:
    static constexpr int kEmpty = 0;
  };

  /*! \brief Number of functions */
  static constexpr int num_entries_ = sizeof(names_) / sizeof(names_[0]);

  /*! \brief Entrypoints of each function */
  void* entrypoints_[num_entries_] = {nullptr};

  bool available_ = false;
#if defined(WIN32) || defined(_WIN32)
  HINSTANCE handle_;
#else   // !WIN32
  void* handle_;
#endif  // WIN32
};

};  // namespace runtime
};  // namespace dgl

#undef FUNCCAST

#endif  // DGL_RUNTIME_TENSORDISPATCH_H_
