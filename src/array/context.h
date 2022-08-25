/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/Context.h
 * \brief Context
 */


#ifndef DGL_ARRAY_CONTEXT_H_
#define DGL_ARRAY_CONTEXT_H_

namespace dgl {
namespace aten {

#ifdef _WIN32
#define DGL_ATEN_CONTEXT_HIDDEN
#define DGL_ATEN_CONTEXT_EXPORT __declspec(dllexport)
#else  // _WIN32
#if defined(__GNUC__)
#define DGL_ATEN_CONTEXT_EXPORT __attribute__((__visibility__("default")))
#define DGL_ATEN_CONTEXT_HIDDEN __attribute__((__visibility__("hidden")))
#else  // defined(__GNUC__)
#define DGL_ATEN_CONTEXT_EXPORT
#define DGL_ATEN_CONTEXT_HIDDEN
#endif  // defined(__GNUC__)
#endif  // _WIN32

#define INIT_CONTEXT() static Context& context = init()

class DGL_ATEN_CONTEXT_EXPORT Context {
 public:
  Context();

  // Enabling or disable use libxsmm for Spmm
  void setLibxsmm(bool);
  bool libxsmm() const;

 private:
  bool _libxsmm = true;
};

DGL_ATEN_CONTEXT_EXPORT Context& globalContext();

static inline Context& init() {
    return globalContext();
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CONTEXT_H_
