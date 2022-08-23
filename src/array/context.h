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
#define Context_HIDDEN
#define Context_EXPORT __declspec(dllexport)
#else  // _WIN32
#if defined(__GNUC__)
#define Context_EXPORT __attribute__((__visibility__("default")))
#define Context_HIDDEN __attribute__((__visibility__("hidden")))
#else  // defined(__GNUC__)
#define Context_EXPORT
#define Context_HIDDEN
#endif  // defined(__GNUC__)
#endif  // _WIN32

class Context_EXPORT Context {
 public:
  Context();

  // Enabling or disable use libxsmm for Spmm
  void setLibxsmm(bool);
  bool libxsmm() const;

 private:
  bool _libxsmm = false;
};

Context_EXPORT Context& globalContext();

static inline void initContext() {
    globalContext();
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CONTEXT_H_
