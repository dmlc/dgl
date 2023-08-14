#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include <dgl/runtime/ndarray.h>

static constexpr DGLContext CTX = DGLContext{kDGLCPU, 0};
static constexpr DGLContext CPU = DGLContext{kDGLCPU, 0};
#ifdef DGL_USE_CUDA
static constexpr DGLContext GPU = DGLContext{kDGLCUDA, 0};
#endif

template <typename T>
inline T* Ptr(dgl::runtime::NDArray nd) {
  return static_cast<T*>(nd->data);
}

inline int64_t* PI64(dgl::runtime::NDArray nd) {
  return static_cast<int64_t*>(nd->data);
}

inline int32_t* PI32(dgl::runtime::NDArray nd) {
  return static_cast<int32_t*>(nd->data);
}

inline int64_t Len(dgl::runtime::NDArray nd) { return nd->shape[0]; }

template <typename T>
inline bool ArrayEQ(dgl::runtime::NDArray a1, dgl::runtime::NDArray a2) {
  if (a1->ndim != a2->ndim) return false;
  if (a1->dtype != a2->dtype) return false;
  if (a1->ctx != a2->ctx) return false;
  if (a1.NumElements() != a2.NumElements()) return false;
  if (a1.NumElements() == 0) return true;
  int64_t num = 1;
  for (int i = 0; i < a1->ndim; ++i) {
    if (a1->shape[i] != a2->shape[i]) return false;
    num *= a1->shape[i];
  }
  a1 = a1.CopyTo(CPU);
  a2 = a2.CopyTo(CPU);
  for (int64_t i = 0; i < num; ++i)
    if (static_cast<T*>(a1->data)[i] != static_cast<T*>(a2->data)[i])
      return false;
  return true;
}

template <typename T>
inline bool IsInArray(dgl::runtime::NDArray a, T x) {
  if (!a.defined() || a->shape[0] == 0) return false;
  for (int64_t i = 0; i < a->shape[0]; ++i) {
    if (x == static_cast<T*>(a->data)[i]) return true;
  }
  return false;
}

#endif  // TEST_COMMON_H_
