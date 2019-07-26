#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_

#include <dgl/runtime/ndarray.h>

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

inline int64_t Len(dgl::runtime::NDArray nd) {
  return nd->shape[0];
}

template <typename T>
inline bool ArrayEQ(dgl::runtime::NDArray a1, dgl::runtime::NDArray a2) {
  if (a1->ndim != a2->ndim) return false;
  int64_t num = 1;
  for (int i = 0; i < a1->ndim; ++i) {
    if (a1->shape[i] != a2->shape[i])
      return false;
    num *= a1->shape[i];
  }
  for (int64_t i = 0; i < num; ++i)
    if (static_cast<T*>(a1->data)[i] != static_cast<T*>(a2->data)[i])
      return false;
  return true;
}

static constexpr DLContext CTX = DLContext{kDLCPU, 0};

#endif  // TEST_COMMON_H_
