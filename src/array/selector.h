/**
 *  Copyright (c) 2020 by Contributors
 * @file array/selector.h
 * @brief Selector functions to select among src/edge/dst attributes.
 */
#ifndef DGL_ARRAY_SELECTOR_H_
#define DGL_ARRAY_SELECTOR_H_

#include <dmlc/logging.h>

namespace dgl {

namespace {

#ifdef __CUDACC__
#define DGLDEVICE __device__
#define DGLINLINE __forceinline__
#else
#define DGLDEVICE
#define DGLINLINE inline
#endif  // __CUDACC__

}  // namespace

/**
 * @brief Select among src/edge/dst feature/idx.
 * @note the integer argument target specifies which target
 *       to choose, 0: src, 1: edge, 2: dst.
 */
template <int target>
struct Selector {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    LOG(INFO) << "Target " << target << " not recognized.";
    return src;
  }
};

template <>
template <typename T>
DGLDEVICE DGLINLINE T Selector<0>::Call(T src, T edge, T dst) {
  return src;
}

template <>
template <typename T>
DGLDEVICE DGLINLINE T Selector<1>::Call(T src, T edge, T dst) {
  return edge;
}

template <>
template <typename T>
DGLDEVICE DGLINLINE T Selector<2>::Call(T src, T edge, T dst) {
  return dst;
}

}  // namespace dgl

#endif  // DGL_ARRAY_SELECTOR_H_
