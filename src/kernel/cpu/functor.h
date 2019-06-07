/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cpu/functor.h
 * \brief Functors for template on CPU
 */
#ifndef DGL_KERNEL_CPU_FUNCTOR_H_
#define DGL_KERNEL_CPU_FUNCTOR_H_

#include <dmlc/omp.h>

#include <algorithm>

#include "../binary_reduce_common.h"

namespace dgl {
namespace kernel {

// Reducer functor specialization
template <typename DType>
struct ReduceSum<kDLCPU, DType> {
  static void Call(DType* addr, DType val) {
#pragma omp atomic
    *addr += val;
  }
  static DType BackwardCall(DType val, DType accum) {
    return 1;
  }
};

template <typename DType>
struct ReduceMax<kDLCPU, DType> {
  static void Call(DType* addr, DType val) {
#pragma omp critical
    *addr = std::max(*addr, val);
  }
  static DType BackwardCall(DType val, DType accum) {
    return static_cast<DType>(val == accum);
  }
};

template <typename DType>
struct ReduceMin<kDLCPU, DType> {
  static void Call(DType* addr, DType val) {
#pragma omp critical
    *addr = std::min(*addr, val);
  }
  static DType BackwardCall(DType val, DType accum) {
    return static_cast<DType>(val == accum);
  }
};

template <typename DType>
struct ReduceProd<kDLCPU, DType> {
  static void Call(DType* addr, DType val) {
#pragma omp atomic
    *addr *= val;
  }
  static DType BackwardCall(DType val, DType accum) {
    return accum / val;
  }
};

template <typename DType>
struct ReduceNone<kDLCPU, DType> {
  static void Call(DType* addr, DType val) {
    *addr = val;
  }
  static DType BackwardCall(DType val, DType accum) {
    return 1;
  }
};

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CPU_FUNCTOR_H_
