/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/dispatcher.cuh
 * \brief Dispatches to CUDA functions by data types.
 */
#ifndef DGL_ARRAY_CUDA_DISPATCHER_CUH_
#define DGL_ARRAY_CUDA_DISPATCHER_CUH_

#include <cusparse.h>

namespace dgl {
namespace aten {

/*! \brief cusparseXcsrgemm dispatcher */
template <typename DType>
struct CSRGEMM {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    BUG_ON(false) << "This piece of code should not be reached.";
    return 0;
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    BUG_ON(false) << "This piece of code should not be reached.";
    return 0;
  }
};

template <>
struct CSRGEMM<float> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    return cusparseScsrgemm2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    return cusparseScsrgemm2(args...);
  }
};

template <>
struct CSRGEMM<double> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    return cusparseDcsrgemm2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    return cusparseDcsrgemm2(args...);
  }
};

/*! \brief cusparseXcsrgeam dispatcher */
template <typename DType>
struct CSRGEAM {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    BUG_ON(false) << "This piece of code should not be reached.";
    return 0;
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    BUG_ON(false) << "This piece of code should not be reached.";
    return 0;
  }
};

template <>
struct CSRGEAM<float> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    return cusparseScsrgeam2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    return cusparseScsrgeam2(args...);
  }
};

template <>
struct CSRGEAM<double> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    return cusparseDcsrgeam2_bufferSizeExt(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    return cusparseDcsrgeam2(args...);
  }
};

};  // namespace aten
};  // namespace dgl

#endif  // DGL_ARRAY_CUDA_DISPATCHER_CUH_
