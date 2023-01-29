/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/dispatcher.cuh
 * @brief Templates to dispatch into different cuSPARSE routines based on the
 * type argument.
 */
#ifndef DGL_ARRAY_CUDA_CUSPARSE_DISPATCHER_CUH_
#define DGL_ARRAY_CUDA_CUSPARSE_DISPATCHER_CUH_

#include <cusparse.h>
#include <dgl/runtime/c_runtime_api.h>

#include "bf16.cuh"
#include "fp16.cuh"

namespace dgl {
namespace aten {

/** @brief cusparseXcsrgemm dispatcher */
template <typename DType>
struct CSRGEMM {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<cusparseStatus_t>(0);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<cusparseStatus_t>(0);
  }
};

template <>
struct CSRGEMM<__half> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgemm2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL) << "CSRGEMM::bufferSizeExt does not support dtype half (FP16).";
    return static_cast<cusparseStatus_t>(0);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgemm2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEMM::compute does not support dtype half (FP16).";
    return static_cast<cusparseStatus_t>(0);
  }
};

#if BF16_ENABLED
template <>
struct CSRGEMM<__nv_bfloat16> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgemm2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL)
        << "CSRGEMM::bufferSizeExt does not support dtype bfloat16 (BF16).";
    return static_cast<cusparseStatus_t>(0);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgemm2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgemm2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEMM::compute does not support dtype bfloat16 (BF16).";
    return static_cast<cusparseStatus_t>(0);
  }
};
#endif  // BF16_ENABLED

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

/** @brief cusparseXcsrgeam dispatcher */
template <typename DType>
struct CSRGEAM {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<cusparseStatus_t>(0);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    BUG_IF_FAIL(false) << "This piece of code should not be reached.";
    return static_cast<cusparseStatus_t>(0);
  }
};

template <>
struct CSRGEAM<__half> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgeam2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL) << "CSRGEAM::bufferSizeExt does not support dtype half (FP16).";
    return static_cast<cusparseStatus_t>(0);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgeam2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEAM::compute does not support dtype half (FP16).";
    return static_cast<cusparseStatus_t>(0);
  }
};

#if BF16_ENABLED
template <>
struct CSRGEAM<__nv_bfloat16> {
  template <typename... Args>
  static inline cusparseStatus_t bufferSizeExt(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgeam2_bufferSizeExt, so a
    // different implementation would be required.
    LOG(FATAL)
        << "CSRGEAM::bufferSizeExt does not support dtype bfloat16 (BF16).";
    return static_cast<cusparseStatus_t>(0);
  }

  template <typename... Args>
  static inline cusparseStatus_t nnz(Args... args) {
    return cusparseXcsrgeam2Nnz(args...);
  }

  template <typename... Args>
  static inline cusparseStatus_t compute(Args... args) {
    // TODO(ndickson): There is no cusparseHcsrgeam2, so a different
    // implementation would be required.
    LOG(FATAL) << "CSRGEAM::compute does not support dtype bfloat16 (BF16).";
    return static_cast<cusparseStatus_t>(0);
  }
};
#endif  // BF16_ENABLED

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

#endif  // DGL_ARRAY_CUDA_CUSPARSE_DISPATCHER_CUH_
