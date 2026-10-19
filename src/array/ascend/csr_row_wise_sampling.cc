/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling.cc
 * @brief Ascend host launcher for weighted (probability/mask) CSR row-wise
 * sampling.
 *
 * Implements impl::CSRRowWiseSampling<kDGLAscend, IdType, FloatType> as a
 * native AscendC kernel path (no CPU fallback). Computes max degree of the
 * requested rows (via CSRGetRowNNZ) to size per-row workspaces, over-allocates
 * the output, launches a single fused sampling kernel (blockDim=1), then trims
 * via CreateView.
 *
 * NOTE: AscendC forbids double-precision operations in __aicore__ functions,
 * so the kernel operates in float. Only FloatType=float is instantiated; the
 * dispatch rejects float64 probability up front.
 */

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;              \
  }

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_int32_f32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data, void* rows, void* prob,
    void* out_ptr, void* out_rows, void* out_cols, void* out_idxs,
    void* prob_ws, void* cdf_ws, void* used_ws, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_row_wise_sampling_int64_f32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data, void* rows, void* prob,
    void* out_ptr, void* out_rows, void* out_cols, void* out_idxs,
    void* prob_ws, void* cdf_ws, void* used_ws, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_USE_ASCEND

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/aten/csr.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND
namespace {

// Normalize a probability/mask NDArray to float32 on the same device as the
// input. The AscendC weighted kernel operates in float32 (double is forbidden
// in __aicore__). float32 passes through unchanged; float64 is down-cast;
// int8/uint8 masks are treated as 0.0/1.0. This is input dtype normalization
// only — the sampling itself runs natively on the NPU kernel.
NDArray NormalizeProbToFloat32(NDArray prob) {
  const auto& dt = prob->dtype;
  if (dt.code == kDGLFloat && dt.bits == 32) return prob;
  const int64_t n = prob->shape[0];
  DGLContext cpu_ctx{kDGLCPU, 0};
  DGLDataType f32{kDGLFloat, 32, 1};
  NDArray prob_cpu = prob.CopyTo(cpu_ctx);
  NDArray prob_f32_cpu = NDArray::Empty({n}, f32, cpu_ctx);
  float* dst = static_cast<float*>(prob_f32_cpu->data);
  if (dt.code == kDGLFloat && dt.bits == 64) {
    const double* src = static_cast<const double*>(prob_cpu->data);
    for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
  } else if (dt.code == kDGLInt && dt.bits == 8) {
    const int8_t* src = static_cast<const int8_t*>(prob_cpu->data);
    for (int64_t i = 0; i < n; ++i) dst[i] = (src[i] != 0) ? 1.0f : 0.0f;
  } else if (dt.code == kDGLUInt && dt.bits == 8) {
    const uint8_t* src = static_cast<const uint8_t*>(prob_cpu->data);
    for (int64_t i = 0; i < n; ++i) dst[i] = (src[i] != 0) ? 1.0f : 0.0f;
  } else {
    LOG(FATAL) << "Unsupported probability dtype for Ascend weighted sampling";
  }
  return prob_f32_cpu.CopyTo(prob->ctx);
}

// Dispatch IdType -> the matching aclrtlaunch_* symbol (float probability).
template <typename IdType>
struct WeightedSamplingLauncher;

template <>
struct WeightedSamplingLauncher<int32_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream, void* indptr,
                     void* indices, void* data, void* rows, void* prob,
                     void* out_ptr, void* out_rows, void* out_cols,
                     void* out_idxs, void* prob_ws, void* cdf_ws,
                     void* used_ws, void* tiling) {
    aclError err = aclrtlaunch_csr_row_wise_sampling_int32_f32(
        blockDim, stream, indptr, indices, data, rows, prob, out_ptr, out_rows,
        out_cols, out_idxs, prob_ws, cdf_ws, used_ws, tiling);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_int32_f32 launch failed: " << err;
  }
};

template <>
struct WeightedSamplingLauncher<int64_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream, void* indptr,
                     void* indices, void* data, void* rows, void* prob,
                     void* out_ptr, void* out_rows, void* out_cols,
                     void* out_idxs, void* prob_ws, void* cdf_ws,
                     void* used_ws, void* tiling) {
    aclError err = aclrtlaunch_csr_row_wise_sampling_int64_f32(
        blockDim, stream, indptr, indices, data, rows, prob, out_ptr, out_rows,
        out_cols, out_idxs, prob_ws, cdf_ws, used_ws, tiling);
    CHECK(err == ACL_SUCCESS)
        << "csr_row_wise_sampling_int64_f32 launch failed: " << err;
  }
};

}  // anonymous namespace
#endif  // DGL_USE_ASCEND

template <DGLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace) {
#ifdef DGL_USE_ASCEND
  // AscendC forbids double in __aicore__; NormalizeProbToFloat32 above casts
  // any float64/int8/uint8 input to float32, so FloatType is always float.
  static_assert(std::is_same<FloatType, float>::value,
                "Ascend weighted sampling only supports float32 probability");

  auto ctx = mat.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRRowWiseSampling";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  // Normalize probability/mask to float32 (host cast; sampling stays native).
  NDArray prob = NormalizeProbToFloat32(prob_or_mask);

  // If num_samples is -1, select all (prob > 0) neighbors without replacement.
  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.indptr->dtype.bits;

  if (num_rows == 0 || (num_samples == 0 && !select_all)) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(mat.num_rows, mat.num_cols, empty_row, empty_row,
                     empty_row);
  }

  // Per-row degrees: workspace sizing (max_deg) and select-all alloc bound.
  NDArray deg = CSRGetRowNNZ<kDGLAscend, IdType>(mat, rows);
  std::vector<IdType> deg_host(num_rows);
  ASCEND_CALL(aclrtMemcpy(deg_host.data(), num_rows * sizeof(IdType),
      deg->data, num_rows * sizeof(IdType),
      ACL_MEMCPY_DEVICE_TO_HOST));
  IdType max_deg = 0;
  IdType sum_deg = 0;
  for (int64_t i = 0; i < num_rows; ++i) {
    if (deg_host[i] > max_deg) max_deg = deg_host[i];
    sum_deg += deg_host[i];
  }

  // Output upper bound.
  int64_t max_output;
  if (select_all) {
    // Weighted select-all picks at most deg per row (prob>0 count <= deg).
    max_output = static_cast<int64_t>(sum_deg);
  } else {
    max_output = num_rows * num_samples;
  }

  IdArray picked_row = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_col = aten::NewIdArray(max_output, ctx, nbits);
  IdArray picked_idx = aten::NewIdArray(max_output, ctx, nbits);
  IdArray out_ptr = aten::NewIdArray(num_rows + 1, ctx, nbits);

  if (max_output == 0 || max_deg == 0) {
    return COOMatrix(mat.num_rows, mat.num_cols,
                     picked_row.CreateView({0}, picked_row->dtype),
                     picked_col.CreateView({0}, picked_col->dtype),
                     picked_idx.CreateView({0}, picked_idx->dtype));
  }

  const uint32_t umax_deg = static_cast<uint32_t>(max_deg);
  // Workspaces (sized to max_deg; reused per row, single core).
  void* prob_ws = nullptr;
  void* cdf_ws = nullptr;
  void* used_ws = nullptr;
  ASCEND_CALL(aclrtMalloc(&prob_ws, umax_deg * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMalloc(&cdf_ws, umax_deg * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMalloc(&used_ws, umax_deg * sizeof(uint32_t),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  // used_ws uses generation id (i+1); 0 means unused.
  ASCEND_CALL(aclrtMemset(used_ws, umax_deg * sizeof(uint32_t), 0,
                          umax_deg * sizeof(uint32_t)));

  auto stream = dgl::runtime::getCurrentAscendStream();
  const bool has_data = aten::CSRHasData(mat);
  void* data_ptr = has_data ? mat.data->data : nullptr;

  uint32_t tiling_data[6] = {
      static_cast<uint32_t>(num_rows),
      select_all ? 0u : static_cast<uint32_t>(num_samples),
      static_cast<uint32_t>(replace ? 1 : 0),
      static_cast<uint32_t>(has_data ? 1 : 0),
      static_cast<uint32_t>(
          RandomEngine::ThreadLocal()->RandInt(1000000000)),
      static_cast<uint32_t>(select_all ? 1 : 0),
  };
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, sizeof(tiling_data),
                          ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tiling_dev, sizeof(tiling_data), tiling_data,
                          sizeof(tiling_data), ACL_MEMCPY_HOST_TO_DEVICE));

  const uint32_t block_dim = 1;
  WeightedSamplingLauncher<IdType>::Launch(
      block_dim, stream, mat.indptr->data, mat.indices->data, data_ptr,
      rows->data, prob->data, out_ptr->data, picked_row->data,
      picked_col->data, picked_idx->data, prob_ws, cdf_ws, used_ws, tiling_dev);

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_dev));
  ASCEND_CALL(aclrtFree(prob_ws));
  ASCEND_CALL(aclrtFree(cdf_ws));
  ASCEND_CALL(aclrtFree(used_ws));

  // Read back the true total to trim the over-allocated output.
  IdType total = 0;
  ASCEND_CALL(aclrtMemcpy(&total, sizeof(IdType),
      static_cast<char*>(out_ptr->data) + num_rows * sizeof(IdType),
      sizeof(IdType), ACL_MEMCPY_DEVICE_TO_HOST));
  const int64_t new_len = static_cast<int64_t>(total);

  return COOMatrix(
      mat.num_rows, mat.num_cols,
      picked_row.CreateView({new_len}, picked_row->dtype),
      picked_col.CreateView({new_len}, picked_col->dtype),
      picked_idx.CreateView({new_len}, picked_idx->dtype));
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template COOMatrix CSRRowWiseSampling<kDGLAscend, int32_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLAscend, int64_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
