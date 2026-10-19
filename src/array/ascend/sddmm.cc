// ============================================================================
// SDDMM Ascend host dispatch — bridges DGL framework to sddmm_dot_coo kernel
// ============================================================================
// Pattern: follows src/array/ascend/spmm.cc
//   - extern "C" aclrtlaunch_sddmm_dot_coo declaration
//   - SDDMMCooAscend<IdType, DType> template (tiling + launch)
//   - SDDMMCoo<kDGLAscend, ...> / SDDMMCsr<kDGLAscend, ...> explicit specializations
//
// MVP scope: COO format, "dot" op, lhs_target=src(0), rhs_target=dst(2)
// Unsupported combos fall back to LOG(FATAL)
// ============================================================================

#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/aten/coo.h>
#include <dgl/aten/macro.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <dmlc/logging.h>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <Python.h>
#include <torch/extension.h>
#include "sddmm_copy_lhs_tiling.h"


#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

// ============================================================================
// Tiling struct (must match kernel definition)
// ============================================================================
constexpr uint32_t SDDMM_UB_RESERVED = 2 * 1024;
constexpr uint32_t SDDMM_MAX_BATCH = 255;
constexpr uint32_t SDDMM_DTYPE_FP32 = 0;
constexpr uint32_t SDDMM_DTYPE_FP16 = 1;
constexpr uint32_t SDDMM_ALIGN_BYTES = 32;
constexpr uint32_t SDDMM_HALF_SIZE = 2;

struct SddmmTilingData {
    uint32_t numEdges;
    uint32_t featDim;
    uint32_t blockDim;
    uint32_t edgesPerCore;
    uint32_t batchSize;
    uint32_t dtype;
    uint32_t featDimAligned;
    uint32_t ubSize;
};

// ============================================================================
// Kernel launch declaration
// ============================================================================
extern "C" uint32_t aclrtlaunch_sddmm_dot_coo(
    uint32_t blockDim, aclrtStream stream,
    void* lhs, void* rhs, void* row, void* col,
    void* out, void* tiling);

// New sddmm_copy_lhs kernel (NPU-native gather, no CPU roundtrip)
extern "C" uint32_t aclrtlaunch_sddmm_copy_lhs_kernel(
    uint32_t blockDim, aclrtStream stream,
    void* feat, void* index, void* out, void* tiling);

// SDDMM binary kernel (add/sub/mul/div: gather lhs + gather rhs + element-wise)
#include "sddmm_binary_tiling.h"
extern "C" uint32_t aclrtlaunch_sddmm_binary_kernel(
    uint32_t blockDim, aclrtStream stream,
    void* lhs, void* rhs, void* index_lhs, void* index_rhs,
    void* out, void* tiling);

// ============================================================================
// UB size — 910B3 = 192 KB
// ============================================================================
static uint32_t GetUbSize() {
  return 192 * 1024;
}

// ============================================================================
// Tiling computation
// ============================================================================
static void ComputeSddmmTiling(SddmmTilingData& tiling, uint32_t numEdges,
                                uint32_t featDim, uint32_t dtype,
                                int64_t coreNum, uint32_t ubSize) {
  tiling.numEdges = numEdges;
  tiling.featDim = featDim;
  tiling.dtype = dtype;
  tiling.ubSize = ubSize;

  tiling.blockDim = (numEdges < static_cast<uint32_t>(coreNum))
                     ? numEdges : static_cast<uint32_t>(coreNum);
  if (tiling.blockDim == 0) {
    tiling.edgesPerCore = 0;
  } else {
    tiling.edgesPerCore = (numEdges + tiling.blockDim - 1) / tiling.blockDim;
  }

  tiling.featDimAligned = (featDim * sizeof(float) + SDDMM_ALIGN_BYTES - 1)
                          / SDDMM_ALIGN_BYTES * SDDMM_ALIGN_BYTES / sizeof(float);
  if (tiling.featDimAligned == 0) tiling.featDimAligned = SDDMM_ALIGN_BYTES / sizeof(float);

  uint32_t ubAvailable = ubSize - SDDMM_UB_RESERVED;
  uint32_t featDimAligned = tiling.featDimAligned;
  uint32_t featDimAlignedH = (featDim * SDDMM_HALF_SIZE + SDDMM_ALIGN_BYTES - 1)
                             / SDDMM_ALIGN_BYTES * SDDMM_ALIGN_BYTES / SDDMM_HALF_SIZE;
  if (featDimAlignedH == 0) featDimAlignedH = SDDMM_ALIGN_BYTES / SDDMM_HALF_SIZE;

  uint32_t fixedCost = 0;
  if (dtype == SDDMM_DTYPE_FP32) {
    fixedCost += 2 * featDimAligned * sizeof(float);
    fixedCost += 2 * featDimAligned * sizeof(float);
  } else {
    fixedCost += 2 * featDimAlignedH * SDDMM_HALF_SIZE;
    fixedCost += 2 * featDimAlignedH * SDDMM_HALF_SIZE;
    fixedCost += featDimAligned * sizeof(float);
    fixedCost += featDimAligned * sizeof(float);
  }

  uint32_t outElementSize = (dtype == SDDMM_DTYPE_FP32) ? sizeof(float) : SDDMM_HALF_SIZE;
  uint32_t perBatch = featDimAligned * sizeof(float)
                    + sizeof(int32_t) * 2
                    + outElementSize
                    + sizeof(float);
  if (dtype == SDDMM_DTYPE_FP16) perBatch += sizeof(float);

  if (fixedCost + perBatch > ubAvailable) {
    tiling.batchSize = 1;
  } else {
    uint32_t remaining = ubAvailable - fixedCost;
    uint32_t batch = remaining / perBatch;
    if (batch == 0) batch = 1;
    tiling.batchSize = (batch < SDDMM_MAX_BATCH) ? batch : SDDMM_MAX_BATCH;
  }
}

namespace dgl {
namespace aten {

// ============================================================================
// NPU-native fallback for non-dot ops (add, sub, mul, div, copy_lhs, copy_rhs)
// Uses torch NPU operations: index_select + element-wise binary ops.
// All computation stays on NPU, no CPU round-trip.
// ============================================================================

// Convert DGL NDArray to torch::Tensor (zero-copy, shares memory)
static at::Tensor NDArrayToTorch(NDArray arr) {
  auto torch_device = c10::Device(c10::DeviceType::PrivateUse1, arr->ctx.device_id);
  c10::ScalarType dtype;
  if (arr->dtype.code == kDGLFloat && arr->dtype.bits == 32) dtype = torch::kFloat32;
  else if (arr->dtype.code == kDGLFloat && arr->dtype.bits == 16) dtype = torch::kHalf;
  else if (arr->dtype.code == kDGLFloat && arr->dtype.bits == 64) dtype = torch::kFloat64;
  else if (arr->dtype.code == kDGLInt && arr->dtype.bits == 32) dtype = torch::kInt32;
  else if (arr->dtype.code == kDGLInt && arr->dtype.bits == 64) dtype = torch::kInt64;
  else LOG(FATAL) << "Unsupported dtype: code=" << arr->dtype.code << " bits=" << arr->dtype.bits;

  std::vector<int64_t> shape(arr->shape, arr->shape + arr->ndim);
  auto options = torch::TensorOptions().dtype(dtype).device(torch_device);
  return torch::from_blob(arr->data, shape, options);
}

// Convert torch::Tensor back to DGL NDArray (zero-copy)
static NDArray TorchToNDArray(at::Tensor tensor, DGLContext ctx) {
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  DGLDataType dtype;
  auto scalar_type = tensor.scalar_type();
  if (scalar_type == torch::kFloat32) { dtype.code = kDGLFloat; dtype.bits = 32; }
  else if (scalar_type == torch::kHalf) { dtype.code = kDGLFloat; dtype.bits = 16; }
  else if (scalar_type == torch::kFloat64) { dtype.code = kDGLFloat; dtype.bits = 64; }
  else if (scalar_type == torch::kInt32) { dtype.code = kDGLInt; dtype.bits = 32; }
  else if (scalar_type == torch::kInt64) { dtype.code = kDGLInt; dtype.bits = 64; }
  else LOG(FATAL) << "Unsupported torch dtype";
  dtype.lanes = 1;

  // Create NDArray that wraps the torch tensor's data pointer
  NDArray ret = NDArray::Empty(shape, dtype, ctx);
  // Copy data from torch tensor to NDArray
  ASCEND_CALL(aclrtMemcpy(ret->data, ret.GetSize(),
                           tensor.data_ptr(), tensor.nbytes(),
                           ACL_MEMCPY_DEVICE_TO_DEVICE));
  return ret;
}

// Get the index array for a given target (0=src, 1=edge, 2=dst)
// Returns a torch::Tensor suitable for index_select
static at::Tensor GetTargetIndex(const COOMatrix& coo, int target, int64_t num_edges) {
  auto torch_device = c10::Device(c10::DeviceType::PrivateUse1, coo.row->ctx.device_id);
  if (target == 0) {
    // src = coo.row
    return NDArrayToTorch(coo.row);
  } else if (target == 2) {
    // dst = coo.col
    return NDArrayToTorch(coo.col);
  } else {
    // edge = [0, 1, ..., num_edges-1]
    return torch::arange(0, num_edges,
                         torch::TensorOptions().dtype(torch::kInt32).device(torch_device));
  }
}

// NPU-native SDDMMCoo for non-dot ops
// Pure aclrtMemcpy implementation - NO torch API, NO GIL needed
// Avoids stream sync issues between PyTorch NPU stream and DGL stream
template <typename IdType, typename DType>
void SDDMMCooNPUFallback(const std::string& op, const BcastOff& bcast,
                         const COOMatrix& coo, NDArray lhs, NDArray rhs,
                         NDArray out, int lhs_target, int rhs_target) {
  int64_t num_edges = coo.row->shape[0];
  if (num_edges == 0) return;

  DGLContext ctx = out->ctx;
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  // Get stream (use DGL stream, not PyTorch stream)
  aclrtStream stream = nullptr;

  // For copy_lhs/copy_rhs: use NPU-native gather kernel
  if (op == "copy_lhs" || op == "copy_rhs") {
    NDArray feat = (op == "copy_lhs") ? lhs : rhs;
    int target = (op == "copy_lhs") ? lhs_target : rhs_target;

    // Get index array (on NPU, no CPU roundtrip)
    // Cast to int32 if needed (kernel expects int32)
    NDArray idx;
    if (target == 0) idx = coo.row;
    else if (target == 2) idx = coo.col;
    else idx = aten::Range(0, num_edges, coo.row->dtype.bits, ctx);
    if (idx->dtype.bits != 32) {
      DGLContext cpu_ctx{kDGLCPU, 0};
      NDArray idx_cpu = idx.CopyTo(cpu_ctx);
      idx = aten::AsNumBits(idx_cpu, 32).CopyTo(ctx);
    }

    // Build tiling
    int64_t num_nodes = feat->shape[0];
    int64_t feat_dim = (feat->ndim > 1) ? feat->shape[1] : 1;
    uint32_t dtype_flag = (feat->dtype.bits == 32) ? 0 : 1;  // 0=FP32, 1=FP16

    // Get vector core count
    int64_t coreNum = 0;
    aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
    if (coreNum <= 0) coreNum = 40;

    SddmmCopyLhsTilingData tiling;
    tiling.numNodes = static_cast<uint32_t>(num_nodes);
    tiling.nnz = static_cast<uint32_t>(num_edges);
    tiling.featDim = static_cast<uint32_t>(feat_dim);
    tiling.dtype = dtype_flag;
    tiling.ubSize = 192 * 1024;

    uint32_t coreNumU32 = static_cast<uint32_t>(coreNum);
    tiling.blockDim = (tiling.nnz < coreNumU32) ? tiling.nnz : coreNumU32;
    if (tiling.blockDim == 0) tiling.blockDim = 1;
    tiling.edgesPerCore = (tiling.nnz + tiling.blockDim - 1) / tiling.blockDim;

    uint32_t alignBytes = 32;
    uint32_t elemSize = (dtype_flag == 0) ? 4 : 2;
    tiling.featDimAligned = (tiling.featDim * elemSize + alignBytes - 1) / alignBytes * alignBytes / elemSize;
    if (tiling.featDimAligned == 0) tiling.featDimAligned = alignBytes / elemSize;

    // Compute batchSize
    uint32_t ubAvailable = tiling.ubSize - 2 * 1024;  // UB_RESERVED
    uint32_t bufSize = tiling.featDimAligned * elemSize;
    uint32_t batchSize = ubAvailable / bufSize;
    if (batchSize == 0) batchSize = 1;
    if (batchSize > 4095) batchSize = 4095;
    tiling.batchSize = batchSize;

    // Allocate tiling on device
    void* tilingDev = nullptr;
    ASCEND_CALL(aclrtMalloc(&tilingDev, sizeof(SddmmCopyLhsTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemcpy(tilingDev, sizeof(SddmmCopyLhsTilingData), &tiling,
                             sizeof(SddmmCopyLhsTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    // Launch kernel
    aclrtStream stream = nullptr;
    uint32_t blockDim = tiling.blockDim;
    aclError launch_err = ACLRT_LAUNCH_KERNEL(sddmm_copy_lhs_kernel)(
        blockDim, stream, feat->data, idx->data, out->data, tilingDev);

    if (launch_err != ACL_SUCCESS) {
      LOG(FATAL) << "sddmm_copy_lhs_kernel launch failed, error code: " << launch_err;
    }

    ASCEND_CALL(aclrtSynchronizeStream(stream));
    ASCEND_CALL(aclrtFree(tilingDev));
  } else {
    // add/sub/mul/div: NPU binary kernel (gather lhs + gather rhs + element-wise)
    uint32_t binaryOp;
    if (op == "add") binaryOp = 0;
    else if (op == "sub") binaryOp = 1;
    else if (op == "mul") binaryOp = 2;
    else if (op == "div") binaryOp = 3;
    else LOG(FATAL) << "Unsupported op: " << op;

    // Get index arrays for lhs and rhs targets
    NDArray idx_lhs, idx_rhs;
    DGLContext ctx = out->ctx;
    if (lhs_target == 0) idx_lhs = coo.row;
    else if (lhs_target == 2) idx_lhs = coo.col;
    else idx_lhs = aten::Range(0, num_edges, coo.row->dtype.bits, ctx);
    if (rhs_target == 0) idx_rhs = coo.row;
    else if (rhs_target == 2) idx_rhs = coo.col;
    else idx_rhs = aten::Range(0, num_edges, coo.row->dtype.bits, ctx);

    // Cast indices to int32 if needed
    if (idx_lhs->dtype.bits != 32) {
      DGLContext cpu_ctx{kDGLCPU, 0};
      NDArray idx_cpu = idx_lhs.CopyTo(cpu_ctx);
      idx_lhs = aten::AsNumBits(idx_cpu, 32).CopyTo(ctx);
    }
    if (idx_rhs->dtype.bits != 32) {
      DGLContext cpu_ctx{kDGLCPU, 0};
      NDArray idx_cpu = idx_rhs.CopyTo(cpu_ctx);
      idx_rhs = aten::AsNumBits(idx_cpu, 32).CopyTo(ctx);
    }

    int64_t num_nodes_lhs = lhs->shape[0];
    int64_t num_nodes_rhs = IsNullArray(rhs) ? 0 : rhs->shape[0];
    int64_t feat_dim = (lhs->ndim > 1) ? lhs->shape[1] : 1;
    uint32_t dtype_flag = (sizeof(DType) == 4) ? 0 : 1;

    int64_t coreNum = 0;
    aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
    if (coreNum <= 0) coreNum = 40;

    SddmmBinaryTilingData tiling;
    tiling.nnz = static_cast<uint32_t>(num_edges);
    tiling.featDim = static_cast<uint32_t>(feat_dim);
    tiling.op = binaryOp;
    tiling.dtype = dtype_flag;
    tiling.numNodesLhs = static_cast<uint32_t>(num_nodes_lhs);
    tiling.numNodesRhs = static_cast<uint32_t>(num_nodes_rhs);
    tiling.ubSize = 192 * 1024;

    uint32_t coreNumU32 = static_cast<uint32_t>(coreNum);
    tiling.blockDim = (tiling.nnz < coreNumU32) ? tiling.nnz : coreNumU32;
    if (tiling.blockDim == 0) tiling.blockDim = 1;
    tiling.edgesPerCore = (tiling.nnz + tiling.blockDim - 1) / tiling.blockDim;

    uint32_t alignBytes = 32;
    uint32_t elemSize = (dtype_flag == 0) ? 4 : 2;
    tiling.featDimAligned = (tiling.featDim * elemSize + alignBytes - 1) / alignBytes * alignBytes / elemSize;
    if (tiling.featDimAligned == 0) tiling.featDimAligned = alignBytes / elemSize;
    tiling.featDimAlignedF32 = (tiling.featDim * sizeof(float) + alignBytes - 1) / alignBytes * alignBytes / sizeof(float);
    if (tiling.featDimAlignedF32 == 0) tiling.featDimAlignedF32 = alignBytes / sizeof(float);

    uint32_t ubAvailable = tiling.ubSize - 2 * 1024;
    uint32_t bufSize = tiling.featDimAligned * elemSize;
    if (dtype_flag == 1) bufSize = tiling.featDimAlignedF32 * sizeof(float);
    uint32_t batchSize = ubAvailable / (bufSize * 2);  // lhs + rhs buffers
    if (batchSize == 0) batchSize = 1;
    if (batchSize > 4095) batchSize = 4095;
    tiling.batchSize = batchSize;

    void* tilingDev = nullptr;
    ASCEND_CALL(aclrtMalloc(&tilingDev, sizeof(SddmmBinaryTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemcpy(tilingDev, sizeof(SddmmBinaryTilingData), &tiling,
                             sizeof(SddmmBinaryTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    aclrtStream stream = nullptr;
    uint32_t blockDim = tiling.blockDim;
    aclError launch_err = ACLRT_LAUNCH_KERNEL(sddmm_binary_kernel)(
        blockDim, stream, lhs->data, rhs->data,
        idx_lhs->data, idx_rhs->data,
        out->data, tilingDev);

    if (launch_err != ACL_SUCCESS) {
      LOG(FATAL) << "sddmm_binary_kernel launch failed, error code: " << launch_err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    ASCEND_CALL(aclrtFree(tilingDev));
  }
}
template <typename IdType, typename DType>

void SDDMMCsrNPUFallback(const std::string& op, const BcastOff& bcast,
                         const CSRMatrix& csr, NDArray lhs, NDArray rhs,
                         NDArray out, int lhs_target, int rhs_target) {
  // Convert CSR to COO (data_as_order=true preserves edge order)
  COOMatrix coo = CSRToCOO(csr, true);
  DGLContext npu_ctx = out->ctx;
  if (coo.row->ctx.device_type != kDGLAscend) {
    coo.row = coo.row.CopyTo(npu_ctx);
  }
  if (coo.col->ctx.device_type != kDGLAscend) {
    coo.col = coo.col.CopyTo(npu_ctx);
  }
  SDDMMCooNPUFallback<IdType, DType>(op, bcast, coo, lhs, rhs, out,
                                      lhs_target, rhs_target);
}

// ============================================================================
// SDDMMCooAscend — core implementation for COO format
// ============================================================================
// Supports: op="dot", lhs_target=0(src), rhs_target=2(dst)
// ============================================================================
template <typename IdType, typename DType>
void SDDMMCooAscend(const BcastOff& bcast, const COOMatrix& coo,
                    NDArray lhs, NDArray rhs, NDArray out,
                    int lhs_target, int rhs_target) {
  // MVP: only support dot op with src/dst targets
  CHECK_EQ(lhs_target, 0) << "Ascend SDDMM currently only supports lhs_target=0 (src)";
  CHECK_EQ(rhs_target, 2) << "Ascend SDDMM currently only supports rhs_target=2 (dst)";

  const IdType* row_ptr = coo.row.Ptr<IdType>();
  const IdType* col_ptr = coo.col.Ptr<IdType>();
  bool has_idx = !IsNullArray(coo.data);

  // Get edge count and feature dim
  int64_t num_edges = coo.row->shape[0];
  int64_t feat_dim = bcast.reduce_size;
  if (feat_dim == 0) feat_dim = bcast.lhs_len;

  if (num_edges == 0) return;

  DGLContext ctx = out->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  // Get stream
  aclrtStream stream = nullptr;

  // Get vector core count
  int64_t coreNum = 0;
  aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
  if (coreNum <= 0) coreNum = 40;

  // Determine dtype flag
  uint32_t dtype_flag;
  if (std::is_same<DType, float>::value) {
    dtype_flag = SDDMM_DTYPE_FP32;
  } else if (std::is_same<DType, uint16_t>::value) {
    dtype_flag = SDDMM_DTYPE_FP16;
  } else {
    LOG(FATAL) << "Ascend SDDMM only supports float32 and float16";
  }

  // Compute tiling
  SddmmTilingData tiling;
  ComputeSddmmTiling(tiling, static_cast<uint32_t>(num_edges),
                     static_cast<uint32_t>(feat_dim), dtype_flag,
                     coreNum, GetUbSize());

  // Allocate tiling on device
  void* tilingDev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tilingDev, sizeof(SddmmTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tilingDev, sizeof(SddmmTilingData), &tiling,
                          sizeof(SddmmTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

  // Get row/col arrays — if has_idx, we need to use coo.data as edge map
  // For u_dot_v, we use row as src index and col as dst index directly
  void* row_data = const_cast<void*>(static_cast<const void*>(row_ptr));
  void* col_data = const_cast<void*>(static_cast<const void*>(col_ptr));
  void* lhs_data = lhs->data;
  void* rhs_data = rhs->data;
  void* out_data = out->data;

  // Zero output first
  ASCEND_CALL(aclrtMemsetAsync(out_data, out->shape[0] * sizeof(DType), 0,
                               out->shape[0] * sizeof(DType), stream));

  // Launch kernel
  uint32_t blockDim = tiling.blockDim;
  aclError launch_err = ACLRT_LAUNCH_KERNEL(sddmm_dot_coo)(
      blockDim, stream,
      lhs_data, rhs_data, row_data, col_data,
      out_data, tilingDev);

  if (launch_err != ACL_SUCCESS) {
    LOG(FATAL) << "sddmm_dot_coo kernel launch failed, error code: " << launch_err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tilingDev));
}

// Helper: cast int64 COO index arrays to int32 (NPU kernel only supports int32)
// Arrays may be on NPU device, so copy to CPU, cast, copy back.
static COOMatrix CastCOOToInt32(const COOMatrix& coo) {
  DGLDataType int32_type{kDGLInt, 32, 1};
  DGLContext cpu_ctx{kDGLCPU, 0};

  // Copy row to CPU if on NPU
  NDArray row_cpu = (coo.row->ctx.device_type == kDGLCPU)
                        ? coo.row
                        : coo.row.CopyTo(cpu_ctx);
  NDArray col_cpu = (coo.col->ctx.device_type == kDGLCPU)
                        ? coo.col
                        : coo.col.CopyTo(cpu_ctx);

  int64_t nnz = coo.row->shape[0];
  NDArray row32_cpu = NDArray::Empty({nnz}, int32_type, cpu_ctx);
  NDArray col32_cpu = NDArray::Empty({nnz}, int32_type, cpu_ctx);

  const int64_t* row_src = static_cast<const int64_t*>(row_cpu->data);
  const int64_t* col_src = static_cast<const int64_t*>(col_cpu->data);
  int32_t* row_dst = static_cast<int32_t*>(row32_cpu->data);
  int32_t* col_dst = static_cast<int32_t*>(col32_cpu->data);
  for (int64_t i = 0; i < nnz; ++i) {
    CHECK_LE(row_src[i], 0x7FFFFFFFL) << "int32 overflow in COO row index";
    CHECK_LE(col_src[i], 0x7FFFFFFFL) << "int32 overflow in COO col index";
    row_dst[i] = static_cast<int32_t>(row_src[i]);
    col_dst[i] = static_cast<int32_t>(col_src[i]);
  }

  // Copy back to original device context
  NDArray row32 = row32_cpu.CopyTo(coo.row->ctx);
  NDArray col32 = col32_cpu.CopyTo(coo.col->ctx);

  return COOMatrix(coo.num_rows, coo.num_cols, row32, col32, coo.data,
                   coo.row_sorted, coo.col_sorted);
}

// Helper: cast int64 CSR index arrays to int32
static CSRMatrix CastCSRToInt32(const CSRMatrix& csr) {
  DGLDataType int32_type{kDGLInt, 32, 1};
  DGLContext cpu_ctx{kDGLCPU, 0};

  NDArray indptr_cpu = (csr.indptr->ctx.device_type == kDGLCPU)
                           ? csr.indptr
                           : csr.indptr.CopyTo(cpu_ctx);
  NDArray indices_cpu = (csr.indices->ctx.device_type == kDGLCPU)
                            ? csr.indices
                            : csr.indices.CopyTo(cpu_ctx);

  int64_t nnz = csr.indices->shape[0];
  int64_t nrows = csr.num_rows;
  NDArray indptr32_cpu = NDArray::Empty({nrows + 1}, int32_type, cpu_ctx);
  NDArray indices32_cpu = NDArray::Empty({nnz}, int32_type, cpu_ctx);

  const int64_t* indptr_src = static_cast<const int64_t*>(indptr_cpu->data);
  const int64_t* indices_src = static_cast<const int64_t*>(indices_cpu->data);
  int32_t* indptr_dst = static_cast<int32_t*>(indptr32_cpu->data);
  int32_t* indices_dst = static_cast<int32_t*>(indices32_cpu->data);
  for (int64_t i = 0; i <= nrows; ++i) {
    CHECK_LE(indptr_src[i], 0x7FFFFFFFL) << "int32 overflow in CSR indptr";
    indptr_dst[i] = static_cast<int32_t>(indptr_src[i]);
  }
  for (int64_t i = 0; i < nnz; ++i) {
    CHECK_LE(indices_src[i], 0x7FFFFFFFL) << "int32 overflow in CSR indices";
    indices_dst[i] = static_cast<int32_t>(indices_src[i]);
  }

  NDArray indptr32 = indptr32_cpu.CopyTo(csr.indptr->ctx);
  NDArray indices32 = indices32_cpu.CopyTo(csr.indices->ctx);

  // data array: cast or keep null
  NDArray data32;
  if (!IsNullArray(csr.data)) {
    NDArray data_cpu = (csr.data->ctx.device_type == kDGLCPU)
                           ? csr.data
                           : csr.data.CopyTo(cpu_ctx);
    NDArray data32_cpu = NDArray::Empty({nnz}, int32_type, cpu_ctx);
    const int64_t* data_src = static_cast<const int64_t*>(data_cpu->data);
    int32_t* data_dst = static_cast<int32_t*>(data32_cpu->data);
    for (int64_t i = 0; i < nnz; ++i) {
      CHECK_LE(data_src[i], 0x7FFFFFFFL) << "int32 overflow in CSR data";
      data_dst[i] = static_cast<int32_t>(data_src[i]);
    }
    data32 = data32_cpu.CopyTo(csr.data->ctx);
  } else {
    data32 = csr.data;
  }

  return CSRMatrix(nrows, csr.num_cols, indptr32, indices32, data32, csr.sorted);
}

// ============================================================================
// Template specializations — SDDMMCoo<kDGLAscend, ...>
// ============================================================================
template <>
void SDDMMCoo<kDGLAscend, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCooNPUFallback<int32_t, float>(op, bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  SDDMMCooAscend<int32_t, float>(bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCoo<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCooNPUFallback<int32_t, uint16_t>(op, bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  SDDMMCooAscend<int32_t, uint16_t>(bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
}

// int64 index: cast to int32 and delegate
template <>
void SDDMMCoo<kDGLAscend, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCooNPUFallback<int64_t, float>(op, bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  COOMatrix coo32 = CastCOOToInt32(coo);
  SDDMMCooAscend<int32_t, float>(bcast, coo32, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCoo<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCooNPUFallback<int64_t, uint16_t>(op, bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  COOMatrix coo32 = CastCOOToInt32(coo);
  SDDMMCooAscend<int32_t, uint16_t>(bcast, coo32, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCoo<kDGLAscend, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SDDMMCooNPUFallback<int32_t, double>(op, bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCoo<kDGLAscend, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SDDMMCooNPUFallback<int64_t, double>(op, bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
}

// ============================================================================
// Template specializations — SDDMMCsr<kDGLAscend, ...>
// Convert CSR to COO and delegate to SDDMMCoo
// ============================================================================
template <typename IdType, typename DType>
void SDDMMCsrAscend(const BcastOff& bcast, const CSRMatrix& csr,
                    NDArray lhs, NDArray rhs, NDArray out,
                    int lhs_target, int rhs_target) {
  // Convert CSR to COO (data_as_order=true produces original edge order on CPU)
  COOMatrix coo = CSRToCOO(csr, true);
  // Copy row/col to NPU device if they are on CPU
  DGLContext npu_ctx = out->ctx;
  if (coo.row->ctx.device_type != kDGLAscend) {
    coo.row = coo.row.CopyTo(npu_ctx);
  }
  if (coo.col->ctx.device_type != kDGLAscend) {
    coo.col = coo.col.CopyTo(npu_ctx);
  }
  SDDMMCooAscend<IdType, DType>(bcast, coo, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCsr<kDGLAscend, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCsrNPUFallback<int32_t, float>(op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  SDDMMCsrAscend<int32_t, float>(bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCsr<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCsrNPUFallback<int32_t, uint16_t>(op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  SDDMMCsrAscend<int32_t, uint16_t>(bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCsr<kDGLAscend, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCsrNPUFallback<int64_t, float>(op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  CSRMatrix csr32 = CastCSRToInt32(csr);
  SDDMMCsrAscend<int32_t, float>(bcast, csr32, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCsr<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  if (op != "dot") {
    SDDMMCsrNPUFallback<int64_t, uint16_t>(op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
    return;
  }
  CSRMatrix csr32 = CastCSRToInt32(csr);
  SDDMMCsrAscend<int32_t, uint16_t>(bcast, csr32, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCsr<kDGLAscend, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SDDMMCsrNPUFallback<int32_t, double>(op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
}

template <>
void SDDMMCsr<kDGLAscend, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target) {
  SDDMMCsrNPUFallback<int64_t, double>(op, bcast, csr, lhs, rhs, out, lhs_target, rhs_target);
}

// ============================================================================
// Hetero stubs — not yet implemented, LOG(FATAL) for all combinations
// ============================================================================
template <>
void SDDMMCsrHetero<kDGLAscend, int32_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCsrHetero<kDGLAscend, int32_t, uint16_t>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCsrHetero<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCsrHetero<kDGLAscend, int64_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCsrHetero<kDGLAscend, int64_t, uint16_t>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCsrHetero<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<CSRMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}

template <>
void SDDMMCooHetero<kDGLAscend, int32_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCooHetero<kDGLAscend, int32_t, uint16_t>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCooHetero<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCooHetero<kDGLAscend, int64_t, float>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCooHetero<kDGLAscend, int64_t, uint16_t>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}
template <>
void SDDMMCooHetero<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&,
    const std::vector<COOMatrix>&,
    const std::vector<NDArray>&, const std::vector<NDArray>&,
    std::vector<NDArray>, int, int,
    const std::vector<dgl_type_t>&, const std::vector<dgl_type_t>&) {
  LOG(FATAL) << "Ascend SDDMM Hetero not yet implemented";
}

} // namespace aten
} // namespace dgl

#endif  // DGL_USE_ASCEND
