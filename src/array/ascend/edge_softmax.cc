// ============================================================================
// EdgeSoftmax Ascend host dispatch — bridges DGL framework to edge_softmax kernel
// ============================================================================
// Pattern: follows src/array/ascend/spmm.cc, sddmm.cc
//   - extern "C" aclrtlaunch_edge_softmax_kernel declaration
//   - EdgeSoftmaxAscend template (tiling + launch)
//   - Edge_softmax_csr_forward/backward<kDGLAscend, ...> explicit specializations
//
// DGL interface:
//   forward:  efeat [num_edges, num_heads] + CSC indptr → out [num_edges, num_heads]
//   backward: out (forward output) + sds (out*grad_out) + CSC indptr → back_out
//
// Kernel backward formula (DGL-adapted):
//   dot = sum(sds) per segment per head
//   back_out = sds - out * dot
// ============================================================================

#include <dgl/array.h>
#include <dgl/aten/csr.h>
#include <dgl/runtime/device_api.h>
#include "../kernel_decl.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <dmlc/logging.h>
#include <cstdint>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include "edge_softmax_tiling.h"
// NPU gather kernel (from sddmm_copy_lhs) — replaces CPU IndexSelectND
// Forward-declare kernel and tiling struct (avoid include conflict with edge_softmax_tiling.h)
struct SddmmCopyLhsTilingData {
    uint32_t numNodes;
    uint32_t nnz;
    uint32_t featDim;
    uint32_t blockDim;
    uint32_t edgesPerCore;
    uint32_t batchSize;
    uint32_t dtype;
    uint32_t featDimAligned;
    uint32_t ubSize;
};
extern "C" uint32_t aclrtlaunch_sddmm_copy_lhs_kernel(
    uint32_t blockDim, aclrtStream stream,
    void* feat, void* index, void* out, void* tiling);

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

static NDArray IndexSelectND(NDArray src, NDArray index, DGLContext ctx) {
  int64_t n = index->shape[0];
  int64_t feat_dim = (src->ndim > 1) ? src->shape[1] : 1;
  int64_t num_nodes = src->shape[0];
  uint32_t dtype_flag = (src->dtype.bits == 32) ? 0 : 1;

  NDArray idx32 = index;
  if (idx32->dtype.bits != 32) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    NDArray idx_cpu = idx32.CopyTo(cpu_ctx);
    idx32 = dgl::aten::AsNumBits(idx_cpu, 32).CopyTo(ctx);
  }

  std::vector<int64_t> out_shape = {n, feat_dim};
  NDArray ret = NDArray::Empty(out_shape, src->dtype, ctx);

  int64_t coreNum = 0;
  aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
  if (coreNum <= 0) coreNum = 40;

  SddmmCopyLhsTilingData tiling;
  tiling.numNodes = static_cast<uint32_t>(num_nodes);
  tiling.nnz = static_cast<uint32_t>(n);
  tiling.featDim = static_cast<uint32_t>(feat_dim);
  tiling.dtype = dtype_flag;
  tiling.ubSize = 192 * 1024;
  uint32_t coreNumU32 = static_cast<uint32_t>(coreNum);
  tiling.blockDim = (tiling.nnz < coreNumU32) ? tiling.nnz : coreNumU32;
  if (tiling.blockDim == 0) tiling.blockDim = 1;
  tiling.edgesPerCore = (tiling.nnz + tiling.blockDim - 1) / tiling.blockDim;
  uint32_t elemSize = (dtype_flag == 0) ? 4 : 2;
  tiling.featDimAligned = (tiling.featDim * elemSize + 31) / 32 * 32 / elemSize;
  if (tiling.featDimAligned == 0) tiling.featDimAligned = 32 / elemSize;
  uint32_t ubAvailable = tiling.ubSize - 2 * 1024;
  uint32_t batchSize = ubAvailable / (tiling.featDimAligned * elemSize);
  if (batchSize == 0) batchSize = 1;
  if (batchSize > 4095) batchSize = 4095;
  tiling.batchSize = batchSize;

  void* tilingDev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tilingDev, sizeof(SddmmCopyLhsTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tilingDev, sizeof(SddmmCopyLhsTilingData), &tiling,
                           sizeof(SddmmCopyLhsTilingData), ACL_MEMCPY_HOST_TO_DEVICE));
  aclrtStream stream = nullptr;
  aclError err = ACLRT_LAUNCH_KERNEL(sddmm_copy_lhs_kernel)(
      tiling.blockDim, stream, src->data, idx32->data, ret->data, tilingDev);
  if (err != ACL_SUCCESS) LOG(FATAL) << "IndexSelectND gather kernel failed: " << err;
  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tilingDev));
  return ret;
}
static void ScatterBackND(NDArray dst, NDArray index, NDArray src, DGLContext ctx) {
  int64_t n = index->shape[0];
  int64_t stride = (src->ndim > 1) ? src->shape[1] : 1;
  uint32_t elemSize = (src->dtype.bits == 32) ? 4 : 2;
  uint32_t rowBytes = stride * elemSize;
  DGLContext cpu_ctx{kDGLCPU, 0};
  NDArray index_cpu = index.CopyTo(cpu_ctx);
  aclrtStream stream = nullptr;
  if (index_cpu->dtype.bits == 32) {
    const int32_t* idx = static_cast<const int32_t*>(index_cpu->data);
    for (int64_t i = 0; i < n; ++i) {
      ASCEND_CALL(aclrtMemcpyAsync(
          static_cast<char*>(dst->data) + static_cast<int64_t>(idx[i]) * rowBytes, rowBytes,
          static_cast<const char*>(src->data) + i * rowBytes, rowBytes,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
  } else {
    const int64_t* idx = static_cast<const int64_t*>(index_cpu->data);
    for (int64_t i = 0; i < n; ++i) {
      ASCEND_CALL(aclrtMemcpyAsync(
          static_cast<char*>(dst->data) + idx[i] * rowBytes, rowBytes,
          static_cast<const char*>(src->data) + i * rowBytes, rowBytes,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
  }
  ASCEND_CALL(aclrtSynchronizeStream(stream));
}
extern "C" uint32_t aclrtlaunch_edge_softmax_kernel(
    uint32_t blockDim, aclrtStream stream,
    void* efeat, void* indptr, void* out, void* gradOut, void* gradEfeat, void* tiling);

static uint32_t GetUbSize() {
  return 192 * 1024;
}

static void ComputeEdgeSoftmaxTiling(EdgeSoftmaxTilingData& tiling,
                                      uint32_t numNodes, uint32_t numEdges,
                                      uint32_t numHeads, uint32_t mode,
                                      uint32_t dtype,
                                      int64_t coreNum, uint32_t ubSize) {
  tiling.numNodes = numNodes;
  tiling.numEdges = numEdges;
  tiling.numHeads = numHeads;
  tiling.mode = mode;
  tiling.dtype = dtype;
  tiling.ubSize = ubSize;

  uint32_t coreNumU32 = static_cast<uint32_t>(coreNum);
  tiling.blockDim = (numNodes < coreNumU32) ? numNodes : coreNumU32;
  if (tiling.blockDim == 0) {
    tiling.rowsPerCore = 0;
  } else {
    tiling.rowsPerCore = (numNodes + tiling.blockDim - 1) / tiling.blockDim;
  }

  tiling.numHeadsAlignedF = (numHeads * sizeof(float) + ALIGN_BYTES - 1)
                            / ALIGN_BYTES * ALIGN_BYTES / sizeof(float);
  if (tiling.numHeadsAlignedF == 0)
    tiling.numHeadsAlignedF = ALIGN_BYTES / sizeof(float);

  tiling.numHeadsAlignedH = (numHeads * HALF_SIZE + ALIGN_BYTES - 1)
                            / ALIGN_BYTES * ALIGN_BYTES / HALF_SIZE;
  if (tiling.numHeadsAlignedH == 0)
    tiling.numHeadsAlignedH = ALIGN_BYTES / HALF_SIZE;

  uint32_t ubAvailable = ubSize > UB_RESERVED ? ubSize - UB_RESERVED : 0;
  uint32_t alignedColsF = tiling.numHeadsAlignedF;
  uint32_t alignedColsH = tiling.numHeadsAlignedH;
  uint32_t alignedCols = (dtype == DTYPE_FP16) ? alignedColsH : alignedColsF;
  bool isAR = (numHeads == 1);
  uint32_t scalarBufs = (mode == MODE_BACKWARD) ? 2 : 3;

  uint32_t indptrCost = (tiling.rowsPerCore + 1) * sizeof(int32_t);
  uint32_t scalarBufSize = isAR ? ALIGN_BYTES : alignedCols * sizeof(float);
  uint32_t fixedCost = indptrCost + scalarBufs * scalarBufSize + TMP_BUF_SIZE;

  uint32_t batchCoeff = (mode == MODE_BACKWARD) ? 5 : 4;
  uint32_t elemBytes = (dtype == DTYPE_FP16) ? (sizeof(float) + HALF_SIZE) : sizeof(float);
  uint32_t elemSize = isAR ? elemBytes : alignedCols * elemBytes;
  uint32_t batchCostPerRow = batchCoeff * elemSize;

  uint32_t maxBatchLimit = isAR ? MAX_BATCH_AR : MAX_BATCH;
  uint32_t maxBatch = 1;
  if (fixedCost < ubAvailable && batchCostPerRow > 0) {
    uint32_t remaining = ubAvailable - fixedCost;
    maxBatch = remaining / batchCostPerRow;
    if (maxBatch < 1) maxBatch = 1;
    if (maxBatch > maxBatchLimit) maxBatch = maxBatchLimit;
  }
  tiling.maxBatch = maxBatch;
}

static dgl::aten::CSRMatrix CastCSRToInt32(const dgl::aten::CSRMatrix& csr) {
  DGLContext cpu_ctx{kDGLCPU, 0};
  DGLDataType int32_type{kDGLInt, 32, 1};
  auto indptr_cpu = csr.indptr.CopyTo(cpu_ctx);
  auto indices_cpu = csr.indices.CopyTo(cpu_ctx);
  int64_t nnz = csr.indices->shape[0];
  int64_t nrows = csr.num_rows;
  dgl::runtime::NDArray indptr32_cpu = dgl::runtime::NDArray::Empty({nrows + 1}, int32_type, cpu_ctx);
  dgl::runtime::NDArray indices32_cpu = dgl::runtime::NDArray::Empty({nnz}, int32_type, cpu_ctx);
  const int64_t* ip = static_cast<const int64_t*>(indptr_cpu->data);
  const int64_t* idx = static_cast<const int64_t*>(indices_cpu->data);
  int32_t* ip32 = static_cast<int32_t*>(indptr32_cpu->data);
  int32_t* idx32 = static_cast<int32_t*>(indices32_cpu->data);
  for (int64_t i = 0; i <= nrows; ++i) ip32[i] = static_cast<int32_t>(ip[i]);
  for (int64_t i = 0; i < nnz; ++i) idx32[i] = static_cast<int32_t>(idx[i]);
  dgl::runtime::NDArray data32 = csr.data;
  if (!dgl::aten::IsNullArray(csr.data)) {
    auto data_cpu = csr.data.CopyTo(cpu_ctx);
    data32 = dgl::runtime::NDArray::Empty({nnz}, int32_type, cpu_ctx);
    const int64_t* d = static_cast<const int64_t*>(data_cpu->data);
    int32_t* d32 = static_cast<int32_t*>(data32->data);
    for (int64_t i = 0; i < nnz; ++i) d32[i] = static_cast<int32_t>(d[i]);
    data32 = data32.CopyTo(csr.indptr->ctx);
  }
  return dgl::aten::CSRMatrix(nrows, csr.num_cols,
                    indptr32_cpu.CopyTo(csr.indptr->ctx),
                    indices32_cpu.CopyTo(csr.indptr->ctx),
                    data32, csr.sorted);
}

namespace dgl {
namespace aten {

// ============================================================================
// Unified Ascend implementation for both forward and backward
// ============================================================================
template <typename IdType, typename DType>
static void EdgeSoftmaxAscendImpl(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out,
    NDArray sds, NDArray back_out, bool is_backward) {

  DGLContext ctx = is_backward ? out->ctx : efeat->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  // Convert CSR to int32 if needed
  CSRMatrix csr_used = csr;
  if (csr.indptr->dtype.bits == 64) {
    csr_used = CastCSRToInt32(csr);
  }

  int64_t num_nodes = csr_used.num_rows;
  int64_t num_edges = csr_used.indices->shape[0];

  if (num_nodes == 0 || num_edges == 0) {
    aclrtStream stream = nullptr;
    if (is_backward) {
      ASCEND_CALL(aclrtMemsetAsync(back_out->data, back_out.GetSize(), 0,
                                    back_out.GetSize(), stream));
      ASCEND_CALL(aclrtSynchronizeStream(stream));
    } else {
      ASCEND_CALL(aclrtMemsetAsync(out->data, out.GetSize(), 0,
                                    out.GetSize(), stream));
      ASCEND_CALL(aclrtSynchronizeStream(stream));
    }
    return;
  }

  // Determine num_heads from efeat (forward) or out (backward)
  NDArray feat_arr = is_backward ? out : efeat;
  int64_t num_heads = (feat_arr->ndim > 1) ? feat_arr->shape[1] : 1;

  // Determine dtype
  uint32_t dtype;
  if (feat_arr->dtype.code == kDGLFloat && feat_arr->dtype.bits == 32) {
    dtype = DTYPE_FP32;
  } else if (feat_arr->dtype.code == kDGLFloat && feat_arr->dtype.bits == 16) {
    dtype = DTYPE_FP16;
  } else {
    LOG(FATAL) << "Unsupported dtype for edge_softmax on Ascend: code="
               << feat_arr->dtype.code << " bits=" << feat_arr->dtype.bits;
  }

  uint32_t mode = is_backward ? MODE_BACKWARD : MODE_FORWARD;

  // Get core count
  int64_t coreNum = 0;
  aclrtGetDeviceInfo(ctx.device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
  if (coreNum <= 0) coreNum = 40;

  // Build tiling
  EdgeSoftmaxTilingData tiling;
  ComputeEdgeSoftmaxTiling(tiling,
                            static_cast<uint32_t>(num_nodes),
                            static_cast<uint32_t>(num_edges),
                            static_cast<uint32_t>(num_heads),
                            mode, dtype, coreNum, GetUbSize());

  // Allocate tiling on device
  void* tilingDev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tilingDev, sizeof(EdgeSoftmaxTilingData),
                           ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(tilingDev, sizeof(EdgeSoftmaxTilingData), &tiling,
                           sizeof(EdgeSoftmaxTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

  aclrtStream stream = nullptr;

  // Handle edge ID remapping: CSC may reorder edges, csr.data maps CSC pos → edge ID
  bool has_idx = !IsNullArray(csr_used.data);
  // Check if edge IDs are sequential (0,1,2,...) — if so, no remapping needed
  bool need_remap = false;
  if (has_idx) {
    // Copy edge IDs to host and check if sequential
    int64_t num_edges_check = csr_used.data->shape[0];
    if (csr_used.data->dtype.bits == 32) {
      std::vector<int32_t> ids_host(num_edges_check);
      ASCEND_CALL(aclrtMemcpy(ids_host.data(), num_edges_check * sizeof(int32_t),
                               csr_used.data->data, num_edges_check * sizeof(int32_t),
                               ACL_MEMCPY_DEVICE_TO_HOST));
      bool seq = true;
      for (int64_t i = 0; i < num_edges_check; ++i) {
        if (ids_host[i] != static_cast<int32_t>(i)) { seq = false; break; }
      }
      need_remap = !seq;
    } else {
      // int64: cast to int32 and check
      NDArray data_cpu = csr_used.data.CopyTo(DGLContext{kDGLCPU, 0});
      int64_t num_edges_check = data_cpu->shape[0];
      const int64_t* ids64 = static_cast<const int64_t*>(data_cpu->data);
      bool seq = true;
      for (int64_t i = 0; i < num_edges_check; ++i) {
        if (ids64[i] != i) { seq = false; break; }
      }
      need_remap = !seq;
    }
  }
  NDArray edge_ids = csr_used.data;

  if (!is_backward) {
    // Forward: efeat -> out
    // If has_idx, gather efeat by edge IDs to get CSC-ordered features
    NDArray efeat_used = efeat;
    if (need_remap) {
      efeat_used = IndexSelectND(efeat, edge_ids, ctx);
    }
    void* efeat_ptr = efeat_used->data;
    void* indptr_ptr = const_cast<void*>(static_cast<const void*>(csr_used.indptr->data));

    // Kernel outputs in CSC order
    NDArray out_csc;
    void* out_ptr;
    if (need_remap) {
      out_csc = NDArray::Empty({num_edges, static_cast<int64_t>(num_heads)},
                               efeat->dtype, ctx);
      out_ptr = out_csc->data;
    } else {
      out_ptr = out->data;
    }

    uint32_t blockDim = tiling.blockDim;
    aclError launch_err = ACLRT_LAUNCH_KERNEL(edge_softmax_kernel)(
        blockDim, stream,
        efeat_ptr, indptr_ptr, out_ptr,
        nullptr, nullptr, tilingDev);

    if (launch_err != ACL_SUCCESS) {
      LOG(FATAL) << "edge_softmax_kernel forward launch failed, error code: " << launch_err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    // Scatter back to original edge order
    if (need_remap) {
      ScatterBackND(out, edge_ids, out_csc, ctx);
    }
  } else {
    // Backward: out (forward output) + sds (out*grad_out) -> back_out
    // Kernel: efeat=null, indptr=graph indptr, out=forward output,
    //         gradOut=sds, gradEfeat=back_out
    // If has_idx, gather out and sds by edge IDs to CSC order
    NDArray out_used = out;
    NDArray sds_used = sds;
    if (need_remap) {
      out_used = IndexSelectND(out, edge_ids, ctx);
      sds_used = IndexSelectND(sds, edge_ids, ctx);
    }
    void* out_ptr = out_used->data;
    void* indptr_ptr = const_cast<void*>(static_cast<const void*>(csr_used.indptr->data));
    void* sds_ptr = sds_used->data;

    // Kernel outputs in CSC order
    NDArray back_out_csc;
    void* back_out_ptr;
    if (need_remap) {
      back_out_csc = NDArray::Empty({num_edges, static_cast<int64_t>(num_heads)},
                                    out->dtype, ctx);
      back_out_ptr = back_out_csc->data;
    } else {
      back_out_ptr = back_out->data;
    }

    uint32_t blockDim = tiling.blockDim;
    aclError launch_err = ACLRT_LAUNCH_KERNEL(edge_softmax_kernel)(
        blockDim, stream,
        nullptr, indptr_ptr, out_ptr,
        sds_ptr, back_out_ptr, tilingDev);

    if (launch_err != ACL_SUCCESS) {
      LOG(FATAL) << "edge_softmax_kernel backward launch failed, error code: " << launch_err;
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    // Scatter back to original edge order
    if (need_remap) {
      ScatterBackND(back_out, edge_ids, back_out_csc, ctx);
    }
  }

  ASCEND_CALL(aclrtFree(tilingDev));
}

// ============================================================================
// Forward specializations
// ============================================================================
template <>
void Edge_softmax_csr_forward<kDGLAscend, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out) {
  EdgeSoftmaxAscendImpl<int32_t, float>(op, bcast, csr, ufeat, efeat, out,
                                         NullArray(), NullArray(), false);
}

template <>
void Edge_softmax_csr_forward<kDGLAscend, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out) {
  EdgeSoftmaxAscendImpl<int64_t, float>(op, bcast, csr, ufeat, efeat, out,
                                         NullArray(), NullArray(), false);
}

template <>
void Edge_softmax_csr_forward<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out) {
  EdgeSoftmaxAscendImpl<int32_t, uint16_t>(op, bcast, csr, ufeat, efeat, out,
                                            NullArray(), NullArray(), false);
}

template <>
void Edge_softmax_csr_forward<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out) {
  EdgeSoftmaxAscendImpl<int64_t, uint16_t>(op, bcast, csr, ufeat, efeat, out,
                                            NullArray(), NullArray(), false);
}

// ============================================================================
// Backward specializations
// ============================================================================
template <>
void Edge_softmax_csr_backward<kDGLAscend, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray out, NDArray sds, NDArray back_out) {
  EdgeSoftmaxAscendImpl<int32_t, float>(op, bcast, csr, NullArray(),
                                         NullArray(), out, sds, back_out, true);
}

template <>
void Edge_softmax_csr_backward<kDGLAscend, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray out, NDArray sds, NDArray back_out) {
  EdgeSoftmaxAscendImpl<int64_t, float>(op, bcast, csr, NullArray(),
                                         NullArray(), out, sds, back_out, true);
}

template <>
void Edge_softmax_csr_backward<kDGLAscend, int32_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray out, NDArray sds, NDArray back_out) {
  EdgeSoftmaxAscendImpl<int32_t, uint16_t>(op, bcast, csr, NullArray(),
                                            NullArray(), out, sds, back_out, true);
}

template <>
void Edge_softmax_csr_backward<kDGLAscend, int64_t, uint16_t>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray out, NDArray sds, NDArray back_out) {
  EdgeSoftmaxAscendImpl<int64_t, uint16_t>(op, bcast, csr, NullArray(),
                                            NullArray(), out, sds, back_out, true);
}

// ============================================================================
// Double precision — not supported on Ascend, LOG(FATAL)
// ============================================================================
template <>
void Edge_softmax_csr_forward<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray) {
  LOG(FATAL) << "Double precision not supported for edge_softmax on Ascend.";
}
template <>
void Edge_softmax_csr_forward<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray) {
  LOG(FATAL) << "Double precision not supported for edge_softmax on Ascend.";
}
template <>
void Edge_softmax_csr_backward<kDGLAscend, int32_t, double>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray) {
  LOG(FATAL) << "Double precision not supported for edge_softmax on Ascend.";
}
template <>
void Edge_softmax_csr_backward<kDGLAscend, int64_t, double>(
    const std::string&, const BcastOff&, const CSRMatrix&,
    NDArray, NDArray, NDArray) {
  LOG(FATAL) << "Double precision not supported for edge_softmax on Ascend.";
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND
