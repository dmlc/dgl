#include <dgl/array.h>
#include "../array_op.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <vector>

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

struct CooToCsrIndptrTiling {
  uint32_t nnz;
  uint32_t num_rows;
  uint32_t block_idx;
  uint32_t block_num;
};

extern "C" uint32_t aclrtlaunch_coo_to_csr_indptr(
    uint32_t blockDim, aclrtStream stream,
    void* rows, void* indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_to_csr_indptr_int64(
    uint32_t blockDim, aclrtStream stream,
    void* rows, void* indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_gather_int32(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_gather_int64(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_gather_float(
    uint32_t blockDim, aclrtStream stream,
    void* src, void* idx, void* dst, void* tiling);

#endif

namespace dgl {
namespace aten {
namespace impl {

namespace {

template <typename IdType>
struct IndptrKernelLauncher;

template <>
struct IndptrKernelLauncher<int32_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* rows, void* indptr, void* tiling) {
    aclError err = aclrtlaunch_coo_to_csr_indptr(
        blockDim, stream, rows, indptr, tiling);
    CHECK(err == ACL_SUCCESS) << "coo_to_csr_indptr launch failed: " << err;
  }
};

template <>
struct IndptrKernelLauncher<int64_t> {
  static void Launch(uint32_t blockDim, aclrtStream stream,
                     void* rows, void* indptr, void* tiling) {
    aclError err = aclrtlaunch_coo_to_csr_indptr_int64(
        blockDim, stream, rows, indptr, tiling);
    CHECK(err == ACL_SUCCESS) << "coo_to_csr_indptr_int64 launch failed: " << err;
  }
};

template <typename IdType>
CSRMatrix COOToCSRWithKernel(COOMatrix coo) {
  auto ctx = coo.row->ctx;
  CHECK(ctx.device_type == kDGLAscend) << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  ASCEND_CALL(aclrtSynchronizeDevice());

  const int64_t nnz = coo.row->shape[0];
  CHECK_NO_OVERFLOW(coo.row->dtype, nnz);

  if (nnz == 0) {
    NDArray indptr = NDArray::Empty({coo.num_rows + 1}, coo.row->dtype, ctx);
    ASCEND_CALL(aclrtMemset(indptr->data, (coo.num_rows + 1) * sizeof(IdType),
                             0, (coo.num_rows + 1) * sizeof(IdType)));
    NDArray indices = NDArray::Empty({0}, coo.row->dtype, ctx);
    NDArray data;
    if (COOHasData(coo)) {
      data = coo.data;
    } else {
      data = NDArray::Empty({0}, coo.row->dtype, ctx);
    }
    return CSRMatrix(
        coo.num_rows, coo.num_cols, indptr, indices, data, coo.col_sorted);
  }

  bool col_sorted = coo.col_sorted;

  // Helper to launch gather on NPU: dst[i] = src[perm[i]]
  auto launch_gather = [&](aclrtStream stream, NDArray dst,
                           NDArray src, NDArray perm_npu) {
    if (dst->shape[0] == 0) return;
    uint32_t gather_n = static_cast<uint32_t>(dst->shape[0]);
    void* gather_tiling_dev;
    ASCEND_CALL(aclrtMalloc(
        &gather_tiling_dev, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemcpyAsync(
        gather_tiling_dev, sizeof(uint32_t),
        &gather_n, sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE, stream));
    if (src->dtype.code == kDGLFloat && src->dtype.bits == 32) {
      aclrtlaunch_coo_gather_float(1, stream,
          src->data, perm_npu->data, dst->data, gather_tiling_dev);
    } else if (src->dtype.bits == 32) {
      aclrtlaunch_coo_gather_int32(1, stream,
          src->data, perm_npu->data, dst->data, gather_tiling_dev);
    } else {
      aclrtlaunch_coo_gather_int64(1, stream,
          src->data, perm_npu->data, dst->data, gather_tiling_dev);
    }
    ASCEND_CALL(aclrtSynchronizeStream(stream));
    ASCEND_CALL(aclrtFree(gather_tiling_dev));
  };

  if (!coo.row_sorted) {
    auto cpu_ctx = DGLContext{kDGLCPU, 0};

    // Only copy rows to CPU; cols and data stay on NPU
    IdArray cpu_row = coo.row.CopyTo(cpu_ctx);
    IdType* rows = cpu_row.Ptr<IdType>();

    std::vector<int64_t> perm(nnz);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [rows](int64_t a, int64_t b) { return rows[a] < rows[b]; });

    // Build sorted rows on CPU
    std::vector<IdType> sorted_rows(nnz);
    for (int64_t i = 0; i < nnz; i++) {
      sorted_rows[i] = rows[perm[i]];
    }

    // Copy sorted_rows and perm to NPU
    NDArray npu_sorted_rows = NDArray::Empty({nnz}, coo.row->dtype, ctx);
    NDArray perm_npu = aten::NewIdArray(nnz, ctx, 64);
    ASCEND_CALL(aclrtMemcpy(npu_sorted_rows->data, nnz * sizeof(IdType),
                            sorted_rows.data(), nnz * sizeof(IdType),
                            ACL_MEMCPY_HOST_TO_DEVICE));
    ASCEND_CALL(aclrtMemcpy(perm_npu->data, nnz * sizeof(int64_t),
                            perm.data(), nnz * sizeof(int64_t),
                            ACL_MEMCPY_HOST_TO_DEVICE));

    // Gather cols and data on NPU
    NDArray npu_sorted_cols = NDArray::Empty({nnz}, coo.col->dtype, ctx);
    NDArray npu_sorted_data;
    {
      aclrtStream gather_stream;
      ASCEND_CALL(aclrtCreateStream(&gather_stream));
      launch_gather(gather_stream, npu_sorted_cols, coo.col, perm_npu);

      if (COOHasData(coo)) {
        npu_sorted_data = NDArray::Empty({nnz}, coo.data->dtype, ctx);
        launch_gather(gather_stream, npu_sorted_data, coo.data, perm_npu);
      } else {
        // Create identity [0,1,...,nnz-1] and gather by perm
        // so npu_sorted_data[j] = perm[j] (original edge index for j-th sorted edge)
        IdArray cpu_data = aten::Range(0, nnz, coo.row->dtype.bits, cpu_ctx);
        NDArray npu_identity = cpu_data.CopyTo(ctx);
        npu_sorted_data = NDArray::Empty({nnz}, coo.row->dtype, ctx);
        launch_gather(gather_stream, npu_sorted_data, npu_identity, perm_npu);
      }
      ASCEND_CALL(aclrtDestroyStream(gather_stream));
    }

    coo.row = npu_sorted_rows;
    coo.col = npu_sorted_cols;
    coo.data = npu_sorted_data;
    col_sorted = false;
  } else {
    if (!COOHasData(coo)) {
      auto cpu_ctx = DGLContext{kDGLCPU, 0};
      IdArray cpu_data = aten::Range(0, nnz, coo.row->dtype.bits, cpu_ctx);
      coo.data = cpu_data.CopyTo(ctx);
    }
  }

  // -----------------------------------------------------------------------
  //  Multi-stream parallel: each stream processes a row slice and writes
  //  to its own private indptr buffer (full size). No GM coherence issue
  //  since each kernel only writes to its own buffer. After sync, merge
  //  by copying each block's row range into the final indptr.
  // -----------------------------------------------------------------------
  NDArray indptr = NDArray::Empty({coo.num_rows + 1}, coo.row->dtype, ctx);

  int64_t indptr_bytes = (coo.num_rows + 1) * sizeof(IdType);
  ASCEND_CALL(aclrtMemset(indptr->data, indptr_bytes, 0, indptr_bytes));

  // Max 8 streams = up to 8x parallel binary search, avoids excessive memory
  static constexpr uint32_t kMaxStreams = 8;
  uint32_t numStreams = std::max(
      std::min(static_cast<uint32_t>(coo.num_rows), kMaxStreams),
      static_cast<uint32_t>(1));

  // Cache streams per device to avoid create/destroy overhead
  static std::map<int, std::vector<aclrtStream>> streamCache;
  auto& streams = streamCache[ctx.device_id];
  if (streams.size() < numStreams) {
    for (uint32_t i = streams.size(); i < numStreams; i++) {
      aclrtStream s;
      ASCEND_CALL(aclrtCreateStream(&s));
      streams.push_back(s);
    }
  }

  // Each stream gets a private indptr buffer → no GM coherence issue
  std::vector<void*> perBlockIndptr(numStreams, nullptr);
  for (uint32_t i = 0; i < numStreams; i++) {
    ASCEND_CALL(aclrtMalloc(&perBlockIndptr[i], indptr_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemset(perBlockIndptr[i], indptr_bytes, 0, indptr_bytes));
  }
  ASCEND_CALL(aclrtSynchronizeDevice());

  void* row_ptr = coo.row.Ptr<IdType>();

  for (uint32_t i = 0; i < numStreams; i++) {
    CooToCsrIndptrTiling tilingHost;
    tilingHost.nnz = static_cast<uint32_t>(nnz);
    tilingHost.num_rows = static_cast<uint32_t>(coo.num_rows);
    tilingHost.block_idx = i;
    tilingHost.block_num = numStreams;

    CooToCsrIndptrTiling* tilingDev = nullptr;
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&tilingDev),
        sizeof(CooToCsrIndptrTiling),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMemcpyAsync(
        tilingDev, sizeof(CooToCsrIndptrTiling),
        &tilingHost, sizeof(CooToCsrIndptrTiling),
        ACL_MEMCPY_HOST_TO_DEVICE, streams[i]));

    IndptrKernelLauncher<IdType>::Launch(
        1, streams[i], row_ptr, perBlockIndptr[i], tilingDev);

    ASCEND_CALL(aclrtSynchronizeStream(streams[i]));
    ASCEND_CALL(aclrtFree(tilingDev));
  }

  // Full device sync ensures private buffer writes are visible to D2D memcpy
  ASCEND_CALL(aclrtSynchronizeDevice());

  // Merge: copy each block's row slice from private buffer to final indptr
  for (uint32_t i = 0; i < numStreams; i++) {
    uint32_t rowsPerBlock = (static_cast<uint32_t>(coo.num_rows) + numStreams - 1) / numStreams;
    uint32_t rowStart = i * rowsPerBlock;
    uint32_t rowEnd = std::min(rowStart + rowsPerBlock, static_cast<uint32_t>(coo.num_rows));
    if (rowEnd > rowStart) {
      int64_t copy_offset = (rowStart + 1) * sizeof(IdType);
      int64_t copy_bytes = (rowEnd - rowStart) * sizeof(IdType);
      ASCEND_CALL(aclrtMemcpy(
          static_cast<char*>(indptr->data) + copy_offset, copy_bytes,
          static_cast<char*>(perBlockIndptr[i]) + copy_offset, copy_bytes,
          ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
    ASCEND_CALL(aclrtFree(perBlockIndptr[i]));
  }

  ASCEND_CALL(aclrtSynchronizeDevice());

  return CSRMatrix(
      coo.num_rows, coo.num_cols, indptr, coo.col, coo.data, col_sorted);
}

}  // anonymous namespace

template <>
CSRMatrix COOToCSR<kDGLAscend, int32_t>(COOMatrix coo) {
#ifdef DGL_USE_ASCEND
  return COOToCSRWithKernel<int32_t>(coo);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template <>
CSRMatrix COOToCSR<kDGLAscend, int64_t>(COOMatrix coo) {
#ifdef DGL_USE_ASCEND
  return COOToCSRWithKernel<int64_t>(coo);
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl
