#ifdef DGL_USE_PYTORCH_NPU_STREAM
// Include torch_npu stream headers FIRST: c10 redefines LOG/CHECK
// macros; dmlc's definitions below must win so LOG(FATAL) throws
// (catchable) instead of aborting the process.
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#endif

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

extern "C" uint32_t aclrtlaunch_csr_slice_rows_scalar_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_rows_scalar_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* ret_indptr, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_rows_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data_or_null,
    void* rows, void* ret_indptr,
    void* ret_indices, void* ret_data, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_slice_rows_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data_or_null,
    void* rows, void* ret_indptr,
    void* ret_indices, void* ret_data, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}

// Cross-framework ordering: torch_npu executes eager ops on its own NPU
// stream and ACL context; DGL-side syncs (aclrtSynchronizeStream from the
// default context) do not reliably wait on it. Use torch's own wrapper on
// the torch current stream at every entry that consumes torch-written
// buffers. (No-op when torch stream integration is not compiled in.)
#ifdef DGL_USE_PYTORCH_NPU_STREAM
inline aclrtStream ResolveSliceStream(int64_t device_id) {
  return c10_npu::getCurrentNPUStream(
      static_cast<c10::DeviceIndex>(device_id)).stream();
}
inline void SyncSliceStream(int64_t device_id) {
  c10_npu::stream_synchronize(ResolveSliceStream(device_id));
}
#else
inline void SyncSliceStream(int64_t) {}
#endif
#endif

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
#ifdef DGL_USE_ASCEND
  // CPU fallback: the NPU slice kernels mishandle non-contiguous / unsorted
  // row selections in release builds (memory corruption). Correctness first —
  // same trade-off as COOSort_'s CPU fallback. The torch-stream sync below
  // guarantees torch-written inputs (H2D) are visible before the D2H copies.
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);
  SyncSliceStream(ctx.device_id);
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  DGLContext cpu_ctx{kDGLCPU, 0};
  CSRMatrix cpu_csr{
      csr.num_rows, csr.num_cols,
      csr.indptr.CopyTo(cpu_ctx), csr.indices.CopyTo(cpu_ctx),
      CSRHasData(csr) ? csr.data.CopyTo(cpu_ctx) : csr.data,
      csr.sorted};
  CSRMatrix out = CSRSliceRows<kDGLCPU, IdType>(cpu_csr, start, end);
  out.indptr = out.indptr.CopyTo(ctx);
  out.indices = out.indices.CopyTo(ctx);
  if (CSRHasData(out)) out.data = out.data.CopyTo(ctx);
  return out;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template CSRMatrix CSRSliceRows<kDGLAscend, int32_t>(CSRMatrix, int64_t, int64_t);
template CSRMatrix CSRSliceRows<kDGLAscend, int64_t>(CSRMatrix, int64_t, int64_t);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
#ifdef DGL_USE_ASCEND
  // CPU fallback: see the (start, end) variant above.
  auto ctx = csr.indptr->ctx;
  CHECK(ctx.device_type == kDGLAscend);
  SyncSliceStream(ctx.device_id);
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  DGLContext cpu_ctx{kDGLCPU, 0};
  CSRMatrix cpu_csr{
      csr.num_rows, csr.num_cols,
      csr.indptr.CopyTo(cpu_ctx), csr.indices.CopyTo(cpu_ctx),
      CSRHasData(csr) ? csr.data.CopyTo(cpu_ctx) : csr.data,
      csr.sorted};
  CSRMatrix out = CSRSliceRows<kDGLCPU, IdType>(cpu_csr, rows.CopyTo(cpu_ctx));
  out.indptr = out.indptr.CopyTo(ctx);
  out.indices = out.indices.CopyTo(ctx);
  if (CSRHasData(out)) out.data = out.data.CopyTo(ctx);
  return out;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

template CSRMatrix CSRSliceRows<kDGLAscend, int32_t>(CSRMatrix, NDArray);
template CSRMatrix CSRSliceRows<kDGLAscend, int64_t>(CSRMatrix, NDArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

