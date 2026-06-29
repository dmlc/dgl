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

extern "C" uint32_t aclrtlaunch_csr_get_data_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data,
    void* rows, void* cols, void* out, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_get_data_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data,
    void* rows, void* cols, void* out, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_get_data_weighted_int32(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data,
    void* rows, void* cols, void* weights,
    void* out, void* tiling);

extern "C" uint32_t aclrtlaunch_csr_get_data_weighted_int64(
    uint32_t blockDim, aclrtStream stream,
    void* indptr, void* indices, void* data,
    void* rows, void* cols, void* weights,
    void* out, void* tiling);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}
}
#endif

#include <dgl/array.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids,
    NDArray weights, DType filler) {
#ifdef DGL_USE_ASCEND
  auto ctx = rows->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for CSRGetData";

  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];
  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
      << "Invalid row and col id array.";

  const int64_t rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen},
      return_eids ? csr.indices->dtype : weights->dtype, ctx);
  if (rstlen == 0) return rst;

  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  auto stream = dgl::runtime::getCurrentAscendStream();

  uint32_t numRows = static_cast<uint32_t>(csr.num_rows);
  uint32_t numCols = static_cast<uint32_t>(csr.num_cols);
  uint32_t nnz = static_cast<uint32_t>(csr.indices->shape[0]);
  uint32_t n = static_cast<uint32_t>(rstlen);
  uint32_t block_dim = 1;

  uint32_t tiling_bytes = 24;  // 5 u32 + 8 bytes (int64 or float)
  void* tiling_dev = nullptr;
  ASCEND_CALL(aclrtMalloc(&tiling_dev, tiling_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  // Pack tiling: [numRows, numCols, nnz, n, filler_bytes]
  // filler uses 8 bytes: int64 for return_eids, float for weighted
  struct {
    uint32_t numRows;
    uint32_t numCols;
    uint32_t nnz;
    uint32_t n;
    union {
      int64_t i64;
      float f32;
    } filler;
  } tiling_host;
  tiling_host.numRows = numRows;
  tiling_host.numCols = numCols;
  tiling_host.nnz = nnz;
  tiling_host.n = n;
  if (return_eids) {
    tiling_host.filler.i64 = static_cast<int64_t>(filler);
  } else {
    tiling_host.filler.f32 = static_cast<float>(filler);
  }

  ASCEND_CALL(aclrtMemcpy(tiling_dev, tiling_bytes,
                           &tiling_host, tiling_bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE));

  if (return_eids) {
    // return_eids=true: returning edge IDs (IdType == DType)
    if (std::is_same<IdType, int32_t>::value) {
      aclError err = aclrtlaunch_csr_get_data_int32(
          block_dim, stream,
          csr.indptr->data, csr.indices->data,
          CSRHasData(csr) ? csr.data->data : csr.indices->data,
          rows->data, cols->data, rst->data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_get_data_int32 launch failed: " << err;
    } else {
      aclError err = aclrtlaunch_csr_get_data_int64(
          block_dim, stream,
          csr.indptr->data, csr.indices->data,
          CSRHasData(csr) ? csr.data->data : csr.indices->data,
          rows->data, cols->data, rst->data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_get_data_int64 launch failed: " << err;
    }
  } else {
    // Weighted path: return weight values (DType = float)
    if (std::is_same<IdType, int32_t>::value) {
      aclError err = aclrtlaunch_csr_get_data_weighted_int32(
          block_dim, stream,
          csr.indptr->data, csr.indices->data,
          CSRHasData(csr) ? csr.data->data : csr.indices->data,
          rows->data, cols->data, weights->data, rst->data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_get_data_weighted_int32 launch failed: " << err;
    } else {
      aclError err = aclrtlaunch_csr_get_data_weighted_int64(
          block_dim, stream,
          csr.indptr->data, csr.indices->data,
          CSRHasData(csr) ? csr.data->data : csr.indices->data,
          rows->data, cols->data, weights->data, rst->data, tiling_dev);
      CHECK(err == ACL_SUCCESS) << "csr_get_data_weighted_int64 launch failed: " << err;
    }
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));

  return rst;
#else
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
#endif
}

// Explicit instantiations for the 6-arg version (return_eids=true, IdType=DType)
template NDArray CSRGetData<kDGLAscend, int32_t, int32_t>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, int32_t);
template NDArray CSRGetData<kDGLAscend, int64_t, int64_t>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, int64_t);

// Explicit instantiations for weighted version (return_eids=false, DType=float/double)
template NDArray CSRGetData<kDGLAscend, int32_t, float>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, float);
template NDArray CSRGetData<kDGLAscend, int64_t, float>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, float);
template NDArray CSRGetData<kDGLAscend, int32_t, double>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, double);
template NDArray CSRGetData<kDGLAscend, int64_t, double>(
    CSRMatrix, NDArray, NDArray, bool, NDArray, double);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

