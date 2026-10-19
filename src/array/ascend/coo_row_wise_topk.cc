/**
 * Copyright (c) 2024 by Contributors
 * @file coo_row_wise_topk.cc
 * @brief Ascend host implementation for COO row-wise topk (assembly route).
 *
 * COOToCSR (full graph, NPU counting sort) followed by the native CSR
 * topk kernel — the same assembly structure as the COO uniform sampler
 * (upstream CUDA does the same for its COO sampling). Sorting inside
 * COOToCSR handles unsorted input; rows listed in `rows` index the
 * full-graph indptr directly, so the output rows are the real node ids.
 */

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#endif  // DGL_USE_ASCEND

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>

#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COORowWiseTopk(
    COOMatrix mat, IdArray rows, int64_t k, NDArray weight, bool ascending) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.row->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for COORowWiseTopk";
  CHECK(rows->dtype == mat.row->dtype)
      << "Expected rows to have the same dtype as the graph";

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.row->dtype.bits;

  if (num_rows == 0 || mat.row->shape[0] == 0 || k == 0) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(
        mat.num_rows, mat.num_cols, empty_row, empty_row, empty_row);
  }

  // Full-graph conversion: the weight array is per-edge in COO order, and
  // COOToCSR carries the original edge ids in csr.data, so the CSR kernel
  // gathers weights through the same eid mapping the CPU path uses.
  CSRMatrix csr = COOToCSR(mat);

  return CSRRowWiseTopk<kDGLAscend, IdType, DType>(
      csr, rows, k, weight, ascending);
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template COOMatrix COORowWiseTopk<kDGLAscend, int32_t, float>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int64_t, float>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int32_t, double>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int64_t, double>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int32_t, int32_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int64_t, int32_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int32_t, int64_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseTopk<kDGLAscend, int64_t, int64_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
