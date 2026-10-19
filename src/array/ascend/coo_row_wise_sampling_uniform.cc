/**
 * Copyright (c) 2024 by Contributors
 * @file coo_row_wise_sampling_uniform.cc
 * @brief Ascend host assembler for uniform COO row-wise sampling.
 *
 * Assembles existing NPU operators instead of launching a dedicated kernel:
 * COOToCSR (full graph) followed by the native CSR uniform-sampling kernel.
 * The row ids in `rows` are passed through unchanged because the full-graph
 * CSR indptr is indexed by real node ids, so the kernel's output rows are
 * already real node ids (no Range + IndexSelect remapping needed).
 */

#include <dgl/array.h>

#include <cstdint>

#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
COOMatrix COORowWiseSamplingUniform(
    COOMatrix mat, IdArray rows, int64_t num_samples, bool replace) {
#ifdef DGL_USE_ASCEND
  auto ctx = mat.row->ctx;
  CHECK(ctx.device_type == kDGLAscend)
      << "Expected Ascend device context for COORowWiseSamplingUniform";

  const bool select_all = (num_samples == -1);
  replace = (replace && !select_all);

  const int64_t num_rows = rows->shape[0];
  const uint8_t nbits = mat.row->dtype.bits;

  // Structural early exit for degenerate inputs (mirrors the CSR path).
  // Note: num_samples == 0 implies !select_all (select_all means -1), so
  // the redundant conjunct is omitted.
  if (num_rows == 0 || mat.row->shape[0] == 0 || num_samples == 0) {
    IdArray empty_row = aten::NewIdArray(0, ctx, nbits);
    return COOMatrix(
        mat.num_rows, mat.num_cols, empty_row, empty_row, empty_row);
  }

  // Full-graph conversion. Sorting inside COOToCSR handles unsorted input;
  // rows listed in `rows` index the full-graph indptr directly, so the
  // sampler's output rows are the real node ids.
  CHECK(rows->dtype == mat.row->dtype)
      << "Expected rows to have the same dtype as the graph";

  CSRMatrix csr = COOToCSR(mat);

  return CSRRowWiseSamplingUniform<kDGLAscend, IdType>(
      csr, rows, num_samples, replace);
#else
  LOG(FATAL) << "Ascend support is not compiled. "
                "Please compile with -DUSE_ASCEND=ON";
  return {};
#endif  // DGL_USE_ASCEND
}

template COOMatrix COORowWiseSamplingUniform<kDGLAscend, int32_t>(
    COOMatrix, IdArray, int64_t, bool);
template COOMatrix COORowWiseSamplingUniform<kDGLAscend, int64_t>(
    COOMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
