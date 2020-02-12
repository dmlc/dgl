/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_rowwise_sampling.cc
 * \brief CSR rowwise sampling
 */
#include <dgl/random.h>
#include <numeric>
#include "./rowwise_pick.h"

namespace dgl {
namespace aten {
namespace impl {
namespace {
// Equivalent to numpy expression: array[idx[offset:offset+len]]
template <typename IdxType, typename FloatType>
inline FloatArray DoubleSlice(FloatArray array, IdArray idx,
                             int64_t offset, int64_t len) {
  FloatArray ret = FloatArray::Empty({len}, array->dtype, array->ctx);
  const IdxType* idx_data = static_cast<IdxType*>(idx->data);
  const FloatType* array_data = static_cast<FloatType*>(array->data);
  FloatType* ret_data = static_cast<FloatType*>(ret->data);
  for (int64_t j = 0; j < len; ++j) {
    ret_data[j] = array_data[idx_data[offset + j]];
  }
  return ret;
}
}  // namespace

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseSampling(CSRMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);

  PickFn<IdxType> pick_fn = [&mat, indptr, indices, &prob, num_samples, replace]
    (IdxType rowid, IdxType* out_row, IdxType* out_col, IdxType* out_idx) {
      const IdxType off = indptr[rowid];
      const IdxType len = indptr[rowid + 1] - off;
      // TODO(minjie): If efficiency is a problem, consider avoid creating
      //   explicit NDArrays by directly manipulating buffers.
      FloatArray prob_selected = DoubleSlice<IdxType, FloatType>(prob, mat.data, off, len);
      IdArray sampled = RandomEngine::ThreadLocal()->Choice<IdxType, FloatType>(
          num_samples, prob_selected, replace);
      const IdxType* sampled_data = static_cast<IdxType*>(sampled->data);
      for (int64_t j = 0; j < num_samples; ++j) {
        out_row[j] = rowid;
        out_col[j] = indices[off + sampled_data[j]];
        out_idx[j] = off + sampled_data[j];
      }
    };

  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix CSRRowWiseSampling<kDLCPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLCPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLCPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDLCPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

template <DLDeviceType XPU, typename IdxType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat, IdArray rows,
                                int64_t num_samples, bool replace) {
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);

  PickFn<IdxType> pick_fn = [indptr, indices, num_samples, replace]
    (IdxType rowid, IdxType* out_row, IdxType* out_col, IdxType* out_idx) {
      const IdxType off = indptr[rowid];
      const IdxType len = indptr[rowid + 1] - off;
      // TODO(minjie): If efficiency is a problem, consider avoid creating
      //   explicit NDArrays by directly manipulating buffers.
      IdArray sampled = RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
          num_samples, len, replace);
      const IdxType* sampled_data = static_cast<IdxType*>(sampled->data);
      for (int64_t j = 0; j < num_samples; ++j) {
        out_row[j] = rowid;
        out_col[j] = indices[off + sampled_data[j]];
        out_idx[j] = off + sampled_data[j];
      }
    };

  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix CSRRowWiseSamplingUniform<kDLCPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDLCPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
