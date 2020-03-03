/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/rowwise_sampling.cc
 * \brief rowwise sampling
 */
#include <dgl/random.h>
#include <numeric>
#include "./rowwise_pick.h"

namespace dgl {
namespace aten {
namespace impl {
namespace {
// Equivalent to numpy expression: array[idx[off:off + len]]
template <typename IdxType, typename FloatType>
inline FloatArray DoubleSlice(FloatArray array, const IdxType* idx_data,
                              IdxType off, IdxType len) {
  const FloatType* array_data = static_cast<FloatType*>(array->data);
  FloatArray ret = FloatArray::Empty({len}, array->dtype, array->ctx);
  FloatType* ret_data = static_cast<FloatType*>(ret->data);
  for (int64_t j = 0; j < len; ++j) {
    if (idx_data)
      ret_data[j] = array_data[idx_data[off + j]];
    else
      ret_data[j] = array_data[off + j];
  }
  return ret;
}

template <typename IdxType, typename FloatType>
inline PickFn<IdxType> GetSamplingPickFn(
    int64_t num_samples, FloatArray prob, bool replace) {
  PickFn<IdxType> pick_fn = [prob, num_samples, replace]
    (IdxType rowid, IdxType off, IdxType len,
     const IdxType* col, const IdxType* data,
     IdxType* out_idx) {
      FloatArray prob_selected = DoubleSlice<IdxType, FloatType>(prob, data, off, len);
      RandomEngine::ThreadLocal()->Choice<IdxType, FloatType>(
          num_samples, prob_selected, out_idx, replace);
      for (int64_t j = 0; j < num_samples; ++j) {
        out_idx[j] += off;
      }
    };
  return pick_fn;
}

template <typename IdxType>
inline PickFn<IdxType> GetSamplingUniformPickFn(
    int64_t num_samples, bool replace) {
  PickFn<IdxType> pick_fn = [num_samples, replace]
    (IdxType rowid, IdxType off, IdxType len,
     const IdxType* col, const IdxType* data,
     IdxType* out_idx) {
      RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
          num_samples, len, out_idx, replace);
      for (int64_t j = 0; j < num_samples; ++j) {
        out_idx[j] += off;
      }
    };
  return pick_fn;
}
}  // namespace

/////////////////////////////// CSR ///////////////////////////////

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseSampling(CSRMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  CHECK(prob.defined());
  auto pick_fn = GetSamplingPickFn<IdxType, FloatType>(num_samples, prob, replace);
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
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix CSRRowWiseSamplingUniform<kDLCPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDLCPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

/////////////////////////////// COO ///////////////////////////////

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix COORowWiseSampling(COOMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  CHECK(prob.defined());
  auto pick_fn = GetSamplingPickFn<IdxType, FloatType>(num_samples, prob, replace);
  return COORowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix COORowWiseSampling<kDLCPU, int32_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDLCPU, int64_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDLCPU, int32_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDLCPU, int64_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);

template <DLDeviceType XPU, typename IdxType>
COOMatrix COORowWiseSamplingUniform(COOMatrix mat, IdArray rows,
                                    int64_t num_samples, bool replace) {
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return COORowWisePick(mat, rows, num_samples, replace, pick_fn);
}

template COOMatrix COORowWiseSamplingUniform<kDLCPU, int32_t>(
    COOMatrix, IdArray, int64_t, bool);
template COOMatrix COORowWiseSamplingUniform<kDLCPU, int64_t>(
    COOMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
