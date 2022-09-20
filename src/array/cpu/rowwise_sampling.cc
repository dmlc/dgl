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

template <typename IdxType>
inline NumPicksFn<IdxType> GetSamplingUniformNumPicksFn(
    int64_t num_samples, bool replace) {
  NumPicksFn<IdxType> num_picks_fn = [&]
    (IdxType rowid, IdxType off, IdxType len,
     const IdxType* col, const IdxType* data) {
      if (num_samples == -1)
        return len;
      else if (replace)
        return static_cast<IdxType>((len == 0) ? 0 : num_samples);
      else
        return std::min(len, static_cast<IdxType>(num_samples));
    };
  return num_picks_fn;
}

template <typename IdxType>
inline PickFn<IdxType> GetSamplingUniformPickFn(
    int64_t num_samples, bool replace) {
  PickFn<IdxType> pick_fn = [num_samples, replace]
    (IdxType rowid, IdxType off, IdxType len, IdxType num_picks,
     const IdxType* col, const IdxType* data,
     IdxType* out_idx) {
      if (num_samples == -1 || (!replace && len == num_picks)) {
        // fast path for selecting all
        for (int64_t j = 0; j < len; ++j)
          out_idx[j] = off + j;
      } else {
        RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
            num_picks, len, out_idx, replace);
        for (int64_t j = 0; j < num_picks; ++j)
          out_idx[j] += off;
      }
    };
  return pick_fn;
}

template <typename IdxType, typename FloatType>
inline NumPicksFn<IdxType> GetSamplingNumPicksFn(
    int64_t num_samples, FloatArray prob, bool replace) {
  NumPicksFn<IdxType> num_picks_fn = [&]
    (IdxType rowid, IdxType off, IdxType len,
     const IdxType* col, const IdxType* data) {
      const FloatType* prob_data = prob.Ptr<FloatType>();
      IdxType num_possible_picks = 0;
      for (int64_t j = 0; j < len; ++j) {
        const IdxType eid = data ? data[off + j] : off + j;
        if (prob_data[eid] > 0)
          ++num_possible_picks;
      }

      if (num_samples == -1)
        return num_possible_picks;
      else if (replace)
        return static_cast<IdxType>((len == 0) ? 0 : num_samples);
      else
        return std::min(static_cast<IdxType>(num_samples), num_possible_picks);
    };
  return num_picks_fn;
}

template <typename IdxType, typename FloatType>
inline PickFn<IdxType> GetSamplingPickFn(
    int64_t num_samples, FloatArray prob, bool replace) {
  PickFn<IdxType> pick_fn = [&]
    (IdxType rowid, IdxType off, IdxType len, IdxType num_picks,
     const IdxType* col, const IdxType* data,
     IdxType* out_idx) {
      const FloatType* prob_data = prob.Ptr<FloatType>();

      if (num_samples == -1 || (!replace && len == num_picks)) {
        // fast path for selecting all
        int64_t i = 0;
        for (int64_t j = 0; j < len; ++j) {
          const IdxType eid = data ? data[off + j] : off + j;
          if (prob_data[eid] > 0)
            out_idx[i++] = off + j;
        }
        CHECK_EQ(i, num_picks);  // correctness check
      } else {
        FloatArray prob_selected = DoubleSlice<IdxType, FloatType>(prob, data, off, len);
        RandomEngine::ThreadLocal()->Choice<IdxType, FloatType>(
            num_picks, prob_selected, out_idx, replace);
        for (int64_t j = 0; j < num_picks; ++j)
          out_idx[j] += off;
      }
    };
  return pick_fn;
}

template <typename IdxType, typename FloatType>
inline NumPicksFn<IdxType> GetSamplingBiasedNumPicksFn(
    int64_t num_samples, IdArray split, FloatArray bias, bool replace) {
  NumPicksFn<IdxType> num_picks_fn = [&]
    (IdxType rowid, IdxType off, IdxType len,
     const IdxType* col, const IdxType* data) {
      const FloatType* bias_data = bias.Ptr<FloatType>();
      const IdxType* tag_offset = split.Ptr<IdxType>() + rowid * split->shape[1];
      int64_t num_tags = bias->shape[0];
      IdxType num_possible_picks = 0;
      for (int64_t i = 0; i < num_tags; ++i) {
        if (bias_data[i] > 0)
          num_possible_picks += tag_offset[i + 1] - tag_offset[i];
      }

      if (num_samples == -1)
        return num_possible_picks;
      else if (replace)
        return static_cast<IdxType>((len == 0) ? 0 : num_samples);
      else
        return std::min(static_cast<IdxType>(num_samples), num_possible_picks);
    };
  return num_picks_fn;
}

template <typename IdxType, typename FloatType>
inline PickFn<IdxType> GetSamplingBiasedPickFn(
    int64_t num_samples, IdArray split, FloatArray bias, bool replace) {
  PickFn<IdxType> pick_fn = [&]
    (IdxType rowid, IdxType off, IdxType len, IdxType num_picks,
     const IdxType* col, const IdxType* data,
     IdxType* out_idx) {
    const FloatType* bias_data = bias.Ptr<FloatType>();
    const IdxType* tag_offset = split.Ptr<IdxType>() + rowid * split->shape[1];

    if (num_samples == -1 || (!replace && len == num_picks)) {
      // fast path for selecting all
      int64_t j = 0;
      for (int64_t i = 0; i < bias->shape[0]; ++i) {
        if (bias_data[i] > 0) {
          for (int64_t k = tag_offset[i]; k < tag_offset[i + 1]; ++k)
            out_idx[j++] = off + k;
        }
      }
      CHECK_EQ(j, num_picks);  // correctness check
    } else {
      RandomEngine::ThreadLocal()->BiasedChoice<IdxType, FloatType>(
              num_picks, tag_offset, bias, out_idx, replace);
      for (int64_t j = 0; j < num_picks; ++j) {
        out_idx[j] += off;
      }
    }
  };
  return pick_fn;
}

}  // namespace

/////////////////////////////// CSR ///////////////////////////////

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseSampling(CSRMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  CHECK(prob.defined());
  auto num_picks_fn = GetSamplingNumPicksFn<IdxType, FloatType>(num_samples, prob, replace);
  auto pick_fn = GetSamplingPickFn<IdxType, FloatType>(num_samples, prob, replace);
  return CSRRowWisePick(mat, rows, num_samples, pick_fn, num_picks_fn);
}

template COOMatrix CSRRowWiseSampling<kDGLCPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

template <DGLDeviceType XPU, typename IdxType>
COOMatrix CSRRowWiseSamplingUniform(CSRMatrix mat, IdArray rows,
                                    int64_t num_samples, bool replace) {
  auto num_picks_fn = GetSamplingUniformNumPicksFn<IdxType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return CSRRowWisePick(mat, rows, num_samples, pick_fn, num_picks_fn);
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseSamplingBiased(
    CSRMatrix mat,
    IdArray rows,
    int64_t num_samples,
    NDArray tag_offset,
    FloatArray bias,
    bool replace
) {
  auto num_picks_fn = GetSamplingBiasedNumPicksFn<IdxType, FloatType>(
      num_samples, tag_offset, bias, replace);
  auto pick_fn = GetSamplingBiasedPickFn<IdxType, FloatType>(
      num_samples, tag_offset, bias, replace);
  return CSRRowWisePick(mat, rows, num_samples, pick_fn, num_picks_fn);
}

template COOMatrix CSRRowWiseSamplingBiased<kDGLCPU, int32_t, float>(
  CSRMatrix, IdArray, int64_t, NDArray, FloatArray, bool);

template COOMatrix CSRRowWiseSamplingBiased<kDGLCPU, int64_t, float>(
  CSRMatrix, IdArray, int64_t, NDArray, FloatArray, bool);

template COOMatrix CSRRowWiseSamplingBiased<kDGLCPU, int32_t, double>(
  CSRMatrix, IdArray, int64_t, NDArray, FloatArray, bool);

template COOMatrix CSRRowWiseSamplingBiased<kDGLCPU, int64_t, double>(
  CSRMatrix, IdArray, int64_t, NDArray, FloatArray, bool);


/////////////////////////////// COO ///////////////////////////////

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix COORowWiseSampling(COOMatrix mat, IdArray rows, int64_t num_samples,
                             FloatArray prob, bool replace) {
  CHECK(prob.defined());
  auto num_picks_fn = GetSamplingNumPicksFn<IdxType, FloatType>(num_samples, prob, replace);
  auto pick_fn = GetSamplingPickFn<IdxType, FloatType>(num_samples, prob, replace);
  return COORowWisePick(mat, rows, num_samples, pick_fn, num_picks_fn);
}

template COOMatrix COORowWiseSampling<kDGLCPU, int32_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int64_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int32_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int64_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, bool);

template <DGLDeviceType XPU, typename IdxType>
COOMatrix COORowWiseSamplingUniform(COOMatrix mat, IdArray rows,
                                    int64_t num_samples, bool replace) {
  auto num_picks_fn = GetSamplingUniformNumPicksFn<IdxType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return COORowWisePick(mat, rows, num_samples, pick_fn, num_picks_fn);
}

template COOMatrix COORowWiseSamplingUniform<kDGLCPU, int32_t>(
    COOMatrix, IdArray, int64_t, bool);
template COOMatrix COORowWiseSamplingUniform<kDGLCPU, int64_t>(
    COOMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
