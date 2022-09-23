/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/rowwise_etype_sampling.h
 * \brief rowwise sampling by edge type.
 */
#ifndef DGL_ARRAY_CPU_ROWWISE_ETYPE_SAMPLING_H_
#define DGL_ARRAY_CPU_ROWWISE_ETYPE_SAMPLING_H_

#include <dgl/array.h>
#include <dgl/random.h>
#include <algorithm>
#include <numeric>
#include "./rowwise_etype_pick.h"

namespace dgl {
namespace aten {
namespace impl {
namespace {

template <typename IdxType, typename EType>
inline ETypeNumPicksFn<IdxType, EType> GetSamplingUniformETypeNumPicksFn(
    const std::vector<int64_t>& num_samples, bool replace) {
  // num_samples is captured by value.  This may have performance impact but allows
  // passing in an rvalue (e.g. GetSamplingUniformETypeNumPicksFn({1,2,3}, false)).
  ETypeNumPicksFn<IdxType, EType> num_picks_fn = [=]
    (IdxType rid, IdxType off, IdxType len, const IdxType* col, const IdxType* data,
     EType et, IdxType et_off, IdxType et_len, const IdxType* et_idx) {
      if (num_samples[et] == -1)
        return et_len;
      else if (replace)
        return static_cast<IdxType>(et_len == 0 ? 0 : num_samples[et]);
      else
        return std::min(static_cast<IdxType>(num_samples[et]), et_len);
    };
  return num_picks_fn;
}

template <typename IdxType, typename EType>
inline ETypePickFn<IdxType, EType> GetSamplingUniformETypePickFn(
    const std::vector<int64_t>& num_samples, bool replace) {
  ETypePickFn<IdxType, EType> pick_fn = [=]
    (IdxType rid, IdxType off, IdxType len, IdxType num_picks,
     const IdxType* col, const IdxType* data,
     EType et, IdxType et_off, IdxType et_len, const IdxType* et_idx,
     IdxType* out_idx) {
      if (num_samples[et] == -1 || (!replace && et_len == num_picks)) {
        // fast path for selecting all
        for (int64_t j = 0; j < num_picks; ++j)
          out_idx[j] = et_idx[et_off + j];
      } else {
        RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
            num_picks, et_len, out_idx, replace);
        for (int64_t j = 0; j < num_picks; ++j)
          out_idx[j] = et_idx[out_idx[j] + et_off];
      }
    };
  return pick_fn;
}

template <typename IdxType, typename FloatType, typename EType>
inline ETypeNumPicksFn<IdxType, EType> GetSamplingETypeNumPicksFn(
    const std::vector<int64_t>& num_samples, FloatArray prob, bool replace) {
  ETypeNumPicksFn<IdxType, EType> num_picks_fn = [=]
    (IdxType rid, IdxType off, IdxType len, const IdxType* col, const IdxType* data,
     EType et, IdxType et_off, IdxType et_len, const IdxType* et_idx) {
      const FloatType* p_data = prob.Ptr<FloatType>();
      int64_t num_possible_picks = 0;
      for (int64_t j = 0; j < et_len; ++j) {
        const IdxType loc = et_idx[et_off + j];
        const IdxType eid = data ? data[loc] : loc;
        if (p_data[eid] > 0)
          ++num_possible_picks;
      }

      if (num_samples[et] == -1)
        return num_possible_picks;
      else if (replace)
        return (num_possible_picks == 0) ? 0 : num_samples[et];
      else
        return std::min(num_possible_picks, num_samples[et]);
    };
  return num_picks_fn;
}

template <typename IdxType, typename FloatType, typename EType>
inline ETypePickFn<IdxType, EType> GetSamplingETypePickFn(
    const std::vector<int64_t>& num_samples, FloatArray prob, bool replace) {
  ETypePickFn<IdxType, EType> pick_fn = [=]
    (IdxType rid, IdxType off, IdxType len, IdxType num_picks,
     const IdxType* col, const IdxType* data,
     EType et, IdxType et_off, IdxType et_len, const IdxType* et_idx,
     IdxType* out_idx) {
      const FloatType* p_data = prob.Ptr<FloatType>();

      if (num_samples[et] == -1 || (!replace && et_len == num_picks)) {
        // fast path for selecting all
        int64_t i = 0;
        for (int64_t j = 0; j < et_len; ++j) {
          const IdxType loc = et_idx[et_off + j];
          const IdxType eid = data ? data[loc] : loc;
          if (p_data[eid] > 0)
            out_idx[i++] = et_idx[et_off + j];
        }
        CHECK_EQ(i, num_picks);  // correctness check
      } else {
        FloatArray probs = FloatArray::Empty({et_len}, prob->dtype, prob->ctx);
        FloatType* probs_data = static_cast<FloatType*>(probs->data);
        for (int64_t j = 0; j < et_len; ++j) {
          const IdxType loc = et_idx[et_off + j];
          probs_data[j] = data ? p_data[data[loc]] : p_data[loc];
        }

        RandomEngine::ThreadLocal()->Choice<IdxType, FloatType>(
            num_picks, probs, out_idx, replace);
        for (int64_t i = 0; i < num_picks; ++i) {
          out_idx[i] = et_idx[out_idx[i] + et_off];
        }
      }
    };
  return pick_fn;
}

}  // namespace

template <DGLDeviceType XPU, typename IdxType, typename FloatType, typename EType>
COOMatrix CSRRowWisePerEtypeSampling(CSRMatrix mat, IdArray rows, IdArray etypes,
                                     const std::vector<int64_t>& num_samples,
                                     FloatArray prob, bool replace, bool etype_sorted) {
  CHECK(prob.defined());
  auto num_picks_fn = GetSamplingETypeNumPicksFn<IdxType, FloatType, EType>(
      num_samples, prob, replace);
  auto pick_fn = GetSamplingETypePickFn<IdxType, FloatType, EType>(
      num_samples, prob, replace);
  return CSRRowWisePerEtypePick<IdxType, EType>(
      mat, rows, etypes, num_samples, etype_sorted, pick_fn, num_picks_fn);
}

template <DGLDeviceType XPU, typename IdxType, typename EType>
COOMatrix CSRRowWisePerEtypeSamplingUniform(CSRMatrix mat, IdArray rows, IdArray etypes,
                                            const std::vector<int64_t>& num_samples,
                                            bool replace, bool etype_sorted) {
  auto num_picks_fn = GetSamplingUniformETypeNumPicksFn<IdxType, EType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformETypePickFn<IdxType, EType>(num_samples, replace);
  return CSRRowWisePerEtypePick<IdxType, EType>(
      mat, rows, etypes, num_samples, etype_sorted, pick_fn, num_picks_fn);
}

template <DGLDeviceType XPU, typename IdxType, typename FloatType, typename EType>
COOMatrix COORowWisePerEtypeSampling(COOMatrix mat, IdArray rows, IdArray etypes,
                                     const std::vector<int64_t>& num_samples,
                                     FloatArray prob, bool replace, bool etype_sorted) {
  CHECK(prob.defined());
  auto num_picks_fn = GetSamplingETypeNumPicksFn<IdxType, FloatType, EType>(
      num_samples, prob, replace);
  auto pick_fn = GetSamplingETypePickFn<IdxType, FloatType, EType>(
      num_samples, prob, replace);
  return COORowWisePerEtypePick<IdxType, EType>(
      mat, rows, etypes, num_samples, etype_sorted, pick_fn, num_picks_fn);
}

template <DGLDeviceType XPU, typename IdxType, typename EType>
COOMatrix COORowWisePerEtypeSamplingUniform(COOMatrix mat, IdArray rows, IdArray etypes,
                                    const std::vector<int64_t>& num_samples,
                                    bool replace, bool etype_sorted) {
  auto num_picks_fn = GetSamplingUniformETypeNumPicksFn<IdxType, EType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformETypePickFn<IdxType, EType>(num_samples, replace);
  return COORowWisePerEtypePick<IdxType, EType>(
      mat, rows, etypes, num_samples, etype_sorted, pick_fn, num_picks_fn);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_ETYPE_SAMPLING_H_
