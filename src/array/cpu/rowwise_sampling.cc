/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/rowwise_sampling.cc
 * @brief rowwise sampling
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
inline FloatArray DoubleSlice(
    FloatArray array, const IdxType* idx_data, IdxType off, IdxType len) {
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

template <typename IdxType, typename DType>
inline NumPicksFn<IdxType> GetSamplingNumPicksFn(
    int64_t num_samples, NDArray prob_or_mask, bool replace) {
  NumPicksFn<IdxType> num_picks_fn = [prob_or_mask, num_samples, replace](
                                         IdxType rowid, IdxType off,
                                         IdxType len, const IdxType* col,
                                         const IdxType* data) {
    const int64_t max_num_picks = (num_samples == -1) ? len : num_samples;
    const DType* prob_or_mask_data = prob_or_mask.Ptr<DType>();
    IdxType nnz = 0;
    for (IdxType i = off; i < off + len; ++i) {
      const IdxType eid = data ? data[i] : i;
      if (prob_or_mask_data[eid] > 0) {
        ++nnz;
      }
    }

    if (replace) {
      return static_cast<IdxType>(nnz == 0 ? 0 : max_num_picks);
    } else {
      return std::min(static_cast<IdxType>(max_num_picks), nnz);
    }
  };
  return num_picks_fn;
}

template <typename IdxType, typename DType>
inline PickFn<IdxType> GetSamplingPickFn(
    int64_t num_samples, NDArray prob_or_mask, bool replace) {
  PickFn<IdxType> pick_fn = [prob_or_mask, num_samples, replace](
                                IdxType rowid, IdxType off, IdxType len,
                                IdxType num_picks, const IdxType* col,
                                const IdxType* data, IdxType* out_idx) {
    NDArray prob_or_mask_selected =
        DoubleSlice<IdxType, DType>(prob_or_mask, data, off, len);
    RandomEngine::ThreadLocal()->Choice<IdxType, DType>(
        num_picks, prob_or_mask_selected, out_idx, replace);
    for (int64_t j = 0; j < num_picks; ++j) {
      out_idx[j] += off;
    }
  };
  return pick_fn;
}

template <typename IdxType, typename FloatType>
inline EtypeRangePickFn<IdxType> GetSamplingRangePickFn(
    const std::vector<int64_t>& num_samples,
    const std::vector<FloatArray>& prob, bool replace) {
  EtypeRangePickFn<IdxType> pick_fn =
      [prob, num_samples, replace](
          IdxType off, IdxType et_offset, IdxType cur_et, IdxType et_len,
          const std::vector<IdxType>& et_idx,
          const std::vector<IdxType>& et_eid, const IdxType* eid,
          IdxType* out_idx) {
        const FloatArray& p = prob[cur_et];
        const FloatType* p_data = IsNullArray(p) ? nullptr : p.Ptr<FloatType>();
        FloatArray probs = FloatArray::Empty({et_len}, p->dtype, p->ctx);
        FloatType* probs_data = probs.Ptr<FloatType>();
        for (int64_t j = 0; j < et_len; ++j) {
          const IdxType cur_eid = et_eid[et_idx[et_offset + j]];
          probs_data[j] = p_data ? p_data[cur_eid] : static_cast<FloatType>(1.);
        }

        RandomEngine::ThreadLocal()->Choice<IdxType, FloatType>(
            num_samples[cur_et], probs, out_idx, replace);
      };
  return pick_fn;
}

template <typename IdxType>
inline NumPicksFn<IdxType> GetSamplingUniformNumPicksFn(
    int64_t num_samples, bool replace) {
  NumPicksFn<IdxType> num_picks_fn = [num_samples, replace](
                                         IdxType rowid, IdxType off,
                                         IdxType len, const IdxType* col,
                                         const IdxType* data) {
    const int64_t max_num_picks = (num_samples == -1) ? len : num_samples;
    if (replace) {
      return static_cast<IdxType>(len == 0 ? 0 : max_num_picks);
    } else {
      return std::min(static_cast<IdxType>(max_num_picks), len);
    }
  };
  return num_picks_fn;
}

template <typename IdxType>
inline PickFn<IdxType> GetSamplingUniformPickFn(
    int64_t num_samples, bool replace) {
  PickFn<IdxType> pick_fn = [num_samples, replace](
                                IdxType rowid, IdxType off, IdxType len,
                                IdxType num_picks, const IdxType* col,
                                const IdxType* data, IdxType* out_idx) {
    RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
        num_picks, len, out_idx, replace);
    for (int64_t j = 0; j < num_picks; ++j) {
      out_idx[j] += off;
    }
  };
  return pick_fn;
}

template <typename IdxType>
inline EtypeRangePickFn<IdxType> GetSamplingUniformRangePickFn(
    const std::vector<int64_t>& num_samples, bool replace) {
  EtypeRangePickFn<IdxType> pick_fn =
      [num_samples, replace](
          IdxType off, IdxType et_offset, IdxType cur_et, IdxType et_len,
          const std::vector<IdxType>& et_idx,
          const std::vector<IdxType>& et_eid, const IdxType* data,
          IdxType* out_idx) {
        RandomEngine::ThreadLocal()->UniformChoice<IdxType>(
            num_samples[cur_et], et_len, out_idx, replace);
      };
  return pick_fn;
}

template <typename IdxType, typename FloatType>
inline NumPicksFn<IdxType> GetSamplingBiasedNumPicksFn(
    int64_t num_samples, IdArray split, FloatArray bias, bool replace) {
  NumPicksFn<IdxType> num_picks_fn = [num_samples, split, bias, replace](
                                         IdxType rowid, IdxType off,
                                         IdxType len, const IdxType* col,
                                         const IdxType* data) {
    const int64_t max_num_picks = (num_samples == -1) ? len : num_samples;
    const int64_t num_tags = split->shape[1] - 1;
    const IdxType* tag_offset = split.Ptr<IdxType>() + rowid * split->shape[1];
    const FloatType* bias_data = bias.Ptr<FloatType>();
    IdxType nnz = 0;
    for (int64_t j = 0; j < num_tags; ++j) {
      if (bias_data[j] > 0) {
        nnz += tag_offset[j + 1] - tag_offset[j];
      }
    }

    if (replace) {
      return static_cast<IdxType>(nnz == 0 ? 0 : max_num_picks);
    } else {
      return std::min(static_cast<IdxType>(max_num_picks), nnz);
    }
  };
  return num_picks_fn;
}

template <typename IdxType, typename FloatType>
inline PickFn<IdxType> GetSamplingBiasedPickFn(
    int64_t num_samples, IdArray split, FloatArray bias, bool replace) {
  PickFn<IdxType> pick_fn = [num_samples, split, bias, replace](
                                IdxType rowid, IdxType off, IdxType len,
                                IdxType num_picks, const IdxType* col,
                                const IdxType* data, IdxType* out_idx) {
    const IdxType* tag_offset = split.Ptr<IdxType>() + rowid * split->shape[1];
    RandomEngine::ThreadLocal()->BiasedChoice<IdxType, FloatType>(
        num_picks, tag_offset, bias, out_idx, replace);
    for (int64_t j = 0; j < num_picks; ++j) {
      out_idx[j] += off;
    }
  };
  return pick_fn;
}

}  // namespace

/////////////////////////////// CSR ///////////////////////////////

template <DGLDeviceType XPU, typename IdxType, typename DType>
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  CHECK(prob_or_mask.defined());
  auto num_picks_fn =
      GetSamplingNumPicksFn<IdxType, DType>(num_samples, prob_or_mask, replace);
  auto pick_fn =
      GetSamplingPickFn<IdxType, DType>(num_samples, prob_or_mask, replace);
  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn, num_picks_fn);
}

template COOMatrix CSRRowWiseSampling<kDGLCPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int32_t, int8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int64_t, int8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int32_t, uint8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix CSRRowWiseSampling<kDGLCPU, int64_t, uint8_t>(
    CSRMatrix, IdArray, int64_t, NDArray, bool);

template <
    DGLDeviceType XPU, typename IdxType, typename DType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdxType>* new_seed_nodes, int64_t num_samples,
    NDArray prob_or_mask, bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  CHECK(prob_or_mask.defined());
  auto num_picks_fn =
      GetSamplingNumPicksFn<IdxType, DType>(num_samples, prob_or_mask, replace);
  auto pick_fn =
      GetSamplingPickFn<IdxType, DType>(num_samples, prob_or_mask, replace);
  return CSRRowWisePickFused<IdxType, map_seed_nodes>(
      mat, rows, seed_mapping, new_seed_nodes, num_samples, replace, pick_fn,
      num_picks_fn);
}

template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, float, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, float, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, double, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, double, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, int8_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, int8_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, uint8_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, uint8_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);

template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, float, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, float, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, double, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, double, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, int8_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, int8_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int32_t, uint8_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, NDArray, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingFused<kDGLCPU, int64_t, uint8_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, NDArray, bool);

template <DGLDeviceType XPU, typename IdxType, typename DType>
COOMatrix CSRRowWisePerEtypeSampling(
    CSRMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples,
    const std::vector<NDArray>& prob_or_mask, bool replace,
    bool rowwise_etype_sorted) {
  CHECK(prob_or_mask.size() == num_samples.size())
      << "the number of probability tensors does not match the number of edge "
         "types.";
  for (auto& p : prob_or_mask) CHECK(p.defined());
  auto pick_fn = GetSamplingRangePickFn<IdxType, DType>(
      num_samples, prob_or_mask, replace);
  return CSRRowWisePerEtypePick<IdxType, DType>(
      mat, rows, eid2etype_offset, num_samples, replace, rowwise_etype_sorted,
      pick_fn, prob_or_mask);
}

template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, float>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, float>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, double>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, double>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, int8_t>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, int8_t>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, uint8_t>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, uint8_t>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool, bool);

template <DGLDeviceType XPU, typename IdxType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, int64_t num_samples, bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  auto num_picks_fn =
      GetSamplingUniformNumPicksFn<IdxType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn, num_picks_fn);
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCPU, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCPU, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

template <DGLDeviceType XPU, typename IdxType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingUniformFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdxType>* new_seed_nodes, int64_t num_samples, bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  auto num_picks_fn =
      GetSamplingUniformNumPicksFn<IdxType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return CSRRowWisePickFused<IdxType, map_seed_nodes>(
      mat, rows, seed_mapping, new_seed_nodes, num_samples, replace, pick_fn,
      num_picks_fn);
}

template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLCPU, int32_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLCPU, int64_t, true>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLCPU, int32_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int32_t>*, int64_t, bool);
template std::pair<CSRMatrix, IdArray>
CSRRowWiseSamplingUniformFused<kDGLCPU, int64_t, false>(
    CSRMatrix, IdArray, IdArray, std::vector<int64_t>*, int64_t, bool);

template <DGLDeviceType XPU, typename IdxType>
COOMatrix CSRRowWisePerEtypeSamplingUniform(
    CSRMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples, bool replace,
    bool rowwise_etype_sorted) {
  auto pick_fn = GetSamplingUniformRangePickFn<IdxType>(num_samples, replace);
  return CSRRowWisePerEtypePick<IdxType, float>(
      mat, rows, eid2etype_offset, num_samples, replace, rowwise_etype_sorted,
      pick_fn, {});
}

template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDGLCPU, int32_t>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDGLCPU, int64_t>(
    CSRMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, bool, bool);

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseSamplingBiased(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray tag_offset,
    FloatArray bias, bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  auto num_picks_fn = GetSamplingBiasedNumPicksFn<IdxType, FloatType>(
      num_samples, tag_offset, bias, replace);
  auto pick_fn = GetSamplingBiasedPickFn<IdxType, FloatType>(
      num_samples, tag_offset, bias, replace);
  return CSRRowWisePick(mat, rows, num_samples, replace, pick_fn, num_picks_fn);
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

template <DGLDeviceType XPU, typename IdxType, typename DType>
COOMatrix COORowWiseSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  CHECK(prob_or_mask.defined());
  auto num_picks_fn =
      GetSamplingNumPicksFn<IdxType, DType>(num_samples, prob_or_mask, replace);
  auto pick_fn =
      GetSamplingPickFn<IdxType, DType>(num_samples, prob_or_mask, replace);
  return COORowWisePick(mat, rows, num_samples, replace, pick_fn, num_picks_fn);
}

template COOMatrix COORowWiseSampling<kDGLCPU, int32_t, float>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int64_t, float>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int32_t, double>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int64_t, double>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int32_t, int8_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int64_t, int8_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int32_t, uint8_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);
template COOMatrix COORowWiseSampling<kDGLCPU, int64_t, uint8_t>(
    COOMatrix, IdArray, int64_t, NDArray, bool);

template <DGLDeviceType XPU, typename IdxType, typename DType>
COOMatrix COORowWisePerEtypeSampling(
    COOMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples,
    const std::vector<NDArray>& prob_or_mask, bool replace) {
  CHECK(prob_or_mask.size() == num_samples.size())
      << "the number of probability tensors do not match the number of edge "
         "types.";
  for (auto& p : prob_or_mask) CHECK(p.defined());
  auto pick_fn = GetSamplingRangePickFn<IdxType, DType>(
      num_samples, prob_or_mask, replace);
  return COORowWisePerEtypePick<IdxType, DType>(
      mat, rows, eid2etype_offset, num_samples, replace, pick_fn, prob_or_mask);
}

template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, float>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, float>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, double>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, double>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, int8_t>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, int8_t>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, uint8_t>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, uint8_t>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, const std::vector<NDArray>&, bool);

template <DGLDeviceType XPU, typename IdxType>
COOMatrix COORowWiseSamplingUniform(
    COOMatrix mat, IdArray rows, int64_t num_samples, bool replace) {
  // If num_samples is -1, select all neighbors without replacement.
  replace = (replace && num_samples != -1);
  auto num_picks_fn =
      GetSamplingUniformNumPicksFn<IdxType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return COORowWisePick(mat, rows, num_samples, replace, pick_fn, num_picks_fn);
}

template COOMatrix COORowWiseSamplingUniform<kDGLCPU, int32_t>(
    COOMatrix, IdArray, int64_t, bool);
template COOMatrix COORowWiseSamplingUniform<kDGLCPU, int64_t>(
    COOMatrix, IdArray, int64_t, bool);

template <DGLDeviceType XPU, typename IdxType>
COOMatrix COORowWisePerEtypeSamplingUniform(
    COOMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples, bool replace) {
  auto pick_fn = GetSamplingUniformRangePickFn<IdxType>(num_samples, replace);
  return COORowWisePerEtypePick<IdxType, float>(
      mat, rows, eid2etype_offset, num_samples, replace, pick_fn, {});
}

template COOMatrix COORowWisePerEtypeSamplingUniform<kDGLCPU, int32_t>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, bool);
template COOMatrix COORowWisePerEtypeSamplingUniform<kDGLCPU, int64_t>(
    COOMatrix, IdArray, const std::vector<int64_t>&,
    const std::vector<int64_t>&, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
