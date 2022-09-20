/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/rowwise_etype_sampling.cc
 * \brief rowwise sampling by edge type.
 */
#include <dgl/random.h>
#include <numeric>
#include "./rowwise_pick.h"

namespace dgl {
namespace aten {
namespace impl {
namespace {

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWisePerEtypeSampling(CSRMatrix mat, IdArray rows, IdArray etypes,
                                     const std::vector<int64_t>& num_samples,
                                     FloatArray prob, bool replace, bool etype_sorted) {
  CHECK(prob.defined());
  auto pick_fn = GetSamplingRangePickFn<IdxType, FloatType>(num_samples, prob, replace);
  return CSRRowWisePerEtypePick(mat, rows, etypes, num_samples, replace, etype_sorted, pick_fn);
}

template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int32_t, float>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int64_t, float>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int32_t, double>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int64_t, double>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix COORowWisePerEtypeSampling(COOMatrix mat, IdArray rows, IdArray etypes,
                                     const std::vector<int64_t>& num_samples,
                                     FloatArray prob, bool replace, bool etype_sorted) {
  CHECK(prob.defined());
  auto pick_fn = GetSamplingRangePickFn<IdxType, FloatType>(num_samples, prob, replace);
  return COORowWisePerEtypePick(mat, rows, etypes, num_samples, replace, etype_sorted, pick_fn);
}

template <DLDeviceType XPU, typename IdxType>
COOMatrix CSRRowWisePerEtypeSamplingUniform(CSRMatrix mat, IdArray rows, IdArray etypes,
                                            const std::vector<int64_t>& num_samples,
                                            bool replace, bool etype_sorted) {
  auto pick_fn = GetSamplingUniformRangePickFn<IdxType>(num_samples, replace);
  return CSRRowWisePerEtypePick(mat, rows, etypes, num_samples, replace, etype_sorted, pick_fn);
}

template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDLCPU, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDLCPU, int64_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);

template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int32_t, float>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int64_t, float>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int32_t, double>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int64_t, double>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);

template <DLDeviceType XPU, typename IdxType>
COOMatrix COORowWiseSamplingUniform(COOMatrix mat, IdArray rows,
                                    int64_t num_samples, bool replace) {
  auto num_picks_fn = GetSamplingNumPicksFn<IdxType>(num_samples, replace);
  auto pick_fn = GetSamplingUniformPickFn<IdxType>(num_samples, replace);
  return COORowWisePick(mat, rows, num_samples, pick_fn, num_picks_fn);
}

template COOMatrix COORowWiseSamplingUniform<kDLCPU, int32_t>(
    COOMatrix, IdArray, int64_t, bool);
template COOMatrix COORowWiseSamplingUniform<kDLCPU, int64_t>(
    COOMatrix, IdArray, int64_t, bool);

template <DLDeviceType XPU, typename IdxType>
COOMatrix COORowWisePerEtypeSamplingUniform(COOMatrix mat, IdArray rows, IdArray etypes,
                                    const std::vector<int64_t>& num_samples,
                                    bool replace, bool etype_sorted) {
  auto pick_fn = GetSamplingUniformRangePickFn<IdxType>(num_samples, replace);
  return COORowWisePerEtypePick(mat, rows, etypes, num_samples, replace, etype_sorted, pick_fn);
}

template COOMatrix COORowWisePerEtypeSamplingUniform<kDLCPU, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
template COOMatrix COORowWisePerEtypeSamplingUniform<kDLCPU, int64_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);


}  // namespace impl
}  // namespace aten
}  // namespace dgl
