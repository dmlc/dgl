/**
 *   Copyright (c) 2022, NVIDIA Corporation
 *   Copyright (c) 2022, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * @file array/cpu/labor_pick.h
 * @brief Template implementation for layerwise pick operators.
 */

#ifndef DGL_ARRAY_CPU_LABOR_PICK_H_
#define DGL_ARRAY_CPU_LABOR_PICK_H_

#include <dgl/array.h>
#include <dgl/random.h>
#include <dgl/runtime/parallel_for.h>
#include <dmlc/omp.h>
#include <tsl/robin_map.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "../../random/continuous_seed.h"

namespace dgl {
namespace aten {
namespace impl {

using dgl::random::continuous_seed;

template <typename K, typename V>
using map_t = tsl::robin_map<K, V>;
template <typename iterator>
auto& mutable_value_ref(iterator it) {
  return it.value();
}

constexpr double eps = 0.0001;

template <typename IdxType, typename FloatType>
auto compute_importance_sampling_probabilities(
    DGLContext ctx, DGLDataType dtype, const IdxType max_degree,
    const IdxType num_rows, const int importance_sampling, const bool weighted,
    const IdxType* rows_data, const IdxType* indptr, const FloatType* A,
    const IdxType* indices, const IdxType num_picks, const FloatType* ds,
    FloatType* cs) {
  constexpr FloatType ONE = 1;
  // ps stands for \pi in arXiv:2210.13339
  FloatArray ps_array = NDArray::Empty({max_degree + 1}, dtype, ctx);
  FloatType* ps = ps_array.Ptr<FloatType>();

  double prev_ex_nodes = max_degree * num_rows;

  map_t<IdxType, FloatType> hop_map, hop_map2;
  for (int iters = 0; iters < importance_sampling || importance_sampling < 0;
       iters++) {
    // NOTE(mfbalin) When the graph is unweighted, the first c values in
    // the first iteration can be computed in O(1) as k / d where k is fanout
    // and d is the degree.

    // If the graph is weighted, the first c values are computed in the inner
    // for loop instead. Therefore the importance_sampling argument should be
    // increased by one in the caller.

    // The later iterations will have correct c values so the if block will be
    // executed.

    if (!weighted || iters) {
      hop_map2.clear();
      for (int64_t i = 0; i < num_rows; ++i) {
        const FloatType c = cs[i];
        const IdxType rid = rows_data[i];
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
          const auto ct = c * (weighted && iters == 1 ? A[j] : 1);
          auto itb = hop_map2.emplace(indices[j], ct);
          if (!itb.second) {
            mutable_value_ref(itb.first) = std::max(ct, itb.first->second);
          }
        }
      }
      if (hop_map.empty())
        hop_map = std::move(hop_map2);
      else
        // Update the pi array according to Eq 18.
        for (auto it : hop_map2) hop_map[it.first] *= it.second;
    }

    // Compute c_s according to Equation (15), (17) is slower because sorting is
    // required.
    for (int64_t i = 0; i < num_rows; ++i) {
      const IdxType rid = rows_data[i];
      const auto d = indptr[rid + 1] - indptr[rid];
      if (d == 0) continue;

      const auto k = std::min(num_picks, d);

      if (hop_map.empty()) {  // weighted first iter, pi = A
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          ps[j - indptr[rid]] = A[j];
      } else {
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          ps[j - indptr[rid]] = hop_map[indices[j]];
      }

      // stands for RHS of Equation (22) in arXiv:2210.13339 after moving the
      // other terms without c_s to RHS.
      double var_target = ds[i] * ds[i] / k;
      if (weighted) {
        var_target -= ds[i] * ds[i] / d;
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          var_target += A[j] * A[j];
      }
      FloatType c = cs[i];
      // stands for left handside of Equation (22) in arXiv:2210.13339 after
      // moving the other terms without c_s to RHS.
      double var_1;
      // Compute c_s in Equation (22) via fixed-point iteration.
      do {
        var_1 = 0;
        if (weighted) {
          for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
            // The check for zero is necessary for numerical stability
            var_1 += A[j] > 0
                         ? A[j] * A[j] / std::min(ONE, c * ps[j - indptr[rid]])
                         : 0;
        } else {
          for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
            var_1 += ONE / std::min(ONE, c * ps[j - indptr[rid]]);
        }

        c *= var_1 / var_target;
      } while (std::min(var_1, var_target) / std::max(var_1, var_target) <
               1 - eps);

      cs[i] = c;
    }

    // Check convergence
    if (!weighted || iters) {
      double cur_ex_nodes = 0;
      for (auto it : hop_map) cur_ex_nodes += std::min((FloatType)1, it.second);
      if (cur_ex_nodes / prev_ex_nodes >= 1 - eps) break;
      prev_ex_nodes = cur_ex_nodes;
    }
  }

  return hop_map;
}

// Template for picking non-zero values row-wise.
template <typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborPick(
    CSRMatrix mat, IdArray rows, int64_t num_picks, FloatArray prob,
    int importance_sampling, IdArray random_seed_arr, float seed2_contribution,
    IdArray NIDs) {
  using namespace aten;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat) ? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const IdxType* nids = IsNullArray(NIDs) ? nullptr : NIDs.Ptr<IdxType>();
  const auto num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;

  const bool weighted = !IsNullArray(prob);
  // O(1) c computation not possible, so one more iteration is needed.
  if (importance_sampling >= 0) importance_sampling += weighted;
  // A stands for the same notation in arXiv:2210.13339, i.e. the edge weights.
  auto A_arr = prob;
  FloatType* A = A_arr.Ptr<FloatType>();
  constexpr FloatType ONE = 1;

  constexpr auto dtype = DGLDataTypeTraits<FloatType>::dtype;

  // cs stands for c_s in arXiv:2210.13339
  FloatArray cs_array = NDArray::Empty({num_rows}, dtype, ctx);
  FloatType* cs = cs_array.Ptr<FloatType>();
  // ds stands for A_{*s} in arXiv:2210.13339
  FloatArray ds_array = NDArray::Empty({num_rows}, dtype, ctx);
  FloatType* ds = ds_array.Ptr<FloatType>();

  IdxType max_degree = 1;
  IdxType hop_size = 0;
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const auto act_degree = indptr[rid + 1] - indptr[rid];
    max_degree = std::max(act_degree, max_degree);
    double d = weighted
                   ? std::accumulate(A + indptr[rid], A + indptr[rid + 1], 0.0)
                   : act_degree;
    // O(1) c computation, samples more than needed for weighted case, mentioned
    // in the sentence between (10) and (11) in arXiv:2210.13339
    cs[i] = num_picks / d;
    ds[i] = d;
    hop_size += act_degree;
  }

  map_t<IdxType, FloatType> hop_map;

  if (importance_sampling)
    hop_map = compute_importance_sampling_probabilities<IdxType, FloatType>(
        ctx, dtype, max_degree, num_rows, importance_sampling, weighted,
        rows_data, indptr, A, indices, (IdxType)num_picks, ds, cs);

  constexpr auto vidtype = DGLDataTypeTraits<IdxType>::dtype;

  IdArray picked_row = NDArray::Empty({hop_size}, vidtype, ctx);
  IdArray picked_col = NDArray::Empty({hop_size}, vidtype, ctx);
  IdArray picked_idx = NDArray::Empty({hop_size}, vidtype, ctx);
  FloatArray picked_imp = importance_sampling
                              ? NDArray::Empty({hop_size}, dtype, ctx)
                              : NullArray();
  IdxType* picked_rdata = picked_row.Ptr<IdxType>();
  IdxType* picked_cdata = picked_col.Ptr<IdxType>();
  IdxType* picked_idata = picked_idx.Ptr<IdxType>();
  FloatType* picked_imp_data = picked_imp.Ptr<FloatType>();

  const continuous_seed random_seed =
      IsNullArray(random_seed_arr)
          ? continuous_seed(RandomEngine::ThreadLocal()->RandInt(1000000000))
          : continuous_seed(random_seed_arr, seed2_contribution);

  // compute number of edges first and do sampling
  IdxType num_edges = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    const IdxType rid = rows_data[i];
    const auto c = cs[i];

    FloatType norm_inv_p = 0;
    const auto off = num_edges;
    for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
      const auto v = indices[j];
      const uint64_t t = nids ? nids[v] : v;  // t in the paper
      // rolled random number r_t is a function of the random_seed and t
      const auto rnd = random_seed.uniform(t);
      const auto w = (weighted ? A[j] : 1);
      // if hop_map is initialized, get ps from there, otherwise get it from the
      // alternative.
      const auto ps = std::min(
          ONE, importance_sampling - weighted ? c * hop_map[v] : c * w);
      if (rnd <= ps) {
        picked_rdata[num_edges] = rid;
        picked_cdata[num_edges] = v;
        picked_idata[num_edges] = data ? data[j] : j;
        if (importance_sampling) {
          const auto edge_weight = w / ps;
          norm_inv_p += edge_weight;
          picked_imp_data[num_edges] = edge_weight;
        }
        num_edges++;
      }
    }

    if (importance_sampling) {
      const auto norm_factor = (num_edges - off) / norm_inv_p;
      for (auto i = off; i < num_edges; i++)
        // so that fn.mean can be used
        picked_imp_data[i] *= norm_factor;
    }
  }

  picked_row = picked_row.CreateView({num_edges}, picked_row->dtype);
  picked_col = picked_col.CreateView({num_edges}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({num_edges}, picked_idx->dtype);
  if (importance_sampling)
    picked_imp = picked_imp.CreateView({num_edges}, picked_imp->dtype);

  return std::make_pair(
      COOMatrix(mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx),
      picked_imp);
}

// Template for picking non-zero values row-wise. The implementation first
// slices out the corresponding rows and then converts it to CSR format. It then
// performs row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> COOLaborPick(
    COOMatrix mat, IdArray rows, int64_t num_picks, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows =
      Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto&& picked_importances = CSRLaborPick<IdxType, FloatType>(
      csr, new_rows, num_picks, prob, importance_sampling, random_seed,
      seed2_contribution, NIDs);
  const auto& picked = picked_importances.first;
  const auto& importances = picked_importances.second;
  return std::make_pair(
      COOMatrix(
          mat.num_rows, mat.num_cols,
          IndexSelect(
              rows, picked.row),  // map the row index to the correct one
          picked.col, picked.data),
      importances);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_LABOR_PICK_H_
