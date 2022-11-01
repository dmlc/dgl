/*!
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
 * \file array/cpu/labor_pick.h
 * \brief Template implementation for layerwise pick operators.
 */

#ifndef DGL_ARRAY_CPU_LABOR_PICK_H_
#define DGL_ARRAY_CPU_LABOR_PICK_H_

#include <dgl/random.h>
#include <dgl/array.h>
#include <dmlc/omp.h>
#include <dgl/runtime/parallel_for.h>
#include <parallel_hashmap/phmap.h>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <numeric>
#include <memory>
#include <utility>
#include <cmath>
#include <iostream>
#include <pcg_random.hpp>

namespace dgl {
namespace aten {
namespace impl {

constexpr double eps = 0.0001;

template <typename IdxType, typename FloatType>
void compute_importance_sampling_probabilities(
  DGLContext ctx,
  DGLDataType fidtype,
  const IdxType max_degree,
  const IdxType num_rows,
  const int importance_sampling,
  const bool weights,
  const IdxType* rows_data,
  const IdxType* indptr,
  const FloatType* A,
  const IdxType* indices,
  const IdxType num_picks,
  const FloatType* ds,
  FloatType* cs,
  phmap::flat_hash_map<IdxType, FloatType> &hop_map
) {
  constexpr FloatType ONE = 1;
  // ps stands for \pi in arXiv:2210.13339
  FloatArray ps_array = NDArray::Empty({max_degree + 1}, fidtype, ctx);
  FloatType* ps = ps_array.Ptr<FloatType>();

  double prev_ex_nodes = max_degree * num_rows;

  phmap::flat_hash_map<IdxType, FloatType> hop_map2;
  for (int iters = 0; iters < importance_sampling || importance_sampling < 0; iters++) {
    /*
    An implementation detail about when there are no weights, then the first c values can be computed in O(1) because all the edge weights are uniform. The c value is just k / d where k is fanout and d is the degree. But when there are weights, since the computed c values are incorrect, I have to skip some logic making use of it to compute the right c values below.

    The later iterations will have correct c values so they will take the if in that case.

    I also increase importance_sampling by 1 above so that the weighted case can do one more of this iteration to match the unweighted case.
    */
    if (!weights || iters) {
      hop_map2.clear();
      for (int64_t i = 0; i < num_rows; ++i) {
        const FloatType c = cs[i];
        const IdxType rid = rows_data[i];
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
          const auto ct = c * (weights && iters == 1 ? A[j] : 1);
          auto itb = hop_map2.emplace(indices[j], ct);
          if (!itb.second)
            itb.first->second = std::max(ct, itb.first->second);
        }
      }
      if (hop_map.empty())
        hop_map = std::move(hop_map2);
      else
        for (auto it : hop_map2)
          hop_map[it.first] *= it.second;
    }

    for (int64_t i = 0; i < num_rows; ++i) {
      const IdxType rid = rows_data[i];
      const auto d = indptr[rid + 1] - indptr[rid];
      if (d == 0)
        continue;

      const auto k = std::min(num_picks, d);

      if (hop_map.empty()) {  // weights first iter, pi = A
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          ps[j - indptr[rid]] = A[j];
      } else {
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          ps[j - indptr[rid]] = hop_map[indices[j]];
      }

      // stands for right handside of Equation (22) in arXiv:2210.13339
      double var_target = ds[i] * ds[i] / k;
      if (weights) {
        var_target -= ds[i] * ds[i] / d;
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          var_target += A[j] * A[j];
      }
      FloatType c = cs[i];
      // stands for left handside of Equation (22) in arXiv:2210.13339
      double var_1;
      do {
        var_1 = 0;
        if (weights) {
          for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
            var_1 += A[j] * A[j] / std::min(ONE, c * ps[j - indptr[rid]]);
        } else {
          for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
            var_1 += ONE / std::min(ONE, c * ps[j - indptr[rid]]);
        }

        c *= var_1 / var_target;
      } while (std::min(var_1, var_target) / std::max(var_1, var_target) < 1 - eps);

      cs[i] = c;
    }

    if (!weights || iters) {
      double cur_ex_nodes = 0;
      for (auto it : hop_map)
        cur_ex_nodes += std::min((FloatType)1, it.second);
      if (cur_ex_nodes / prev_ex_nodes >= 1 - eps)
        break;
      prev_ex_nodes = cur_ex_nodes;
    }
  }
}

// Template for picking non-zero values row-wise.
template <typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborPick(
    CSRMatrix mat,
    IdArray rows,
    int64_t num_picks,
    FloatArray prob,
    int importance_sampling,
    IdArray random_seed_arr,
    IdArray NIDs) {
  using namespace aten;
  const IdxType* indptr = mat.indptr.Ptr<IdxType>();
  const IdxType* indices = mat.indices.Ptr<IdxType>();
  const IdxType* data = CSRHasData(mat) ? mat.data.Ptr<IdxType>() : nullptr;
  const IdxType* rows_data = rows.Ptr<IdxType>();
  const IdxType* nids = IsNullArray(NIDs) ? nullptr : NIDs.Ptr<IdxType>();
  const auto num_rows = rows->shape[0];
  const auto& ctx = mat.indptr->ctx;

  const bool weights = !IsNullArray(prob);
  // O(1) c computation not possible, so one more iteration is needed.
  if (importance_sampling >= 0)
    importance_sampling += weights;
  // A stands for A in arXiv:2210.13339, that is edge weights
  auto A_arr = prob;
  FloatType *A = A_arr.Ptr<FloatType>();
  constexpr FloatType ONE = 1;

  constexpr auto fidtype = DGLDataTypeTraits<FloatType>::dtype;

  // cs stands for c_s in arXiv:2210.13339
  FloatArray cs_array = NDArray::Empty({num_rows}, fidtype, ctx);
  FloatType* cs = cs_array.Ptr<FloatType>();
  // ds stands for A_{*s} in arXiv:2210.13339
  FloatArray ds_array = NDArray::Empty({num_rows}, fidtype, ctx);
  FloatType* ds = ds_array.Ptr<FloatType>();

  IdxType max_degree = 1;
  IdxType hop_size = 0;
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const auto act_degree = indptr[rid + 1] - indptr[rid];
    max_degree = std::max(act_degree, max_degree);
    double d = weights ? std::accumulate(A + indptr[rid], A + indptr[rid + 1], 0.0) : act_degree;
    // O(1) c computation, samples more than needed for weighted case, mentioned in the sentence
    // between (10) and (11) in arXiv:2210.13339
    cs[i] = num_picks / d;
    ds[i] = d;
    hop_size += act_degree;
  }

  phmap::flat_hash_map<IdxType, FloatType> hop_map;

  if (importance_sampling)
    compute_importance_sampling_probabilities<IdxType, FloatType>(
        ctx,
        fidtype,
        max_degree,
        num_rows,
        importance_sampling,
        weights,
        rows_data,
        indptr,
        A,
        indices,
        (IdxType)num_picks,
        ds,
        cs,
        hop_map);

  const uint64_t random_seed = IsNullArray(random_seed_arr) ?
      RandomEngine::ThreadLocal()->RandInt(1000000000) : random_seed_arr.Ptr<int64_t>()[0];

  const pcg32 ng0(random_seed);
  std::uniform_real_distribution<FloatType> uni;

  // compute number of edges first and store randoms
  IdxType num_edges = 0;
  phmap::flat_hash_map<IdxType, FloatType> rand_map;
  FloatArray rands_arr = NDArray::Empty({hop_size}, fidtype, ctx);
  auto rands = rands_arr.Ptr<FloatType>();
  for (int64_t i = 0, off = 0; i < num_rows; i++) {
    const IdxType rid = rows_data[i];
    const auto c = cs[i];
    for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
      const auto v = indices[j];
      const auto u = nids ? nids[v] : v;
      // itb stands for a pair of iterator and boolean indicating if insertion was successful
      auto itb = rand_map.emplace(u, 0);
      if (itb.second) {
        auto ng = ng0;
        ng.discard(u);
        uni.reset();
        itb.first->second = uni(ng);
      }
      const auto rnd = itb.first->second;
      // if hop_map is initialized, get ps from there, otherwise get it from the alternative.
      num_edges +=
          rnd <= (importance_sampling - weights ? c * hop_map[v] : c * (weights ? A[j] : 1));
      rands[off++] = rnd;
    }
  }

  constexpr auto vidtype = DGLDataTypeTraits<IdxType>::dtype;

  IdArray picked_row = NDArray::Empty({num_edges}, vidtype, ctx);
  IdArray picked_col = NDArray::Empty({num_edges}, vidtype, ctx);
  IdArray picked_idx = NDArray::Empty({num_edges}, vidtype, ctx);
  FloatArray importances = NDArray::Empty({importance_sampling ? num_edges : 0}, fidtype, ctx);
  IdxType* picked_rdata = picked_row.Ptr<IdxType>();
  IdxType* picked_cdata = picked_col.Ptr<IdxType>();
  IdxType* picked_idata = picked_idx.Ptr<IdxType>();
  FloatType* importances_data = importances.Ptr<FloatType>();

  std::size_t off = 0, idx = 0;

  for (int64_t i = 0; i < num_rows; i++) {
    const IdxType rid = rows_data[i];
    const auto c = cs[i];

    FloatType norm_inv_p = 0;
    const auto off_start = off;
    for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
      const auto v = indices[j];
      const auto w = (weights ? A[j] : 1);
      const auto ps = std::min(ONE, importance_sampling - weights ? c * hop_map[v] : c * w);
      const auto rnd = rands[idx++];
      if (rnd <= ps) {
        norm_inv_p += w / ps;
        picked_rdata[off] = rid;
        picked_cdata[off] = indices[j];
        picked_idata[off] = data ? data[j] : j;
        if (importance_sampling)
          importances_data[off] = w / ps;
        off++;
      }
    }

    if (importance_sampling) {
      for (auto i = off_start; i < off; i++)
        // so that fn.mean can be used
        importances_data[i] *= (off - off_start) / norm_inv_p;
    }
  }

  CHECK((IdxType)off == num_edges) << "computed num_edges should match the number of edges sampled"
      << num_edges << " != " << off;

  return std::make_pair(COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx), importances);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> COOLaborPick(
    COOMatrix mat,
    IdArray rows,
    int64_t num_picks,
    FloatArray prob,
    int importance_sampling,
    IdArray random_seed,
    IdArray NIDs) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto&& picked_importances = CSRLaborPick<IdxType, FloatType>(
      csr, new_rows, num_picks, prob, importance_sampling, random_seed, NIDs);
  const auto& picked = picked_importances.first;
  const auto& importances = picked_importances.second;
  return std::make_pair(COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data), importances);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_LABOR_PICK_H_
