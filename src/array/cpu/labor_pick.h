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

// Template for picking non-zero values first layerwise than row-wise.
template <typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborPick(CSRMatrix mat, IdArray NIDs, IdArray rows,
                         int64_t __num_picks, FloatArray prob, IdArray random_seed, IdArray cnt, int importance_sampling) {
  using namespace aten;
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);
  const IdxType* data = CSRHasData(mat)? static_cast<IdxType*>(mat.data->data) : nullptr;
  const IdxType* rows_data = static_cast<IdxType*>(rows->data);
  const IdxType* nids = NIDs->shape[0] ? static_cast<IdxType*>(NIDs->data) : nullptr;
  const auto num_rows = rows->shape[0];
  const IdxType num_picks = __num_picks;
  const auto& ctx = mat.indptr->ctx;

  constexpr bool linear_c_convergence = false;

  const bool weights = !IsNullArray(prob);
  importance_sampling *= !weights;
  auto A_arr = prob;
  FloatType *A = static_cast<FloatType*>(A_arr->data);
  constexpr double eps = 0.0001;
  constexpr FloatType ONE = 1;

  phmap::flat_hash_map<IdxType, FloatType> hop_map;

  FloatArray cs_array = NDArray::Empty({num_rows},
                                        DGLDataType{kDGLFloat, 8*sizeof(FloatType), 1},
                                        ctx);
  FloatType* cs = static_cast<FloatType*>(cs_array->data);
  FloatArray ds_array = NDArray::Empty({num_rows},
                                        DGLDataType{kDGLFloat, 8*sizeof(FloatType), 1},
                                        ctx);
  FloatType* ds = static_cast<FloatType*>(ds_array->data);
  
  IdxType max_d = 1;
  IdxType hop_size = 0;
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const auto act_d = indptr[rid + 1] - indptr[rid];
    max_d = std::max(act_d, max_d);
    double d = weights ? std::accumulate(A + indptr[rid], A + indptr[rid + 1], 0.0) : act_d;
    cs[i] = num_picks / d;
    ds[i] = d;
    hop_size += act_d;
  }

  if (importance_sampling) {
    FloatArray ps_array = NDArray::Empty({max_d + 1},
                                          DGLDataType{kDGLFloat, 8*sizeof(FloatType), 1},
                                          ctx);
    FloatType* ps = static_cast<FloatType*>(ps_array->data);

    double prev_ex_nodes = max_d * num_rows;

    phmap::flat_hash_map<IdxType, FloatType> hop_map2;
    for (int iters = 0; iters < importance_sampling || importance_sampling < 0; iters++) {
      hop_map2.clear();
      for (int64_t i = 0; i < num_rows; ++i) {
        const FloatType c = cs[i];
        const IdxType rid = rows_data[i];
        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
          const auto ct = weights ? c * A[j] : c;
          auto [it, b] = hop_map2.emplace(indices[j], ct);
          if (!b)
            it->second = std::max(ct, it->second);
        }
      }

      if (hop_map.empty())
        hop_map = hop_map2;
      else
        for (auto it: hop_map2)
          hop_map[it.first] *= it.second;
      
      for (int64_t i = 0; i < num_rows; ++i) {
        const IdxType rid = rows_data[i];
        const auto d = indptr[rid + 1] - indptr[rid];
        if (d == 0)
          continue;
        
        const auto k = std::min(num_picks, d);

        for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
          ps[j - indptr[rid]] = hop_map[indices[j]];

        double var_target = ds[i] * ds[i] / k;
        if (weights) {
          var_target -= ds[i] * ds[i] / d;
          for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
            var_target += A[j] * A[j];
        }
        FloatType c = cs[i];
        double var_1;
        if (linear_c_convergence) {
          // fix weights case
          std::make_heap(ps, ps + d);
          double r = 0;
          if (weights)
            for (IdxType i = 0; i < d; i++) {
              const auto w = A[i + indptr[rid]];
              r += w / ps[i];
            }
          else
            for (IdxType i = 0; i < d; i++)
              r += 1 / ps[i];
          c = r / var_target;
          if (weights) {
            double v = A[d - 1] * A[d - 1];
            for (auto i = d - 1; i > 0 && c * A[i + indptr[rid]] * ps[i] >= 1; i--) {
              // fix this
              r -= ps[i];
              c = r / (var_target - v);
              v += A[i - 1 + indptr[rid]];
            }
          }
          else {
            for (auto i = d - 1; i > 0 && c * ps[i] >= 1; i--) {
              std::pop_heap(ps, ps + i + 1);
              r -= ps[i];
              c = r / (var_target - (d - i));
            }
          }
        }
        else do {
          var_1 = 0;
          if (weights)
            for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
              const auto ct = c * A[j];
              var_1 += A[j] * A[j] / std::min(ONE, ct * ps[j - indptr[rid]]);
            }
          else
            for (auto j = indptr[rid]; j < indptr[rid + 1]; j++)
              var_1 += ONE / std::min(ONE, c * ps[j - indptr[rid]]);

          c *= var_1 / var_target;
        } while (std::min(var_1, var_target) / std::max(var_1, var_target) < 1 - eps);
        
        cs[i] = c;
      }

      double cur_ex_nodes = 0;
      for (auto it: hop_map)
        cur_ex_nodes += std::min((FloatType)1, it.second);
      if (cur_ex_nodes / prev_ex_nodes >= 1 - eps)
        break;
      prev_ex_nodes = cur_ex_nodes;
    }
  }

  // When run distributed, random_seed needs to be equivalent.
  auto seeds = static_cast<const int64_t *>(random_seed->data);
  const auto num_seeds = random_seed->shape[0];
  auto cnts = static_cast<const int64_t *>(cnt->data);
  const auto l = (cnts[0] % cnts[1]) * 1.0 / cnts[1];
  const auto pi = std::acos(-1.0);
  const auto a0 = std::cos(pi * l / 2);
  const auto a1 = std::sin(pi * l / 2);

  const pcg32 ng0(seeds[0]), ng1(seeds[num_seeds - 1]);
  std::uniform_real_distribution<FloatType> uni;
  std::normal_distribution<FloatType> norm;

  // compute number of edges first and store randoms
  IdxType num_edges = 0;
  phmap::flat_hash_map<IdxType, FloatType> rand_map;
  auto rands_arr = NDArray::Empty({hop_size},
                                  DGLDataType{kDGLFloat, 8*sizeof(FloatType), 1},
                                  ctx);
  FloatType* rands = static_cast<FloatType*>(rands_arr->data);
  for (int64_t i = 0, off = 0; i < num_rows; i++) {
    const IdxType rid = rows_data[i];
    const auto c = cs[i];
    for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
      const auto v = indices[j];
      const auto u = nids ? nids[v] : v;
      auto [it, b] = rand_map.emplace(u, 0);
      if (b) {
        auto ng = ng0;
        ng.discard(u);
        if (num_seeds > 1) {
          norm.reset();
          auto rnd = a0 * norm(ng);
          ng = ng1;
          ng.discard(u);
          norm.reset();
          rnd += a1 * norm(ng);
          it->second = std::erfc(-rnd * (FloatType)M_SQRT1_2) / 2;
        }
        else {
          uni.reset();
          it->second = uni(ng);
        }
      }
      const auto rnd = it->second;
      const auto ct = weights ? c * A[j] : c;
      num_edges += rnd <= (importance_sampling ? ct * hop_map[v] : ct);
      rands[off++] = rnd;
    }
  }

  IdArray picked_row = NDArray::Empty({num_edges},
                                      DGLDataType{kDGLInt, 8*sizeof(IdxType), 1},
                                      ctx);
  IdArray picked_col = NDArray::Empty({num_edges},
                                      DGLDataType{kDGLInt, 8*sizeof(IdxType), 1},
                                      ctx);
  IdArray picked_idx = NDArray::Empty({num_edges},
                                      DGLDataType{kDGLInt, 8*sizeof(IdxType), 1},
                                      ctx);
  FloatArray importances = NDArray::Empty({importance_sampling ? num_edges : 1},
                                          DGLDataType{kDGLFloat, 8*sizeof(FloatType), 1},
                                          ctx);
  IdxType* picked_rdata = static_cast<IdxType*>(picked_row->data);
  IdxType* picked_cdata = static_cast<IdxType*>(picked_col->data);
  IdxType* picked_idata = static_cast<IdxType*>(picked_idx->data);
  FloatType* importances_data = static_cast<FloatType*>(importances->data);
  
  std::size_t off = 0, idx = 0;
  
  for (int64_t i = 0; i < num_rows; ++i) {
    const IdxType rid = rows_data[i];
    const auto d = indptr[rid + 1] - indptr[rid];
    if (d == 0)
      continue;
    const auto c = cs[i];

    FloatType norm_inv_p = 0;
    const auto off_start = off;
    for (auto j = indptr[rid]; j < indptr[rid + 1]; j++) {
      const auto v = indices[j];
      const auto w = (weights ? A[j] : 1);
      const auto ct = c * w;
      const auto ps = std::min(ONE, importance_sampling ? ct * hop_map[v] : ct);
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
    
    if (importance_sampling)
      for (auto i = off_start; i < off; i++)
        // so that fn.mean can be used
        importances_data[i] *= (off - off_start) / norm_inv_p;
  }

  assert((IdxType)off == num_edges);

  return std::make_pair(COOMatrix(mat.num_rows, mat.num_cols,
                   picked_row, picked_col, picked_idx), importances);
}

// Template for picking non-zero values row-wise. The implementation first slices
// out the corresponding rows and then converts it to CSR format. It then performs
// row-wise pick on the CSR matrix and rectifies the returned results.
template <typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> COOLaborPick(COOMatrix mat, IdArray NIDs, IdArray rows,
                          int64_t num_picks, FloatArray prob, IdArray random_seed, IdArray cnt, int importance_sampling) {
  using namespace aten;
  const auto& csr = COOToCSR(COOSliceRows(mat, rows));
  const IdArray new_rows = Range(0, rows->shape[0], rows->dtype.bits, rows->ctx);
  const auto& [picked, importances] = CSRLaborPick<IdxType, FloatType>(csr, NIDs, new_rows, num_picks, prob, random_seed, cnt, importance_sampling);
  return std::make_pair(COOMatrix(mat.num_rows, mat.num_cols,
                   IndexSelect(rows, picked.row),  // map the row index to the correct one
                   picked.col,
                   picked.data), importances);
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_H_
