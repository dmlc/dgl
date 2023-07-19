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
 * \file array/cuda/labor_sampling.cc
 * \brief labor sampling
 */
#include "./labor_pick.h"

namespace dgl {
namespace aten {
namespace impl {

/////////////////////////////// CSR ///////////////////////////////

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  return CSRLaborPick<IdxType, FloatType>(
      mat, rows, num_samples, prob, importance_sampling, random_seed,
      seed2_contribution, NIDs);
}

template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLCPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLCPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLCPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
CSRLaborSampling<kDGLCPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);

/////////////////////////////// COO ///////////////////////////////

template <DGLDeviceType XPU, typename IdxType, typename FloatType>
std::pair<COOMatrix, FloatArray> COOLaborSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs) {
  return COOLaborPick<IdxType, FloatType>(
      mat, rows, num_samples, prob, importance_sampling, random_seed,
      seed2_contribution, NIDs);
}

template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLCPU, int32_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLCPU, int64_t, float>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLCPU, int32_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);
template std::pair<COOMatrix, FloatArray>
COOLaborSampling<kDGLCPU, int64_t, double>(
    COOMatrix, IdArray, int64_t, FloatArray, int, IdArray, float, IdArray);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
