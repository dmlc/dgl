/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/rowwise_etype_sampling_int32.cc
 * \brief rowwise sampling by edge type.
 */
#include "rowwise_etype_sampling.h"

namespace dgl {
namespace aten {
namespace impl {

// (BarclayII) Ideally we want to have a code generator that generates these template
// instantiations with all the combination of IdxType, FloatType and EType.
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, float, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, float, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, double, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, double, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, int8_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, int8_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int32_t, uint8_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDGLCPU, int64_t, uint8_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);

template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDGLCPU, int32_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDGLCPU, int64_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);

template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, float, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, float, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, double, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, double, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, int8_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, int8_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int32_t, uint8_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDGLCPU, int64_t, uint8_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);

template COOMatrix COORowWisePerEtypeSamplingUniform<kDGLCPU, int32_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
template COOMatrix COORowWisePerEtypeSamplingUniform<kDGLCPU, int64_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
}  // namespace impl
}  // namespace aten
}  // namespace dgl
