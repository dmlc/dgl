/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/rowwise_etype_sampling.cc
 * \brief rowwise sampling by edge type.
 */
#include "rowwise_etype_sampling.h"

namespace dgl {
namespace aten {
namespace impl {

// (BarclayII) Ideally we want to have a code generator that generates these template
// instantiations with all the combination of IdxType, FloatType and EType.  This
// is not necessary for the current milestone (2022.9.30) though.
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int32_t, float, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int64_t, float, int64_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int32_t, double, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int64_t, double, int64_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int32_t, int8_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int64_t, int8_t, int64_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int32_t, uint8_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix CSRRowWisePerEtypeSampling<kDLCPU, int64_t, uint8_t, int64_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);

template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDLCPU, int32_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
template COOMatrix CSRRowWisePerEtypeSamplingUniform<kDLCPU, int64_t, int32_t>(
    CSRMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);

template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int32_t, float, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int64_t, float, int64_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int32_t, double, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int64_t, double, int64_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int32_t, int8_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int64_t, int8_t, int64_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int32_t, uint8_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);
template COOMatrix COORowWisePerEtypeSampling<kDLCPU, int64_t, uint8_t, int64_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, FloatArray, bool, bool);

template COOMatrix COORowWisePerEtypeSamplingUniform<kDLCPU, int32_t, int32_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
template COOMatrix COORowWisePerEtypeSamplingUniform<kDLCPU, int64_t, int64_t>(
    COOMatrix, IdArray, IdArray, const std::vector<int64_t>&, bool, bool);
}  // namespace impl
}  // namespace aten
}  // namespace dgl
