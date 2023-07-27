/**
 *  Copyright (c) 2019-2022 by Contributors
 * @file array/array_op.h
 * @brief Array operator templates
 */
#ifndef DGL_ARRAY_ARRAY_OP_H_
#define DGL_ARRAY_ARRAY_OP_H_

#include <dgl/array.h>
#include <dgl/graph_traversal.h>

#include <tuple>
#include <utility>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
IdArray Full(IdType val, int64_t length, DGLContext ctx);

template <DGLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DGLContext ctx);

template <DGLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray array);

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index);

template <DGLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index);

template <DGLDeviceType XPU, typename DType>
IdArray NonZero(BoolArray bool_arr);

template <DGLDeviceType XPU, typename IdType>
IdArray NonZero(NDArray array);

template <DGLDeviceType XPU, typename DType>
std::pair<IdArray, IdArray> Sort(IdArray array, int num_bits);

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray Scatter(NDArray array, IdArray indices);

template <DGLDeviceType XPU, typename DType, typename IdType>
void Scatter_(IdArray index, NDArray value, NDArray out);

template <DGLDeviceType XPU, typename DType, typename IdType>
NDArray Repeat(NDArray array, IdArray repeats);

template <DGLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays);

template <DGLDeviceType XPU, typename IdType>
NDArray Concat(const std::vector<IdArray>& arrays);

template <DGLDeviceType XPU, typename DType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, DType pad_value);

template <DGLDeviceType XPU, typename DType, typename IdType>
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths);

template <DGLDeviceType XPU, typename IdType>
IdArray CumSum(IdArray array, bool prepend_zero);

// sparse arrays

template <DGLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray CSRIsNonZero(
    CSRMatrix csr, runtime::NDArray row, runtime::NDArray col);

template <DGLDeviceType XPU, typename IdType>
bool CSRHasDuplicate(CSRMatrix csr);

template <DGLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowNNZ(CSRMatrix csr, runtime::NDArray row);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowData(CSRMatrix csr, int64_t row);

template <DGLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr);

template <DGLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols,
    bool return_eids, runtime::NDArray weights, DType filler);

template <DGLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols,
    runtime::NDArray weights, DType filler) {
  return CSRGetData<XPU, IdType, DType>(
      csr, rows, cols, false, weights, filler);
}

template <DGLDeviceType XPU, typename IdType>
NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols) {
  return CSRGetData<XPU, IdType, IdType>(
      csr, rows, cols, true, NullArray(rows->dtype), -1);
}

template <DGLDeviceType XPU, typename IdType>
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr);

// Convert CSR to COO
template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr);

// Convert CSR to COO using data array as order
template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, runtime::NDArray rows);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DGLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix* csr);

template <DGLDeviceType XPU, typename IdType, typename TagType>
std::pair<CSRMatrix, NDArray> CSRSortByTag(
    const CSRMatrix& csr, IdArray tag_array, int64_t num_tags);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRReorder(
    CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOReorder(
    COOMatrix coo, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries);

template <DGLDeviceType XPU, typename IdType, typename FloatType>
std::pair<COOMatrix, FloatArray> CSRLaborSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs);

// FloatType is the type of probability data.
template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace);

// FloatType is the type of probability data.
template <
    DGLDeviceType XPU, typename IdxType, typename DType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdxType>* new_seed_nodes, int64_t num_samples,
    NDArray prob_or_mask, bool replace);

// FloatType is the type of probability data.
template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRRowWisePerEtypeSampling(
    CSRMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples,
    const std::vector<NDArray>& prob_or_mask, bool replace,
    bool rowwise_etype_sorted);

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, int64_t num_samples, bool replace);

template <DGLDeviceType XPU, typename IdType, bool map_seed_nodes>
std::pair<CSRMatrix, IdArray> CSRRowWiseSamplingUniformFused(
    CSRMatrix mat, IdArray rows, IdArray seed_mapping,
    std::vector<IdType>* new_seed_nodes, int64_t num_samples, bool replace);

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWisePerEtypeSamplingUniform(
    CSRMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples, bool replace,
    bool rowwise_etype_sorted);

// FloatType is the type of weight data.
template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, NDArray weight, bool ascending);

template <DGLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix CSRRowWiseSamplingBiased(
    CSRMatrix mat, IdArray rows, int64_t num_samples, NDArray tag_offset,
    FloatArray bias, bool replace);

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling(
    const CSRMatrix& csr, int64_t num_samples, int num_trials,
    bool exclude_self_loops, bool replace, double redundancy);

// Union CSRMatrixes
template <DGLDeviceType XPU, typename IdType>
CSRMatrix UnionCsr(const std::vector<CSRMatrix>& csrs);

template <DGLDeviceType XPU, typename IdType>
std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple(CSRMatrix csr);

////////////////////////////////////////////////////////////////////////////////

template <DGLDeviceType XPU, typename IdType>
bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray COOIsNonZero(
    COOMatrix coo, runtime::NDArray row, runtime::NDArray col);

template <DGLDeviceType XPU, typename IdType>
bool COOHasDuplicate(COOMatrix coo);

template <DGLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray COOGetRowNNZ(COOMatrix coo, runtime::NDArray row);

template <DGLDeviceType XPU, typename IdType>
std::pair<runtime::NDArray, runtime::NDArray> COOGetRowDataAndIndices(
    COOMatrix coo, int64_t row);

template <DGLDeviceType XPU, typename IdType>
std::vector<runtime::NDArray> COOGetDataAndIndices(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

template <DGLDeviceType XPU, typename IdType>
runtime::NDArray COOGetData(
    COOMatrix mat, runtime::NDArray rows, runtime::NDArray cols);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOTranspose(COOMatrix coo);

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, runtime::NDArray rows);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOSliceMatrix(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

template <DGLDeviceType XPU, typename IdType>
std::pair<COOMatrix, IdArray> COOCoalesce(COOMatrix coo);

template <DGLDeviceType XPU, typename IdType>
COOMatrix DisjointUnionCoo(const std::vector<COOMatrix>& coos);

template <DGLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* mat, bool sort_column);

template <DGLDeviceType XPU, typename IdType>
std::pair<bool, bool> COOIsSorted(COOMatrix coo);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COORemove(COOMatrix coo, IdArray entries);

template <DGLDeviceType XPU, typename IdType, typename FloatType>
std::pair<COOMatrix, FloatArray> COOLaborSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob,
    int importance_sampling, IdArray random_seed, float seed2_contribution,
    IdArray NIDs);

// FloatType is the type of probability data.
template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COORowWiseSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, NDArray prob_or_mask,
    bool replace);

// FloatType is the type of probability data.
template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COORowWisePerEtypeSampling(
    COOMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples,
    const std::vector<NDArray>& prob_or_mask, bool replace);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COORowWiseSamplingUniform(
    COOMatrix mat, IdArray rows, int64_t num_samples, bool replace);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COORowWisePerEtypeSamplingUniform(
    COOMatrix mat, IdArray rows, const std::vector<int64_t>& eid2etype_offset,
    const std::vector<int64_t>& num_samples, bool replace);

// FloatType is the type of weight data.
template <DGLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix COORowWiseTopk(
    COOMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending);

///////////////////////// Graph Traverse routines //////////////////////////

template <DGLDeviceType XPU, typename IdType>
Frontiers BFSNodesFrontiers(const CSRMatrix& csr, IdArray source);

template <DGLDeviceType XPU, typename IdType>
Frontiers BFSEdgesFrontiers(const CSRMatrix& csr, IdArray source);

template <DGLDeviceType XPU, typename IdType>
Frontiers TopologicalNodesFrontiers(const CSRMatrix& csr);

template <DGLDeviceType XPU, typename IdType>
Frontiers DGLDFSEdges(const CSRMatrix& csr, IdArray source);

template <DGLDeviceType XPU, typename IdType>
Frontiers DGLDFSLabeledEdges(
    const CSRMatrix& csr, IdArray source, const bool has_reverse_edge,
    const bool has_nontree_edge, const bool return_labels);

template <DGLDeviceType XPU, typename IdType>
COOMatrix COOLineGraph(const COOMatrix& coo, bool backtracking);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_ARRAY_OP_H_
