/**
 * @file src/array/ascend/coo_ops.cc
 * @brief Ascend NPU implementations of COO operators via CSR composition.
 *
 * Pattern: COO → CSR (COOToCSR, Ascend native) → CSR op (Ascend native) → COO (CSRToCOO, Ascend native).
 * Same composition strategy as csr_transpose.cc.
 */
#include <acl/acl_rt.h>

#include <dgl/array.h>
#include "../array_op.h"

namespace dgl {
namespace aten {
namespace impl {

#ifdef DGL_USE_ASCEND

namespace {

// Ensure the COO matrix carries an explicit data array. When the COO has no
// explicit data, Ascend COOToCSR (coo2csr.cc) renumbers the CSR data array in
// CSR-sorted order instead of carrying the original COO edge positions. This
// breaks eid semantics for downstream ops (COOGetRowDataAndIndices /
// COOSliceRows / COOSliceMatrix) whose results depend on eid == original COO
// position.
//
// Mirrors the CPU COOSort_/COOToCSR convention (coo_sort.cc:157-158,
// coo2csr.cc:184-188): when there is no data, materialize Range(0, nnz) as the
// data array *before* calling COOToCSR. With explicit data present,
// COOToCSR gathers it by the sort permutation, preserving original positions.
template <typename IdType>
void EnsureCOODataForCSR(COOMatrix* coo) {
  if (COOHasData(*coo)) {
    return;
  }
  const int64_t nnz = coo->row->shape[0];
  // Generate Range(0, nnz) on CPU then copy to NPU (aten::Range has no Ascend
  // kernel; this matches the pattern in coo2csr.cc:173-175 / :186-188).
  aclrtSynchronizeDevice();
  DGLContext cpu_ctx{kDGLCPU, 0};
  IdArray cpu_data = aten::Range(0, nnz, coo->row->dtype.bits, cpu_ctx);
  coo->data = cpu_data.CopyTo(coo->row->ctx);
}

}  // anonymous namespace

// =========================================================================
// COOSort_
// =========================================================================

template <typename IdType>
void COOSort_Ascend(COOMatrix* coo, bool sort_column) {
  if (sort_column) {
    // Full sort (row + column): fall back to CPU since CSRSort_ is not
    // yet Ascend-native. The COOToCSR step already sorts by row, but
    // column sorting within each row requires CSRSort_ which currently
    // has no Ascend kernel.
    DGLContext cpu_ctx{kDGLCPU, 0};
    aclrtSynchronizeDevice();
    COOMatrix coo_cpu{
        coo->num_rows, coo->num_cols,
        coo->row.CopyTo(cpu_ctx), coo->col.CopyTo(cpu_ctx),
        coo->data.CopyTo(cpu_ctx), coo->row_sorted, coo->col_sorted};
    COOSort_<kDGLCPU, IdType>(&coo_cpu, sort_column);
    coo->row = coo_cpu.row.CopyTo(coo->row->ctx);
    coo->col = coo_cpu.col.CopyTo(coo->row->ctx);
    coo->data = coo_cpu.data.CopyTo(coo->row->ctx);
    coo->row_sorted = coo_cpu.row_sorted;
    coo->col_sorted = coo_cpu.col_sorted;
    return;
  }

  // Row-only sort: COOToCSR sorts by row, CSRToCOO gives row-sorted COO.
  // Ensure the COO carries explicit data so that COOToCSR preserves the
  // original edge positions (Range(0, nnz)) through the sort permutation.
  EnsureCOODataForCSR<IdType>(coo);
  CSRMatrix csr = COOToCSR(*coo);
  COOMatrix sorted_coo = CSRToCOO(csr, false);
  coo->row = sorted_coo.row;
  coo->col = sorted_coo.col;
  coo->data = sorted_coo.data;
  coo->row_sorted = true;
  // col_sorted: preserved if rows were already sorted; otherwise unknown.
  coo->col_sorted = coo->col_sorted && coo->row_sorted;
}

template <>
void COOSort_<kDGLAscend, int32_t>(COOMatrix* coo, bool sort_column) {
  COOSort_Ascend<int32_t>(coo, sort_column);
}

template <>
void COOSort_<kDGLAscend, int64_t>(COOMatrix* coo, bool sort_column) {
  COOSort_Ascend<int64_t>(coo, sort_column);
}

// =========================================================================
// COOGetRowNNZ (scalar)
// =========================================================================

template <typename IdType>
int64_t COOGetRowNNZ_Ascend(COOMatrix coo, int64_t row) {
  CSRMatrix csr = COOToCSR(coo);
  return CSRGetRowNNZ<kDGLAscend, IdType>(csr, row);
}

template <>
int64_t COOGetRowNNZ<kDGLAscend, int32_t>(COOMatrix coo, int64_t row) {
  return COOGetRowNNZ_Ascend<int32_t>(coo, row);
}

template <>
int64_t COOGetRowNNZ<kDGLAscend, int64_t>(COOMatrix coo, int64_t row) {
  return COOGetRowNNZ_Ascend<int64_t>(coo, row);
}

// =========================================================================
// COOGetRowNNZ (array)
// =========================================================================

template <typename IdType>
NDArray COOGetRowNNZ_Ascend(COOMatrix coo, NDArray rows) {
  CSRMatrix csr = COOToCSR(coo);
  return CSRGetRowNNZ<kDGLAscend, IdType>(csr, rows);
}

template <>
NDArray COOGetRowNNZ<kDGLAscend, int32_t>(COOMatrix coo, NDArray rows) {
  return COOGetRowNNZ_Ascend<int32_t>(coo, rows);
}

template <>
NDArray COOGetRowNNZ<kDGLAscend, int64_t>(COOMatrix coo, NDArray rows) {
  return COOGetRowNNZ_Ascend<int64_t>(coo, rows);
}

// =========================================================================
// COOGetRowDataAndIndices
// =========================================================================

template <typename IdType>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices_Ascend(
    COOMatrix coo, int64_t row) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  NDArray col_indices = CSRGetRowColumnIndices<kDGLAscend, IdType>(csr, row);
  NDArray data = CSRGetRowData<kDGLAscend, IdType>(csr, row);
  return std::make_pair(data, col_indices);
}

template <>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLAscend, int32_t>(
    COOMatrix coo, int64_t row) {
  return COOGetRowDataAndIndices_Ascend<int32_t>(coo, row);
}

template <>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices<kDGLAscend, int64_t>(
    COOMatrix coo, int64_t row) {
  return COOGetRowDataAndIndices_Ascend<int64_t>(coo, row);
}

// =========================================================================
// COOSliceRows (start, end)
// =========================================================================

template <typename IdType>
COOMatrix COOSliceRows_Ascend(COOMatrix coo, int64_t start, int64_t end) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  CSRMatrix sliced = CSRSliceRows<kDGLAscend, IdType>(csr, start, end);
  return CSRToCOO(sliced, false);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int32_t>(COOMatrix coo, int64_t start, int64_t end) {
  return COOSliceRows_Ascend<int32_t>(coo, start, end);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int64_t>(COOMatrix coo, int64_t start, int64_t end) {
  return COOSliceRows_Ascend<int64_t>(coo, start, end);
}

// =========================================================================
// COOSliceRows (rows)
// =========================================================================

template <typename IdType>
COOMatrix COOSliceRows_Ascend(COOMatrix coo, NDArray rows) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  CSRMatrix sliced = CSRSliceRows<kDGLAscend, IdType>(csr, rows);
  return CSRToCOO(sliced, false);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int32_t>(COOMatrix coo, NDArray rows) {
  return COOSliceRows_Ascend<int32_t>(coo, rows);
}

template <>
COOMatrix COOSliceRows<kDGLAscend, int64_t>(COOMatrix coo, NDArray rows) {
  return COOSliceRows_Ascend<int64_t>(coo, rows);
}

// =========================================================================
// COOSliceMatrix
// =========================================================================

template <typename IdType>
COOMatrix COOSliceMatrix_Ascend(COOMatrix coo, NDArray rows, NDArray cols) {
  // Ensure explicit data so eid reflects original COO positions.
  EnsureCOODataForCSR<IdType>(&coo);
  CSRMatrix csr = COOToCSR(coo);
  CSRMatrix sliced = CSRSliceMatrix<kDGLAscend, IdType>(csr, rows, cols);
  return CSRToCOO(sliced, false);
}

template <>
COOMatrix COOSliceMatrix<kDGLAscend, int32_t>(COOMatrix coo, NDArray rows, NDArray cols) {
  return COOSliceMatrix_Ascend<int32_t>(coo, rows, cols);
}

template <>
COOMatrix COOSliceMatrix<kDGLAscend, int64_t>(COOMatrix coo, NDArray rows, NDArray cols) {
  return COOSliceMatrix_Ascend<int64_t>(coo, rows, cols);
}

#endif  // DGL_USE_ASCEND

}  // namespace impl
}  // namespace aten
}  // namespace dgl
