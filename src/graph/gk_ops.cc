/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/gk_ops.cc
 * @brief Graph operation implemented in GKlib
 */

#if !defined(_WIN32)
#include <GKlib.h>
#endif  // !defined(_WIN32)

#include <dgl/graph_op.h>

namespace dgl {

#if !defined(_WIN32)

/**
 * Convert DGL CSR to GKLib CSR.
 * GKLib CSR actually stores a CSR object and a CSC object of a graph.
 * @param mat the DGL CSR matrix.
 * @param is_row the input DGL matrix is CSR or CSC.
 * @return a GKLib CSR.
 */
gk_csr_t *Convert2GKCsr(const aten::CSRMatrix mat, bool is_row) {
  // TODO(zhengda) The conversion will be zero-copy in the future.
  CHECK_EQ(mat.indptr->dtype.bits, sizeof(dgl_id_t) * CHAR_BIT);
  CHECK_EQ(mat.indices->dtype.bits, sizeof(dgl_id_t) * CHAR_BIT);
  const dgl_id_t *indptr = static_cast<dgl_id_t *>(mat.indptr->data);
  const dgl_id_t *indices = static_cast<dgl_id_t *>(mat.indices->data);

  gk_csr_t *gk_csr = gk_csr_Create();
  gk_csr->nrows = mat.num_rows;
  gk_csr->ncols = mat.num_cols;
  uint64_t nnz = mat.indices->shape[0];
  auto gk_indptr = gk_csr->rowptr;
  auto gk_indices = gk_csr->rowind;
  size_t num_ptrs;
  if (is_row) {
    num_ptrs = gk_csr->nrows + 1;
    gk_indptr = gk_csr->rowptr = gk_zmalloc(
        gk_csr->nrows + 1,
        const_cast<char *>("gk_csr_ExtractPartition: rowptr"));
    gk_indices = gk_csr->rowind =
        gk_imalloc(nnz, const_cast<char *>("gk_csr_ExtractPartition: rowind"));
  } else {
    num_ptrs = gk_csr->ncols + 1;
    gk_indptr = gk_csr->colptr = gk_zmalloc(
        gk_csr->ncols + 1,
        const_cast<char *>("gk_csr_ExtractPartition: colptr"));
    gk_indices = gk_csr->colind =
        gk_imalloc(nnz, const_cast<char *>("gk_csr_ExtractPartition: colind"));
  }

  for (size_t i = 0; i < num_ptrs; i++) {
    gk_indptr[i] = indptr[i];
  }
  for (size_t i = 0; i < nnz; i++) {
    gk_indices[i] = indices[i];
  }
  return gk_csr;
}

/**
 * Convert GKLib CSR to DGL CSR.
 * GKLib CSR actually stores a CSR object and a CSC object of a graph.
 * @param gk_csr the GKLib CSR.
 * @param is_row specify whether to convert the CSR or CSC object of GKLib CSR.
 * @return a DGL CSR matrix.
 */
aten::CSRMatrix Convert2DGLCsr(gk_csr_t *gk_csr, bool is_row) {
  // TODO(zhengda) The conversion will be zero-copy in the future.
  size_t num_ptrs;
  size_t nnz;
  auto gk_indptr = gk_csr->rowptr;
  auto gk_indices = gk_csr->rowind;
  if (is_row) {
    num_ptrs = gk_csr->nrows + 1;
    nnz = gk_csr->rowptr[num_ptrs - 1];
    gk_indptr = gk_csr->rowptr;
    gk_indices = gk_csr->rowind;
  } else {
    num_ptrs = gk_csr->ncols + 1;
    nnz = gk_csr->colptr[num_ptrs - 1];
    gk_indptr = gk_csr->colptr;
    gk_indices = gk_csr->colind;
  }

  IdArray indptr_arr = aten::NewIdArray(num_ptrs);
  IdArray indices_arr = aten::NewIdArray(nnz);
  IdArray eids_arr = aten::NewIdArray(nnz);

  dgl_id_t *indptr = static_cast<dgl_id_t *>(indptr_arr->data);
  dgl_id_t *indices = static_cast<dgl_id_t *>(indices_arr->data);
  dgl_id_t *eids = static_cast<dgl_id_t *>(eids_arr->data);
  for (size_t i = 0; i < num_ptrs; i++) {
    indptr[i] = gk_indptr[i];
  }
  for (size_t i = 0; i < nnz; i++) {
    indices[i] = gk_indices[i];
    eids[i] = i;
  }

  return aten::CSRMatrix(
      gk_csr->nrows, gk_csr->ncols, indptr_arr, indices_arr, eids_arr);
}

#endif  // !defined(_WIN32)

GraphPtr GraphOp::ToBidirectedSimpleImmutableGraph(ImmutableGraphPtr ig) {
#if !defined(_WIN32)
  // TODO(zhengda) should we get whatever CSR exists in the graph.
  CSRPtr csr = ig->GetInCSR();
  gk_csr_t *gk_csr = Convert2GKCsr(csr->ToCSRMatrix(), true);
  gk_csr_t *sym_gk_csr = gk_csr_MakeSymmetric(gk_csr, GK_CSR_SYM_SUM);
  auto mat = Convert2DGLCsr(sym_gk_csr, true);
  gk_csr_Free(&gk_csr);
  gk_csr_Free(&sym_gk_csr);

  // This is a symmetric graph now. The in-csr and out-csr are the same.
  csr = CSRPtr(new CSR(mat.indptr, mat.indices, mat.data));
  return GraphPtr(new ImmutableGraph(csr, csr));
#else
  return GraphPtr();
#endif  // !defined(_WIN32)
}

}  // namespace dgl
