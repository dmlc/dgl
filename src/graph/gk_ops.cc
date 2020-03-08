/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/gk_ops.cc
 * \brief Graph operation implemented in GKlib
 */

#include <GKlib.h>

#include <dgl/graph_op.h>

namespace dgl {
namespace {
gk_csr_t *Convert2GKCsr(CSRPtr csr, bool is_row) {
  // TODO(zhengda) The conversion will be zero-copy in the future.
  const aten::CSRMatrix mat = csr->ToCSRMatrix();
  const dgl_id_t *indptr = static_cast<dgl_id_t*>(mat.indptr->data);
  const dgl_id_t *indices = static_cast<dgl_id_t*>(mat.indices->data);

  gk_csr_t *gk_csr = gk_csr_Create();
  gk_csr->nrows = mat.num_rows;
  gk_csr->ncols = mat.num_cols;
  size_t nnz = csr->NumEdges();
  auto gk_indptr = gk_csr->rowptr;
  auto gk_indices = gk_csr->rowind;
  size_t num_ptrs;
  if (is_row) {
    num_ptrs = gk_csr->nrows + 1;
    gk_indptr = gk_csr->rowptr = gk_zmalloc(gk_csr->nrows+1, "gk_csr_ExtractPartition: rowptr");
    gk_indices = gk_csr->rowind = gk_imalloc(nnz, "gk_csr_ExtractPartition: rowind");
  } else {
    num_ptrs = gk_csr->ncols + 1;
    gk_indptr = gk_csr->colptr = gk_zmalloc(gk_csr->ncols+1, "gk_csr_ExtractPartition: colptr");
    gk_indices = gk_csr->colind = gk_imalloc(nnz, "gk_csr_ExtractPartition: colind");
  }

  for (size_t i = 0; i < num_ptrs; i++) {
    gk_indptr[i] = indptr[i];
  }
  for (size_t i = 0; i < nnz; i++) {
    gk_indices[i] = indices[i];
  }
  return gk_csr;
}

CSRPtr Convert2DGLCsr(gk_csr_t *gk_csr, bool is_row) {
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
  CSRPtr csr = CSRPtr(new CSR(indptr_arr, indices_arr, eids_arr));

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

  return csr;
}

}  // namespace

GraphPtr GraphOp::ToBidirectedSimpleImmutableGraph(ImmutableGraphPtr ig) {
  printf("convert to symmetric simple immutable graph\n");
  // TODO(zhengda) should we get whatever CSR exists in the graph.
  CSRPtr csr = ig->GetInCSR();
  gk_csr_t *gk_csr = Convert2GKCsr(csr, true);
  gk_csr_t *sym_gk_csr = gk_csr_MakeSymmetric(gk_csr, GK_CSR_SYM_SUM);
  csr = Convert2DGLCsr(sym_gk_csr, true);
  // This is a symmetric graph now. The in-csr and out-csr are the same.
  return GraphPtr(new ImmutableGraph(csr, csr));
}

}  // namespace dgl
