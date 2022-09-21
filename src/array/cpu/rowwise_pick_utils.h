/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/rowwise_pick_ut9ils.h
 * \brief Utility functions for row-wise sampling.
 */
#ifndef DGL_ARRAY_CPU_ROWWISE_PICK_UTILS_H_
#define DGL_ARRAY_CPU_ROWWISE_PICK_UTILS_H_

#include <dgl/array.h>
#include <dmlc/omp.h>
#include <dgl/runtime/parallel_for.h>

namespace dgl {
namespace aten {
namespace impl {
namespace {

// CSRRowWisePickPartial etc. returns a CSR matrix with seed nodes as rows.  This function
// restores the number of rows to the original matrix and populates the COO row array.
template <typename IdxType>
COOMatrix RowWisePickPartialCSRToCOO(CSRMatrix csr, IdArray rows, int64_t original_rows) {
  IdArray picked_rows = IdArray::Empty(
      {csr.indices->shape[0]}, csr.indices->dtype, csr.indices->ctx);
  IdxType* picked_rows_data = picked_rows.Ptr<IdxType>();
  const IdxType* indptr_data = csr.indptr.Ptr<IdxType>();
  const IdxType* rows_data = rows.Ptr<IdxType>();
  int64_t num_rows = rows->shape[0];

  runtime::parallel_for(0, num_rows, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      for (int64_t j = indptr_data[i]; j < indptr_data[i + 1]; ++j)
        picked_rows_data[j] = rows_data[i];
    }
  });
  return COOMatrix(original_rows, csr.num_cols, picked_rows, csr.indices, csr.data);
}

}  // namespace
}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_ROWWISE_PICK_UTILS_H_
