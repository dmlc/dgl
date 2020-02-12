/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_rowwise_topk.cc
 * \brief CSR rowwise topk
 */
#include <numeric>
#include <algorithm>
#include "./rowwise_pick.h"

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdxType, typename FloatType>
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending) {
  const IdxType* indptr = static_cast<IdxType*>(mat.indptr->data);
  const IdxType* indices = static_cast<IdxType*>(mat.indices->data);
  const IdxType* eids = static_cast<IdxType*>(mat.data->data);
  const FloatType* wdata = static_cast<FloatType*>(weight->data);

  std::function<bool(IdxType, IdxType)> compare_fn;
  if (ascending)
    compare_fn = [wdata, eids] (IdxType i, IdxType j) {
        return wdata[eids[i]] < wdata[eids[j]];
      };
  else
    compare_fn = [wdata, eids] (IdxType i, IdxType j) {
        return wdata[eids[i]] > wdata[eids[j]];
      };

  PickFn<IdxType> pick_fn = [indptr, indices, k, compare_fn]
    (IdxType rowid, IdxType* out_row, IdxType* out_col, IdxType* out_idx) {
      const IdxType off = indptr[rowid];
      const IdxType len = indptr[rowid + 1] - off;
      std::vector<IdxType> idx(len);
      std::iota(idx.begin(), idx.end(), off);
      std::sort(idx.begin(), idx.end(), compare_fn);
      for (int64_t j = 0; j < k; ++j) {
        out_row[j] = rowid;
        out_col[j] = indices[idx[j]];
        out_idx[j] = idx[j];
      }
    };

  return CSRRowWisePick(mat, rows, k, false, pick_fn);
}

template COOMatrix CSRRowWiseTopk<kDLCPU, int32_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseTopk<kDLCPU, int64_t, float>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseTopk<kDLCPU, int32_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);
template COOMatrix CSRRowWiseTopk<kDLCPU, int64_t, double>(
    CSRMatrix, IdArray, int64_t, FloatArray, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
