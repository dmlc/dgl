/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/coo_remove.cc
 * @brief COO matrix remove entries CPU implementation
 */
#include <dgl/array.h>

#include <utility>
#include <vector>

#include "array_utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

namespace {

/** @brief COORemove implementation for COOMatrix with default consecutive edge
 * IDs */
template <DGLDeviceType XPU, typename IdType>
void COORemoveConsecutive(
    COOMatrix coo, IdArray entries, std::vector<IdType> *new_rows,
    std::vector<IdType> *new_cols, std::vector<IdType> *new_eids) {
  const int64_t nnz = coo.row->shape[0];
  const int64_t n_entries = entries->shape[0];
  const IdType *row_data = static_cast<IdType *>(coo.row->data);
  const IdType *col_data = static_cast<IdType *>(coo.col->data);
  const IdType *entry_data = static_cast<IdType *>(entries->data);

  std::vector<IdType> entry_data_sorted(entry_data, entry_data + n_entries);
  std::sort(entry_data_sorted.begin(), entry_data_sorted.end());

  int64_t j = 0;
  for (int64_t i = 0; i < nnz; ++i) {
    if (j < n_entries && entry_data_sorted[j] == i) {
      // Move on to the next different entry
      while (j < n_entries && entry_data_sorted[j] == i) ++j;
      continue;
    }
    new_rows->push_back(row_data[i]);
    new_cols->push_back(col_data[i]);
    new_eids->push_back(i);
  }
}

/** @brief COORemove implementation for COOMatrix with shuffled edge IDs */
template <DGLDeviceType XPU, typename IdType>
void COORemoveShuffled(
    COOMatrix coo, IdArray entries, std::vector<IdType> *new_rows,
    std::vector<IdType> *new_cols, std::vector<IdType> *new_eids) {
  const int64_t nnz = coo.row->shape[0];
  const IdType *row_data = static_cast<IdType *>(coo.row->data);
  const IdType *col_data = static_cast<IdType *>(coo.col->data);
  const IdType *eid_data = static_cast<IdType *>(coo.data->data);

  IdHashMap<IdType> eid_map(entries);

  for (int64_t i = 0; i < nnz; ++i) {
    const IdType eid = eid_data[i];
    if (eid_map.Contains(eid)) continue;
    new_rows->push_back(row_data[i]);
    new_cols->push_back(col_data[i]);
    new_eids->push_back(eid);
  }
}

};  // namespace

template <DGLDeviceType XPU, typename IdType>
COOMatrix COORemove(COOMatrix coo, IdArray entries) {
  const int64_t nnz = coo.row->shape[0];
  const int64_t n_entries = entries->shape[0];
  if (n_entries == 0) return coo;

  std::vector<IdType> new_rows, new_cols, new_eids;
  new_rows.reserve(nnz - n_entries);
  new_cols.reserve(nnz - n_entries);
  new_eids.reserve(nnz - n_entries);

  if (COOHasData(coo))
    COORemoveShuffled<XPU, IdType>(
        coo, entries, &new_rows, &new_cols, &new_eids);
  else
    // Removing from COO ordered by eid has more efficient implementation.
    COORemoveConsecutive<XPU, IdType>(
        coo, entries, &new_rows, &new_cols, &new_eids);

  return COOMatrix(
      coo.num_rows, coo.num_cols, IdArray::FromVector(new_rows),
      IdArray::FromVector(new_cols), IdArray::FromVector(new_eids));
}

template COOMatrix COORemove<kDGLCPU, int32_t>(COOMatrix coo, IdArray entries);
template COOMatrix COORemove<kDGLCPU, int64_t>(COOMatrix coo, IdArray entries);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
