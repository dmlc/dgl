/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/coo_remove.cc
 * @brief CSR matrix remove entries CPU implementation
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

template <DGLDeviceType XPU, typename IdType>
void CSRRemoveConsecutive(
    CSRMatrix csr, IdArray entries, std::vector<IdType> *new_indptr,
    std::vector<IdType> *new_indices, std::vector<IdType> *new_eids) {
  CHECK_SAME_DTYPE(csr.indices, entries);
  const int64_t n_entries = entries->shape[0];
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  const IdType *entry_data = static_cast<IdType *>(entries->data);

  std::vector<IdType> entry_data_sorted(entry_data, entry_data + n_entries);
  std::sort(entry_data_sorted.begin(), entry_data_sorted.end());

  int64_t k = 0;
  new_indptr->push_back(0);
  for (int64_t i = 0; i < csr.num_rows; ++i) {
    for (IdType j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      if (k < n_entries && entry_data_sorted[k] == j) {
        // Move on to the next different entry
        while (k < n_entries && entry_data_sorted[k] == j) ++k;
        continue;
      }
      new_indices->push_back(indices_data[j]);
      new_eids->push_back(k);
    }
    new_indptr->push_back(new_indices->size());
  }
}

template <DGLDeviceType XPU, typename IdType>
void CSRRemoveShuffled(
    CSRMatrix csr, IdArray entries, std::vector<IdType> *new_indptr,
    std::vector<IdType> *new_indices, std::vector<IdType> *new_eids) {
  CHECK_SAME_DTYPE(csr.indices, entries);
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  const IdType *eid_data = static_cast<IdType *>(csr.data->data);

  IdHashMap<IdType> eid_map(entries);

  new_indptr->push_back(0);
  for (int64_t i = 0; i < csr.num_rows; ++i) {
    for (IdType j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      const IdType eid = eid_data ? eid_data[j] : j;
      if (eid_map.Contains(eid)) continue;
      new_indices->push_back(indices_data[j]);
      new_eids->push_back(eid);
    }
    new_indptr->push_back(new_indices->size());
  }
}

};  // namespace

template <DGLDeviceType XPU, typename IdType>
CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries) {
  CHECK_SAME_DTYPE(csr.indices, entries);
  const int64_t nnz = csr.indices->shape[0];
  const int64_t n_entries = entries->shape[0];
  if (n_entries == 0) return csr;

  std::vector<IdType> new_indptr, new_indices, new_eids;
  new_indptr.reserve(nnz - n_entries);
  new_indices.reserve(nnz - n_entries);
  new_eids.reserve(nnz - n_entries);

  if (CSRHasData(csr))
    CSRRemoveShuffled<XPU, IdType>(
        csr, entries, &new_indptr, &new_indices, &new_eids);
  else
    // Removing from CSR ordered by eid has more efficient implementation
    CSRRemoveConsecutive<XPU, IdType>(
        csr, entries, &new_indptr, &new_indices, &new_eids);

  return CSRMatrix(
      csr.num_rows, csr.num_cols, IdArray::FromVector(new_indptr),
      IdArray::FromVector(new_indices), IdArray::FromVector(new_eids));
}

template CSRMatrix CSRRemove<kDGLCPU, int32_t>(CSRMatrix csr, IdArray entries);
template CSRMatrix CSRRemove<kDGLCPU, int64_t>(CSRMatrix csr, IdArray entries);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
