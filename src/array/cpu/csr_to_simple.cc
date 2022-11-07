/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/csr_to_simple.cc
 * @brief CSR sorting
 */
#include <dgl/array.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple(CSRMatrix csr) {
  if (!csr.sorted) csr = CSRSort(csr);

  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);

  std::vector<IdType> indptr;
  std::vector<IdType> indices;
  std::vector<IdType> count;
  indptr.resize(csr.indptr->shape[0]);
  indptr[0] = 0;

  for (int64_t i = 1; i < csr.indptr->shape[0]; ++i) {
    if (indptr_data[i - 1] == indptr_data[i]) {
      indptr[i] = indptr[i - 1];
      continue;
    }

    int64_t cnt = 1;
    int64_t dup_cnt = 1;
    indices.push_back(indices_data[indptr_data[i - 1]]);
    for (int64_t j = indptr_data[i - 1] + 1; j < indptr_data[i]; ++j) {
      if (indices_data[j - 1] == indices_data[j]) {
        ++dup_cnt;
        continue;
      }
      count.push_back(dup_cnt);
      dup_cnt = 1;
      indices.push_back(indices_data[j]);
      ++cnt;
    }
    count.push_back(dup_cnt);
    indptr[i] = indptr[i - 1] + cnt;
  }

  CSRMatrix res_csr = CSRMatrix(
      csr.num_rows, csr.num_cols, IdArray::FromVector(indptr),
      IdArray::FromVector(indices), NullArray(), true);

  const IdArray &edge_count = IdArray::FromVector(count);
  const IdArray new_eids =
      Range(0, res_csr.indices->shape[0], sizeof(IdType) * 8, csr.indptr->ctx);
  const IdArray eids_remapped =
      CSRHasData(csr) ? Scatter(Repeat(new_eids, edge_count), csr.data)
                      : Repeat(new_eids, edge_count);

  return std::make_tuple(res_csr, edge_count, eids_remapped);
}

template std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple<kDGLCPU, int32_t>(
    CSRMatrix);
template std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple<kDGLCPU, int64_t>(
    CSRMatrix);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
