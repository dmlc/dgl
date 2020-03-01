/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_remove.cc
 * \brief CSR matrix remove entries CPU implementation
 */
#include <dgl/array.h>
#include <utility>
#include <vector>
#include "array_utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
std::pair<CSRMatrix, IdArray> CSRRemove(CSRMatrix csr, IdArray entries) {
  const int64_t num_eids = values->shape[0];
  const int64_t num_rows = csr.num_rows;
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  const IdType *eid_data = CSRHasData(csr) ? static_cast<IdType *>(csr.data->data) : nullptr;

  IdHashMap<IdType> eid_map(values);

  std::vector<IdType> new_indptr, new_indices, new_eids;

  new_indptr.push_back(0);
  for (int64_t i = 0; i < num_rows; ++i) {
    for (IdType j = indptr_data[i]; j < indptr_data[i + 1]; ++j) {
      const IdType eid = eid_data ? eid_data[j] : j;
      if (eid_map.Contains(eid))
        continue;
      new_indices.push_back(indices_data[j]);
      new_eids.push_back(eid);
    }
    new_indptr.push_back(new_indices.size());
  }

  const CSRMatrix new_csr = CSRMatrix(
      num_rows, num_cols,
      IdArray::FromVector(new_indptr),
      IdArray::FromVector(new_indices));
  return std::make_pair(new_csr, IdArray::FromVector(new_eids));
}

template std::pair<CSRMatrix, IdArray> CSRRemove<kDLCPU, int32_t>(CSRMatrix csr, IdArray entries);
template std::pair<CSRMatrix, IdArray> CSRRemove<kDLCPU, int64_t>(CSRMatrix csr, IdArray entries);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
