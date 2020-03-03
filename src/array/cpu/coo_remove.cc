/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_remove.cc
 * \brief COO matrix remove entries CPU implementation
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
std::pair<COOMatrix, IdArray> COORemove(COOMatrix coo, IdArray entries) {
  const int64_t nnz = coo.row->shape[0];
  const IdType *row_data = static_cast<IdType *>(coo.row->data);
  const IdType *col_data = static_cast<IdType *>(coo.col->data);
  const IdType *eid_data = COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;

  IdHashMap<IdType> eid_map(entries);

  std::vector<IdType> new_rows, new_cols, new_eids;

  for (int64_t i = 0; i < nnz; ++i) {
    const IdType eid = eid_data ? eid_data[i] : i;
    if (eid_map.Contains(eid))
      continue;
    new_rows.push_back(row_data[i]);
    new_cols.push_back(col_data[i]);
    new_eids.push_back(eid);
  }

  const COOMatrix new_coo = COOMatrix(
      coo.num_rows, coo.num_cols,
      IdArray::FromVector(new_rows),
      IdArray::FromVector(new_cols));
  return std::make_pair(new_coo, IdArray::FromVector(new_eids));
}

template std::pair<COOMatrix, IdArray> COORemove<kDLCPU, int32_t>(COOMatrix coo, IdArray entries);
template std::pair<COOMatrix, IdArray> COORemove<kDLCPU, int64_t>(COOMatrix coo, IdArray entries);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
