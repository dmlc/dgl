/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/coo_remove_if.cc
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

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COORemoveIf(COOMatrix coo, NDArray values, DType criteria) {
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* data = COOHasData(coo) ? coo.data.Ptr<IdType>() : nullptr;
  const DType* val = values.Ptr<DType>();
  const auto idtype = coo.row->dtype;
  const auto ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  IdArray new_row = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_col = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_eid = IdArray::Empty({nnz}, idtype, ctx);
  IdType* new_row_data = new_row.Ptr<IdType>();
  IdType* new_col_data = new_col.Ptr<IdType>();
  IdType* new_eid_data = new_eid.Ptr<IdType>();

  int64_t j = 0;
  for (int64_t i = 0; i < nnz; ++i) {
    const IdType eid = data ? data[i] : i;
    if (val[eid] != criteria) {
      new_row_data[j] = row[i];
      new_col_data[j] = col[i];
      new_eid_data[j] = eid;
      ++j;
    }
  }
  return COOMatrix(
      coo.num_rows,
      coo.num_cols,
      new_row.CreateView({j}, idtype, 0),
      new_col.CreateView({j}, idtype, 0),
      new_eid.CreateView({j}, idtype, 0));
}

template COOMatrix COORemoveIf<kDGLCPU, int32_t, int8_t>(COOMatrix, NDArray, int8_t);
template COOMatrix COORemoveIf<kDGLCPU, int32_t, uint8_t>(COOMatrix, NDArray, uint8_t);
template COOMatrix COORemoveIf<kDGLCPU, int32_t, float>(COOMatrix, NDArray, float);
template COOMatrix COORemoveIf<kDGLCPU, int32_t, double>(COOMatrix, NDArray, double);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, int8_t>(COOMatrix, NDArray, int8_t);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, uint8_t>(COOMatrix, NDArray, uint8_t);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, float>(COOMatrix, NDArray, float);
template COOMatrix COORemoveIf<kDGLCPU, int64_t, double>(COOMatrix, NDArray, double);

template <DGLDeviceType XPU, typename IdType, typename DType>
COOMatrix COOEtypeRemoveIf(
    COOMatrix coo, IdArray etypes, IdArray eids, const std::vector<NDArray>& values,
    DType criteria) {
  CHECK_EQ(etypes->dtype, DGLDataTypeTraits<int32_t>::dtype) <<
    "currently only int32 edge type array is supported.";
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* data = COOHasData(coo) ? coo.data.Ptr<IdType>() : nullptr;
  const int32_t* etype_data = etypes.Ptr<int32_t>();
  const IdType* eid_data = eids.Ptr<IdType>();
  std::vector<DType*> val(etypes->shape[0]);
  for (size_t i = 0; i < values.size(); ++i)
    val[i] = IsNullArray(values[i]) ? nullptr : values[i].Ptr<DType>();
  const auto idtype = coo.row->dtype;
  const auto ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  IdArray new_row = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_col = IdArray::Empty({nnz}, idtype, ctx);
  IdArray new_eid = IdArray::Empty({nnz}, idtype, ctx);
  IdType* new_row_data = new_row.Ptr<IdType>();
  IdType* new_col_data = new_col.Ptr<IdType>();
  IdType* new_eid_data = new_eid.Ptr<IdType>();

  int64_t j = 0;
  for (int64_t i = 0; i < nnz; ++i) {
    const IdType global_eid = data ? data[i] : i;
    const int32_t etype = etype_data[global_eid];
    const IdType etype_eid = eid_data[global_eid];
    if (!val[etype] ||  // Pass if the value array is nullptr (i.e. doesn't matter).
                        // This can happen if some etypes don't matter while others do.
        val[etype][etype_eid] != criteria) {
      new_row_data[j] = row[i];
      new_col_data[j] = col[i];
      new_eid_data[j] = global_eid;
      ++j;
    }
  }
  return COOMatrix(
      coo.num_rows,
      coo.num_cols,
      new_row.CreateView({j}, idtype, 0),
      new_col.CreateView({j}, idtype, 0),
      new_eid.CreateView({j}, idtype, 0));
}

template COOMatrix COOEtypeRemoveIf<kDGLCPU, int32_t, int8_t>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, int8_t);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int32_t, uint8_t>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, uint8_t);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int32_t, float>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, float);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int32_t, double>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, double);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int64_t, int8_t>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, int8_t);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int64_t, uint8_t>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, uint8_t);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int64_t, float>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, float);
template COOMatrix COOEtypeRemoveIf<kDGLCPU, int64_t, double>(
    COOMatrix, IdArray, IdArray, const std::vector<NDArray>&, double);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
