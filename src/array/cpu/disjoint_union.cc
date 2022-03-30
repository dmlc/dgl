/**
*   Copyright (c) 2022, NVIDIA CORPORATION.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.
*
* \file array/cpu/disjoint_union.cc
* \brief Disjoint union CPU implementation.
*/

#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>
#include <tuple>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
std::tuple<IdArray, IdArray, IdArray> _ComputePrefixSums(const std::vector<COOMatrix>& coos) {
  IdArray prefix_src_arr = NewIdArray(
    coos.size(), coos[0].row->ctx, coos[0].row->dtype.bits);
  IdArray prefix_dst_arr = NewIdArray(
    coos.size(), coos[0].row->ctx, coos[0].row->dtype.bits);
  IdArray prefix_elm_arr = NewIdArray(
    coos.size(), coos[0].row->ctx, coos[0].row->dtype.bits);

  IdType* prefix_src = static_cast<IdType*>(prefix_src_arr->data);
  IdType* prefix_dst = static_cast<IdType*>(prefix_dst_arr->data);
  IdType* prefix_elm = static_cast<IdType*>(prefix_elm_arr->data);

  dgl::runtime::parallel_for(0, coos.size(), [&](IdType b, IdType e){
    for (IdType i = b; i < e; ++i) {
      prefix_src[i] = coos[i].num_rows;
      prefix_dst[i] = coos[i].num_cols;
      prefix_elm[i] = coos[i].row->shape[0];
    }
  });

  return {CumSum(prefix_src_arr, true),
          CumSum(prefix_dst_arr, true),
          CumSum(prefix_elm_arr, true)};
}

template <DLDeviceType XPU, typename IdType>
COOMatrix DisjointUnionCoo(const std::vector<COOMatrix>& coos) {
  bool has_data = false;
  bool row_sorted = true;
  bool col_sorted = true;
  // check if data index array
  for (size_t i = 0; i < coos.size(); ++i) {
    CHECK_SAME_DTYPE(coos[0].row, coos[i].row);
    CHECK_SAME_CONTEXT(coos[0].row, coos[i].row);
    has_data |= COOHasData(coos[i]);
  }

  auto prefixes = _ComputePrefixSums<XPU, IdType>(coos);
  auto prefix_src = static_cast<IdType*>(std::get<0>(prefixes)->data);
  auto prefix_dst = static_cast<IdType*>(std::get<1>(prefixes)->data);
  auto prefix_elm = static_cast<IdType*>(std::get<2>(prefixes)->data);

  auto result_src = NewIdArray(
    prefix_elm[coos.size()], coos[0].row->ctx, coos[0].row->dtype.bits);
  auto result_dst = NewIdArray(
    prefix_elm[coos.size()], coos[0].col->ctx, coos[0].col->dtype.bits);
  auto result_dat = NullArray();
  if (has_data) {
    result_dat =  NewIdArray(
      prefix_elm[coos.size()], coos[0].row->ctx, coos[0].row->dtype.bits);
  }

  auto res_src_data = static_cast<IdType*>(result_src->data);
  auto res_dst_data = static_cast<IdType*>(result_dst->data);
  auto res_dat_data = static_cast<IdType*>(result_dat->data);

  dgl::runtime::parallel_for(0, coos.size(), [&](IdType b, IdType e){
    for (IdType i = b; i < e; ++i) {
      const aten::COOMatrix &coo = coos[i];
      if (!coo.row_sorted) row_sorted = false;
      if (!coo.col_sorted) col_sorted = false;

      auto edges_src = static_cast<IdType*>(coo.row->data);
      auto edges_dst = static_cast<IdType*>(coo.col->data);
      auto edges_dat = static_cast<IdType*>(coo.data->data);

      for (IdType j = 0; j < coo.row->shape[0]; j++) {
        res_src_data[prefix_elm[i] + j] = edges_src[j] + prefix_src[i];
      }

      for (IdType j = 0; j < coo.row->shape[0]; j++) {
        res_dst_data[prefix_elm[i] + j] = edges_dst[j] + prefix_dst[i];
      }

      if (has_data) {
        for (IdType j = 0; j < coo.row->shape[0]; j++) {
          const auto d = (!COOHasData(coo)) ? IdType(j) : edges_dat[j];
          res_dat_data[prefix_elm[i]+j] = d + prefix_elm[i];
        }
      }
    }
  });
  return COOMatrix(
    prefix_src[coos.size()], prefix_dst[coos.size()],
    result_src,
    result_dst,
    result_dat,
    row_sorted,
    col_sorted);
}

template COOMatrix DisjointUnionCoo<kDLCPU, int32_t>(const std::vector<COOMatrix>& coos);
template COOMatrix DisjointUnionCoo<kDLCPU, int64_t>(const std::vector<COOMatrix>& coos);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
