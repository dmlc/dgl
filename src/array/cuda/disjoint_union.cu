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
 * @file array/gpu/disjoint_union.cu
 * @brief Disjoint union GPU implementation.
 */

#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

#include <tuple>
#include <vector>

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template <typename IdType>
__global__ void _DisjointUnionKernel(
    IdType** arrs, IdType* prefix, IdType* offset, IdType* out, int64_t n_arrs,
    int n_elms) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < n_elms) {
    IdType i = dgl::cuda::_UpperBound(offset, n_arrs, tx) - 1;
    if (arrs[i] == NULL) {
      out[tx] = tx;
    } else {
      IdType j = tx - offset[i];
      out[tx] = arrs[i][j] + prefix[i];
    }
    tx += stride_x;
  }
}

template <DGLDeviceType XPU, typename IdType>
std::tuple<IdArray, IdArray, IdArray> _ComputePrefixSums(
    const std::vector<COOMatrix>& coos) {
  IdType n = coos.size(), nbits = coos[0].row->dtype.bits;
  IdArray n_rows = NewIdArray(n, CPU, nbits);
  IdArray n_cols = NewIdArray(n, CPU, nbits);
  IdArray n_elms = NewIdArray(n, CPU, nbits);

  IdType* n_rows_data = n_rows.Ptr<IdType>();
  IdType* n_cols_data = n_cols.Ptr<IdType>();
  IdType* n_elms_data = n_elms.Ptr<IdType>();

  dgl::runtime::parallel_for(0, coos.size(), [&](IdType b, IdType e) {
    for (IdType i = b; i < e; ++i) {
      n_rows_data[i] = coos[i].num_rows;
      n_cols_data[i] = coos[i].num_cols;
      n_elms_data[i] = coos[i].row->shape[0];
    }
  });

  return std::make_tuple(
      CumSum(n_rows.CopyTo(coos[0].row->ctx), true),
      CumSum(n_cols.CopyTo(coos[0].row->ctx), true),
      CumSum(n_elms.CopyTo(coos[0].row->ctx), true));
}

template <DGLDeviceType XPU, typename IdType>
void _Merge(
    IdType** arrs, IdType* prefix, IdType* offset, IdType* out, int64_t n_arrs,
    int n_elms, DGLContext ctx, DGLDataType dtype, cudaStream_t stream) {
  auto device = runtime::DeviceAPI::Get(ctx);
  int nt = 256;
  int nb = (n_elms + nt - 1) / nt;

  IdType** arrs_dev = static_cast<IdType**>(
      device->AllocWorkspace(ctx, n_arrs * sizeof(IdType*)));

  device->CopyDataFromTo(
      arrs, 0, arrs_dev, 0, sizeof(IdType*) * n_arrs, DGLContext{kDGLCPU, 0},
      ctx, dtype);

  CUDA_KERNEL_CALL(
      _DisjointUnionKernel, nb, nt, 0, stream, arrs_dev, prefix, offset, out,
      n_arrs, n_elms);

  device->FreeWorkspace(ctx, arrs_dev);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix DisjointUnionCoo(const std::vector<COOMatrix>& coos) {
  cudaStream_t stream = runtime::getCurrentCUDAStream();
  auto device = runtime::DeviceAPI::Get(coos[0].row->ctx);
  uint64_t src_offset = 0, dst_offset = 0;
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

  std::unique_ptr<IdType*[]> rows(new IdType*[coos.size()]);
  std::unique_ptr<IdType*[]> cols(new IdType*[coos.size()]);
  std::unique_ptr<IdType*[]> data(new IdType*[coos.size()]);

  for (size_t i = 0; i < coos.size(); i++) {
    row_sorted &= coos[i].row_sorted;
    col_sorted &= coos[i].col_sorted;
    rows[i] = coos[i].row.Ptr<IdType>();
    cols[i] = coos[i].col.Ptr<IdType>();
    data[i] = coos[i].data.Ptr<IdType>();
  }

  auto ctx = coos[0].row->ctx;
  auto dtype = coos[0].row->dtype;

  IdType n_elements = 0;
  device->CopyDataFromTo(
      &prefix_elm[coos.size()], 0, &n_elements, 0, sizeof(IdType),
      coos[0].row->ctx, DGLContext{kDGLCPU, 0}, coos[0].row->dtype);

  device->CopyDataFromTo(
      &prefix_src[coos.size()], 0, &src_offset, 0, sizeof(IdType),
      coos[0].row->ctx, DGLContext{kDGLCPU, 0}, coos[0].row->dtype);

  device->CopyDataFromTo(
      &prefix_dst[coos.size()], 0, &dst_offset, 0, sizeof(IdType),
      coos[0].row->ctx, DGLContext{kDGLCPU, 0}, coos[0].row->dtype);

  // Union src array
  IdArray result_src =
      NewIdArray(n_elements, coos[0].row->ctx, coos[0].row->dtype.bits);
  _Merge<XPU, IdType>(
      rows.get(), prefix_src, prefix_elm, result_src.Ptr<IdType>(), coos.size(),
      n_elements, ctx, dtype, stream);

  // Union dst array
  IdArray result_dst =
      NewIdArray(n_elements, coos[0].col->ctx, coos[0].col->dtype.bits);
  _Merge<XPU, IdType>(
      cols.get(), prefix_dst, prefix_elm, result_dst.Ptr<IdType>(), coos.size(),
      n_elements, ctx, dtype, stream);

  // Union data array if exists and fetch number of elements
  IdArray result_dat = NullArray();
  if (has_data) {
    result_dat =
        NewIdArray(n_elements, coos[0].row->ctx, coos[0].row->dtype.bits);
    _Merge<XPU, IdType>(
        data.get(), prefix_elm, prefix_elm, result_dat.Ptr<IdType>(),
        coos.size(), n_elements, ctx, dtype, stream);
  }

  return COOMatrix(
      src_offset, dst_offset, result_src, result_dst, result_dat, row_sorted,
      col_sorted);
}

template COOMatrix DisjointUnionCoo<kDGLCUDA, int32_t>(
    const std::vector<COOMatrix>& coos);
template COOMatrix DisjointUnionCoo<kDGLCUDA, int64_t>(
    const std::vector<COOMatrix>& coos);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
