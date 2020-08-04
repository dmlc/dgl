/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/partition.cc
 * \brief CSR partitioning
 */
#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <algorithm>
#include <vector>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl
{
namespace aten
{

///////////////////////////// Partition1D /////////////////////////////
template <typename IdType>
std::vector<IdArray> partition1D(CSRMatrix csr, int64_t num_cols_per_partition) {
  // original csr
  const int64_t M = csr.num_rows;
  const int64_t N = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType *indptr = csr.indptr.Ptr<IdType>();
  const IdType *indices = csr.indices.Ptr<IdType>();
  const IdType *data = csr.data.Ptr<IdType>();
  // partitioned csr
  int64_t num_col_partitions = (N + num_cols_per_partition - 1) / num_cols_per_partition;
  IdArray ret_indptr = NDArray::Empty({num_col_partitions, M + 1}, csr.indptr->dtype, csr.indptr->ctx);
  IdArray ret_indices = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  IdArray ret_eid = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  IdType *ret_indptr_data = static_cast<IdType *>(ret_indptr->data);
  IdType *ret_indices_data = static_cast<IdType *>(ret_indices->data);
  IdType *ret_eid_data = static_cast<IdType *>(ret_eid->data);
  // memo
  std::vector<int64_t> memo(indptr, indptr + M);
  int64_t count = 0;
  for (int64_t j = 1; j <= num_col_partitions; j++) {
    int64_t col_end = (j * num_cols_per_partition) < N ? (j * num_cols_per_partition) : N;
    for (int64_t i = 0; i < M; i++) {
      // binary search the col_end
      // assume the indices are sorted
      int64_t l = memo[i];
      int64_t r = indptr[i + 1];
      ret_indptr_data[(j - 1) * (M + 1) + i] = count;
      if (l == r)
        continue;
      IdType *pos = std::lower_bound(indices + l, indices + r, col_end);
      // update memo
      memo[i] = pos - indices;
      // update count, copy data
      int64_t num_elem = pos - (indices + l);
      std::copy(indices + l, pos, ret_indices_data + count);
      std::copy(data + l, data + (pos - indices), ret_eid_data + count);
      count += num_elem;
    }
    ret_indptr_data[j * (M + 1) - 1] = count;
  }
  return { ret_indptr, ret_indices, ret_eid }
}

template std::vector<IdArray> partition1D<int64_t>(CSRMatrix csr, int64_t num_cols_per_partition);
template std::vector<IdArray> partition1D<int32_t>(CSRMatrix csr, int64_t num_cols_per_partition);

///////////////////////////// Partition1D /////////////////////////////
template <typename IdType>
std::vector<IdArray> partition2D(CSRMatrix csr, int64_t num_rows_per_partition, int64_t num_cols_per_partition) {
  // original csr
  const int64_t M = csr.num_rows;
  const int64_t N = csr.num_cols;
  const int64_t nnz = csr.indices->shape[0];
  const IdType *indptr = csr.indptr.Ptr<IdType>();
  const IdType *indices = csr.indices.Ptr<IdType>();
  const IdType *data = csr.data.Ptr<IdType>();
  // partitioned csr
  int64_t num_col_partitions = (N + num_cols_per_partition - 1) / num_cols_per_partition;
  int64_t num_row_partitions = (M + num_rows_per_partition - 1) / num_rows_per_partition;
  NDArray ret_row = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray ret_col = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  NDArray ret_eid = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  IdType *ret_row_data = static_cast<IdType *>(ret_row->data);
  IdType *ret_col_data = static_cast<IdType *>(ret_col->data);
  IdType *ret_eid_data = static_cast<IdType *>(ret_eid->data);
  // memo
  std::vector<IdType *> memo(num_rows_per_partition, nullptr);
  int64_t count = 0;
  for (int64_t i = 0; i < num_row_partitions; i++) {
    // for each row parition
    int64_t row_start = i * num_rows_per_partition;
    int64_t row_end = row_start + num_rows_per_partition;
    row_end = row_end > M ? M : row_end;
    // initialize memo
    for (int64_t r = row_start; r < row_end; r++) {
      memo[r - row_start] = indices + indptr[r];
    }
    // for each col partition
    for (int64_t j = 1; j <= num_col_partitions; j++) {
      int64_t col_end = (j * num_cols_per_partition) < N ? (j * num_cols_per_partition) : N;
      // for each row in the patition
      for (int64_t ri = row_start; ri < row_end; ri++) {
        // binary search the col_end
        // assume the indices are sorted
        IdType *l = memo[ri - row_start];
        IdType *r = indices + indptr[ri + 1];
        if (l == r)
          continue;
        IdType *pos = std::lower_bound(l, r, col_end);
        // update memo
        memo[ri - row_start] = pos;
        // update count, copy data
        int64_t num_elem = pos - l;
        std::copy(l, pos, ret_col_data + count);
        std::copy(data + (l - indices), data + (pos - indices), ret_eid_data + count);
        std::fill(ret_row_data + count, ret_row_data + count + num_elem, ri);
        count += num_elem;
      }
    }
  }
  return {ret_row, ret_col, ret_eid}
}


template std::vector<IdArray> partition2D<int64_t>(CSRMatrix csr, int64_t num_cols_per_partition, int64_t num_rows_per_partition);
template std::vector<IdArray> partition2D<int32_t>(CSRMatrix csr, int64_t num_cols_per_partition, int64_t num_rows_per_partition);

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLPartition1D")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  HeteroGraphRef hg = args[0];
  dgl_type_t etype = args[1];
  int64_t num_cols_per_partition = args[2];
  CSRMatrix csr = hg->GetCSRMatrix(etype);
  ATEN_ID_TYPE_SWITCH(hg->DataType(), IdType, {
    *rv = ConvertNDArrayVectorToPackedFunc(
      partition1D<IdType>(csr, num_cols_per_partition)
      );
  });
});

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLPartition2D")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  HeteroGraphRef hg = args[0];
  dgl_type_t etype = args[1];
  int64_t num_rows_per_partition = args[2];
  int64_t num_cols_per_partition = args[3];
  CSRMatrix csr = hg->GetCSRMatrix(etype);
  ATEN_ID_TYPE_SWITCH(hg->DataType(), IdType, {
    *rv = ConvertNDArrayVectorToPackedFunc(
      partition2D<IdType>(csr, num_rows_per_partition, num_cols_per_partition)
      );
  });
});

} // namespace aten
} // namespace dgl