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
  IdType *indices = csr.indices.Ptr<IdType>();
  const IdType *data = csr.data.Ptr<IdType>();
  // partitioned csr
  int64_t num_col_partitions = (N + num_cols_per_partition - 1) / num_cols_per_partition;
  IdArray ret_indptr = NDArray::Empty({num_col_partitions, M + 1}, csr.indptr->dtype, csr.indptr->ctx);
  IdArray ret_indices = NDArray::Empty({nnz}, csr.indices->dtype, csr.indices->ctx);
  IdArray ret_eid = NDArray::Empty({nnz}, csr.indptr->dtype, csr.indptr->ctx);
  IdType *ret_indptr_data = static_cast<IdType *>(ret_indptr->data);
  IdType *ret_indices_data = static_cast<IdType *>(ret_indices->data);
  IdType *ret_eid_data = static_cast<IdType *>(ret_eid->data);
  // keep left pointer
  std::vector<int64_t> left(M, 0);
  // keep postition
  std::vector<int64_t> pos(indptr, indptr + M);
  int64_t count = 0;
  for (int64_t j = 1; j <= num_col_partitions; j++) {
    int64_t col_end = (j * num_cols_per_partition) < N ? (j * num_cols_per_partition) : N;
    #pragma omp parallel for
    for (int64_t i = 0; i < M; i++) {
      // binary search the col_end
      // assume the indices are sorted
      int64_t l = pos[i];
      int64_t r = indptr[i + 1];
      left[i] = l;
      if (l == r) {
        pos[i] = l;
        continue;
      }
      IdType *pivot = std::lower_bound(indices + l, indices + r, col_end);
      pos[i] = pivot - indices;
    }
    for (int64_t i = 0; i < M; i++) {
      ret_indptr_data[(j - 1) * (M + 1) + i] = count;
      std::copy(indices + left[i], indices + pos[i], ret_indices_data + count);
      std::copy(data + left[i], data + pos[i], ret_eid_data + count);
      count += pos[i] - left[i];
    }
    ret_indptr_data[j * (M + 1) - 1] = count;
  }
  return { ret_indptr, ret_indices, ret_eid };
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
  IdType *indices = csr.indices.Ptr<IdType>();
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
  // keep left pointer
  std::vector<int64_t> left(num_rows_per_partition, 0);
  // keep position
  std::vector<int64_t> pos(num_rows_per_partition, 0);
  int64_t count = 0;
  for (int64_t i = 0; i < num_row_partitions; i++) {
    // for each row parition
    int64_t row_start = i * num_rows_per_partition;
    int64_t row_end = row_start + num_rows_per_partition;
    row_end = row_end > M ? M : row_end;
    // initialize left pointer
    for (int64_t r = row_start; r < row_end; r++) {
      pos[r - row_start] = indptr[r];
    }
    // for each col partition
    for (int64_t j = 1; j <= num_col_partitions; j++) {
      int64_t col_end = (j * num_cols_per_partition) < N ? (j * num_cols_per_partition) : N;
      // perform binary search in parallel
      #pragma omp parallel for
      for (int64_t ri = 0; ri < row_end - row_start; ri++) {
        // binary search the col_end
        // assume the indices are sorted
        int64_t l = pos[ri];
        int64_t r = indptr[ri + row_start + 1];
        left[ri] = l;
        if (l == r) {
          pos[ri] = l;
          continue;
        }
        IdType *pivot = std::lower_bound(indices + l, indices + r, col_end);
        pos[ri] = pivot - indices;
      }
      // update count and copy data sequentially
      for (int64_t ri = 0; ri < row_end - row_start; ri++) {
        int64_t num_elem = pos[ri] - left[ri];
        std::copy(indices + left[ri], indices + pos[ri], ret_col_data + count);
        std::copy(data + left[ri], data + pos[ri], ret_eid_data + count);
        std::fill(ret_row_data + count, ret_row_data + count + num_elem, ri + row_start);
        count += num_elem;
      }
    }
  }
  return {ret_row, ret_col, ret_eid};
}


template std::vector<IdArray> partition2D<int64_t>(CSRMatrix csr, int64_t num_cols_per_partition, int64_t num_rows_per_partition);
template std::vector<IdArray> partition2D<int32_t>(CSRMatrix csr, int64_t num_cols_per_partition, int64_t num_rows_per_partition);

DGL_REGISTER_GLOBAL("sparse._CAPI_DGLPartition1D")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  HeteroGraphRef hg = args[0];
  dgl_type_t etype = args[1];
  int64_t num_cols_per_partition = args[2];
  CSRMatrix csr = hg->GetCSCMatrix(etype);
  csr = CSRSort(csr);
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
  csr = CSRSort(csr);
  ATEN_ID_TYPE_SWITCH(hg->DataType(), IdType, {
    *rv = ConvertNDArrayVectorToPackedFunc(
      partition2D<IdType>(csr, num_rows_per_partition, num_cols_per_partition)
      );
  });
});

} // namespace aten
} // namespace dgl