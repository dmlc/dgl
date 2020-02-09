/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/coo_rowwise_topk.cc
 * \brief Selecting K-largest elements on each row of a COO matrix
 */

#include <dgl/array.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include "../../c_api_common.h"

namespace dgl {

namespace aten {

namespace impl {

namespace {

// Finds the K-largest elements in weights_data[i:j], and push the corresponding
// row/column/entry/weight into the vectors.
template <DLDeviceType XPU, typename IdType, typename DType, typename WType>
void PushTopK(
    IdType curr_row,                      // current row
    const IdType *col_data,               // column array of the whole matrix
    const IdType *entry_data,             // entry ID array of the whole matrix
    const WType *weights_data,            // weight array of the corresponding entries
    int64_t K,
    bool smallest,                        // whether to find the smallest items instead
    IdType i,
    IdType j,
    std::vector<IdType> *indices,         // a pre-allocated buffer with size NNZ
    std::vector<IdType> *new_rows,        // output row vector
    std::vector<IdType> *new_cols,        // output column vector
    std::vector<DType> *new_entries,      // output entry vector
    std::vector<WType> *new_weights) {    // output weight vector
  if (j - i <= K) {
    for (IdType k = i; k < j; ++k) {
      new_rows->push_back(curr_row);
      new_cols->push_back(col_data[k]);
      new_entries->push_back(entry_data ? entry_data[k] : k);
      new_weights->push_back(weights_data[k]);
    }
  } else {
    auto begin = indices->begin() + i;
    auto end = indices->begin() + j;
    std::nth_element(begin, begin + K, end, [weights_data, smallest] (IdType a, IdType b) {
        return smallest ?
            (weights_data[a] < weights_data[b]) :
            (weights_data[a] > weights_data[b]);
      });

    for (auto ptr = begin; ptr != begin + K; ++ptr) {
      new_rows->push_back(curr_row);
      new_cols->push_back(col_data[*ptr]);
      new_entries->push_back(entry_data ? entry_data[*ptr] : *ptr);
      new_weights->push_back(weights_data[*ptr]);
    }
  }
}

};  // namespace

template <DLDeviceType XPU, typename IdType, typename DType, typename WType>
std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero(
    COOMatrix coo,
    int64_t K,
    NDArray weights,
    bool smallest) {
  CHECK(weights->ctx == coo.row->ctx) <<
    "context of weights and graph mismatch; is weights array or graph on GPU?";
  CHECK(coo.sorted) << "[BUG] Rowwise Top-K is only supported on sorted matrix";

  const int64_t nnz = coo.row->shape[0];
  if (nnz == 0)
    // do nothing; assume that weights is also an empty array
    return std::make_pair(coo, weights);

  const IdType *coo_row_data = static_cast<IdType *>(coo.row->data);
  const IdType *coo_col_data = static_cast<IdType *>(coo.col->data);
  const DType *coo_data = COOHasData(coo) ? static_cast<IdType *>(coo.data->data) : nullptr;
  const WType *weights_data = static_cast<WType *>(weights->data);

  std::vector<IdType> new_rows, new_cols;
  std::vector<DType> new_data;
  std::vector<WType> new_weights;

  IdType prev_row, i;
  std::vector<IdType> indices(nnz);
  std::iota(indices.begin(), indices.end(), 0);
  for (IdType j = 0; j < nnz; ++j) {
    IdType curr_row = coo_row_data[j];

    if (j == 0) {
      prev_row = curr_row;
      i = j;
      continue;
    }

    if (prev_row != curr_row) {
      PushTopK<XPU, IdType, DType, WType>(
          prev_row, coo_col_data, coo_data, weights_data, K, smallest, i, j, &indices,
          &new_rows, &new_cols, &new_data, &new_weights);
      prev_row = curr_row;
      i = j;
    }
  }
  PushTopK<XPU, IdType, DType, WType>(
      prev_row, coo_col_data, coo_data, weights_data, K, smallest, i, nnz, &indices,
      &new_rows, &new_cols, &new_data, &new_weights);

  const auto new_coo = COOMatrix{
      coo.num_rows,
      coo.num_cols,
      NDArray::FromVector(new_rows),
      NDArray::FromVector(new_cols),
      NDArray::FromVector(new_data),
      true};
  return std::make_pair(new_coo, NDArray::FromVector(new_weights));
}

template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int32_t, int32_t, int32_t>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int32_t, int32_t, int64_t>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int32_t, int32_t, float>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int32_t, int32_t, double>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int64_t, int64_t, int32_t>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int64_t, int64_t, int64_t>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int64_t, int64_t, float>(
    COOMatrix, int64_t, NDArray, bool);
template std::pair<COOMatrix, NDArray> COORowwiseTopKNonZero<kDLCPU, int64_t, int64_t, double>(
    COOMatrix, int64_t, NDArray, bool);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
