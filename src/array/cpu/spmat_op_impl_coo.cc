/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/spmat_op_impl.cc
 * \brief Sparse matrix operator CPU implementation
 */
#include <dgl/array.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {
namespace {
/*!
 * \brief A hashmap that maps each ids in the given array to new ids starting from zero.
 */
template <typename IdType>
class IdHashMap {
 public:
  // Construct the hashmap using the given id arrays.
  // The id array could contain duplicates.
  explicit IdHashMap(IdArray ids): filter_(kFilterSize, false) {
    const IdType* ids_data = static_cast<IdType*>(ids->data);
    const int64_t len = ids->shape[0];
    IdType newid = 0;
    for (int64_t i = 0; i < len; ++i) {
      const IdType id = ids_data[i];
      if (!Contains(id)) {
        oldv2newv_[id] = newid++;
        filter_[id & kFilterMask] = true;
      }
    }
  }

  // Return true if the given id is contained in this hashmap.
  bool Contains(IdType id) const {
    return filter_[id & kFilterMask] && oldv2newv_.count(id);
  }

  // Return the new id of the given id. If the given id is not contained
  // in the hash map, returns the default_val instead.
  IdType Map(IdType id, IdType default_val) const {
    if (filter_[id & kFilterMask]) {
      auto it = oldv2newv_.find(id);
      return (it == oldv2newv_.end()) ? default_val : it->second;
    } else {
      return default_val;
    }
  }

 private:
  static constexpr int32_t kFilterMask = 0xFFFFFF;
  static constexpr int32_t kFilterSize = kFilterMask + 1;
  // This bitmap is used as a bloom filter to remove some lookups.
  // Hashtable is very slow. Using bloom filter can significantly speed up lookups.
  std::vector<bool> filter_;
  // The hashmap from old vid to new vid
  std::unordered_map<IdType, IdType> oldv2newv_;
};

struct PairHash {
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

inline bool CSRHasData(CSRMatrix csr) {
  return csr.data.defined();
}

inline bool COOHasData(COOMatrix csr) {
  return csr.data.defined();
}
}  // namespace

///////////////////////////// COOIsNonZero /////////////////////////////

template <DLDeviceType XPU, typename IdType>
bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col) {
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < coo.num_cols) << "Invalid col index: " << col;
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row && coo_col_data[i] == col)
      return true;
  }
  return false;
}

template bool COOIsNonZero<kDLCPU, int32_t>(COOMatrix, int64_t, int64_t);
template bool COOIsNonZero<kDLCPU, int64_t>(COOMatrix, int64_t, int64_t);

template <DLDeviceType XPU, typename IdType>
NDArray COOIsNonZero(COOMatrix coo, NDArray row, NDArray col) {
  const auto rowlen = row->shape[0];
  const auto collen = col->shape[0];
  const auto rstlen = std::max(rowlen, collen);
  NDArray rst = NDArray::Empty({rstlen}, row->dtype, row->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
  const IdType* row_data = static_cast<IdType*>(row->data);
  const IdType* col_data = static_cast<IdType*>(col->data);
  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const int64_t kmax = std::max(rowlen, collen);
#pragma omp parallel for
  for (int64_t k = 0; k < kmax; ++k) {
    int64_t i = row_stride * k;
    int64_t j = col_stride * k;
    rst_data[k] = COOIsNonZero<XPU, IdType>(coo, row_data[i], col_data[j])? 1 : 0;
  }
  return rst;
}

template NDArray COOIsNonZero<kDLCPU, int32_t>(COOMatrix, NDArray, NDArray);
template NDArray COOIsNonZero<kDLCPU, int64_t>(COOMatrix, NDArray, NDArray);

///////////////////////////// COOHasDuplicate /////////////////////////////

template <DLDeviceType XPU, typename IdType>
bool COOHasDuplicate(COOMatrix coo) {
  std::unordered_set<std::pair<IdType, IdType>, PairHash> hashmap;
  const IdType* src_data = static_cast<IdType*>(coo.row->data);
  const IdType* dst_data = static_cast<IdType*>(coo.col->data);
  const auto nnz = coo.row->shape[0];
  for (IdType eid = 0; eid < nnz; ++eid) {
    const auto& p = std::make_pair(src_data[eid], dst_data[eid]);
    if (hashmap.count(p)) {
      return true;
    } else {
      hashmap.insert(p);
    }
  }
  return false;
}

template bool COOHasDuplicate<kDLCPU, int32_t>(COOMatrix coo);
template bool COOHasDuplicate<kDLCPU, int64_t>(COOMatrix coo);

///////////////////////////// COOGetRowNNZ /////////////////////////////

template <DLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row, bool transpose) {
  int64_t num_rows = transpose ? coo.num_cols : coo.num_rows;
  const std::string name = transpose ? "column" : "row";
  CHECK(row >= 0 && row < num_rows) << "Invalid " << name << " index: " << row;

  NDArray coo_row = transpose ? coo.col : coo.row;
  const IdType* coo_row_data = static_cast<IdType*>(coo_row->data);

  int64_t result = 0;
  for (int64_t i = 0; i < coo_row->shape[0]; ++i) {
    if (coo_row_data[i] == row)
      ++result;
  }
  return result;
}

template int64_t COOGetRowNNZ<kDLCPU, int32_t>(COOMatrix, int64_t, bool);
template int64_t COOGetRowNNZ<kDLCPU, int64_t>(COOMatrix, int64_t, bool);

template <DLDeviceType XPU, typename IdType>
NDArray COOGetRowNNZ(COOMatrix coo, NDArray rows, bool transpose) {
  const auto len = rows->shape[0];
  const IdType* vid_data = static_cast<IdType*>(rows->data);
  NDArray rst = NDArray::Empty({len}, rows->dtype, rows->ctx);
  IdType* rst_data = static_cast<IdType*>(rst->data);
#pragma omp parallel for
  for (int64_t i = 0; i < len; ++i)
    rst_data[i] = COOGetRowNNZ<XPU, IdType>(coo, vid_data[i], transpose);
  return rst;
}

template NDArray COOGetRowNNZ<kDLCPU, int32_t>(COOMatrix, NDArray, bool);
template NDArray COOGetRowNNZ<kDLCPU, int64_t>(COOMatrix, NDArray, bool);

///////////////////////////// COOGetRowDataAndIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
std::pair<NDArray, NDArray> COOGetRowDataAndIndices(
    COOMatrix coo, int64_t row, bool transpose) {
  int64_t num_rows = transpose ? coo.num_cols : coo.num_rows;
  const std::string name = transpose ? "column" : "row";
  CHECK(row >= 0 && row < num_rows) << "Invalid " << name << " index: " << row;

  NDArray coo_row = transpose ? coo.col : coo.row;
  NDArray coo_col = transpose ? coo.row : coo.col;
  const IdType* coo_row_data = static_cast<IdType*>(coo_row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo_col->data);
  const DType* coo_data = static_cast<DType*>(coo.data->data);

  std::vector<IdType> indices;
  std::vector<DType> data;

  for (int64_t i = 0; i < coo_row->shape[0]; ++i) {
    if (coo_row_data[i] == row) {
      indices.push_back(coo_col_data[i]);
      data.push_back(coo_data[i]);
    }
  }

  return std::make_pair(NDArray::FromVector(data), NDArray::FromVector(indices));
}

template std::pair<NDArray, NDArray>
COOGetRowDataAndIndices<kDLCPU, int32_t, int32_t>(COOMatrix, int64_t, bool);
template std::pair<NDArray, NDArray>
COOGetRowDataAndIndices<kDLCPU, int64_t, int64_t>(COOMatrix, int64_t, bool);

///////////////////////////// CSRGetData /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray COOGetData(COOMatrix coo, int64_t row, int64_t col) {
  CHECK(COOHasData(coo)) << "missing data array";
  // TODO(minjie): use more efficient binary search when the column indices is sorted
  CHECK(row >= 0 && row < coo.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < coo.num_cols) << "Invalid col index: " << col;
  std::vector<DType> ret_vec;
  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const DType* data = static_cast<DType*>(coo.data->data);
  for (IdType i = 0; i < coo.row->shape[0]; ++i) {
    if (coo_row_data[i] == row && coo_col_data[i] == col)
      ret_vec.push_back(data[i]);
  }
  return NDArray::FromVector(ret_vec, coo.data->dtype, coo.data->ctx);
}

template NDArray COOGetData<kDLCPU, int32_t, int32_t>(COOMatrix, int64_t, int64_t);
template NDArray COOGetData<kDLCPU, int64_t, int64_t>(COOMatrix, int64_t, int64_t);

///////////////////////////// COOGetDataAndIndices /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
std::vector<NDArray> COOGetDataAndIndices(
    COOMatrix coo, NDArray rows, NDArray cols) {
  CHECK(COOHasData(coo)) << "missing data array";
  // TODO(minjie): more efficient implementation for matrix without duplicate entries
  // TODO(minjie): more efficient implementation for sorted column index
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
    << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType* row_data = static_cast<IdType*>(rows->data);
  const IdType* col_data = static_cast<IdType*>(cols->data);

  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const DType* data = static_cast<DType*>(coo.data->data);

  std::vector<IdType> ret_rows, ret_cols;
  std::vector<DType> ret_data;

  for (int64_t i = 0, j = 0; i < rowlen && j < collen; i += row_stride, j += col_stride) {
    const IdType row_id = row_data[i], col_id = col_data[j];
    CHECK(row_id >= 0 && row_id < coo.num_rows) << "Invalid row index: " << row_id;
    CHECK(col_id >= 0 && col_id < coo.num_cols) << "Invalid col index: " << col_id;
    for (int64_t k = 0; k < coo.row->shape[0]; ++k) {
      if (coo_row_data[k] == row_id && coo_col_data[k] == col_id) {
        ret_rows.push_back(row_id);
        ret_cols.push_back(col_id);
        ret_data.push_back(data[k]);
      }
    }
  }

  return {NDArray::FromVector(ret_rows),
          NDArray::FromVector(ret_cols),
          NDArray::FromVector(ret_data)};
}

template std::vector<NDArray> COOGetDataAndIndices<kDLCPU, int32_t, int32_t>(
    COOMatrix coo, NDArray rows, NDArray cols);
template std::vector<NDArray> COOGetDataAndIndices<kDLCPU, int64_t, int64_t>(
    COOMatrix coo, NDArray rows, NDArray cols);

///////////////////////////// COOTranspose /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix COOTranspose(COOMatrix coo) {
  return COOMatrix{coo.num_cols, coo.num_rows, coo.col, coo.row, coo.data};
}

template COOMatrix COOTranspose<kDLCPU, int32_t, int32_t>(COOMatrix coo);
template COOMatrix COOTranspose<kDLCPU, int64_t, int64_t>(COOMatrix coo);

///////////////////////////// COOToCSR /////////////////////////////

// complexity: time O(NNZ), space O(1)
template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix COOToCSR(COOMatrix coo) {
  const int64_t N = coo.num_rows;
  const int64_t NNZ = coo.row->shape[0];
  const IdType* row_data = static_cast<IdType*>(coo.row->data);
  const IdType* col_data = static_cast<IdType*>(coo.col->data);
  NDArray ret_indptr = NDArray::Empty({N + 1}, coo.row->dtype, coo.row->ctx);
  NDArray ret_indices = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  NDArray ret_data;
  if (COOHasData(coo)) {
    ret_data = NDArray::Empty({NNZ}, coo.data->dtype, coo.data->ctx);
  } else {
    // if no data array in the input coo, the return data array is a shuffle index.
    ret_data = NDArray::Empty({NNZ}, coo.row->dtype, coo.row->ctx);
  }

  IdType* Bp = static_cast<IdType*>(ret_indptr->data);
  IdType* Bi = static_cast<IdType*>(ret_indices->data);

  std::fill(Bp, Bp + N, 0);

  for (int64_t i = 0; i < NNZ; ++i) {
    Bp[row_data[i]]++;
  }

  // cumsum
  for (int64_t i = 0, cumsum = 0; i < N; ++i) {
    const IdType temp = Bp[i];
    Bp[i] = cumsum;
    cumsum += temp;
  }
  Bp[N] = NNZ;

  for (int64_t i = 0; i < NNZ; ++i) {
    const IdType r = row_data[i];
    Bi[Bp[r]] = col_data[i];
    if (COOHasData(coo)) {
      const DType* data = static_cast<DType*>(coo.data->data);
      DType* Bx = static_cast<DType*>(ret_data->data);
      Bx[Bp[r]] = data[i];
    } else {
      IdType* Bx = static_cast<IdType*>(ret_data->data);
      Bx[Bp[r]] = i;
    }
    Bp[r]++;
  }

  // correct the indptr
  for (int64_t i = 0, last = 0; i <= N; ++i) {
    IdType temp = Bp[i];
    Bp[i] = last;
    last = temp;
  }

  return CSRMatrix{coo.num_rows, coo.num_cols, ret_indptr, ret_indices, ret_data};
}

template CSRMatrix COOToCSR<kDLCPU, int32_t, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDLCPU, int64_t, int64_t>(COOMatrix coo);

///////////////////////////// COOSliceRows /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end, bool transpose) {
  CHECK(COOHasData(coo)) << "missing data array.";
  int64_t num_rows = transpose ? coo.num_cols : coo.num_rows;
  const std::string name = transpose ? "column" : "row";
  CHECK(start >= 0 && start < num_rows) << "Invalid start " << name << " " << start;
  CHECK(end > 0 && end <= num_rows) << "Invalid end " << name << " " << end;

  NDArray coo_row = transpose ? coo.col : coo.row;
  NDArray coo_col = transpose ? coo.row : coo.col;

  const IdType* coo_row_data = static_cast<IdType*>(coo_row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo_col->data);
  const DType* coo_data = static_cast<DType*>(coo.data->data);

  std::vector<IdType> ret_row, ret_col;
  std::vector<DType> ret_data;

  for (int64_t i = 0; i < coo_row->shape[0]; ++i) {
    const IdType row_id = coo_row_data[i];
    const IdType col_id = coo_col_data[i];
    if (row_id < end && row_id >= start) {
      ret_row.push_back(row_id - start);
      ret_col.push_back(col_id);
      ret_data.push_back(coo_data[i]);
    }
  }
  if (!transpose)
    return COOMatrix{
      end - start,
      coo.num_cols,
      NDArray::FromVector(ret_row, coo_row->dtype, coo_row->ctx),
      NDArray::FromVector(ret_col, coo_col->dtype, coo_col->ctx),
      NDArray::FromVector(ret_data, coo.data->dtype, coo.data->ctx)};
  else
    return COOMatrix{
      coo.num_rows,
      end - start,
      NDArray::FromVector(ret_col, coo_col->dtype, coo_col->ctx),
      NDArray::FromVector(ret_row, coo_row->dtype, coo_row->ctx),
      NDArray::FromVector(ret_data, coo.data->dtype, coo.data->ctx)};
}

template COOMatrix COOSliceRows<kDLCPU, int32_t, int32_t>(COOMatrix, int64_t, int64_t, bool);
template COOMatrix COOSliceRows<kDLCPU, int64_t, int64_t>(COOMatrix, int64_t, int64_t, bool);

template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix COOSliceRows(COOMatrix coo, NDArray rows, bool transpose) {
  CHECK(COOHasData(coo)) << "missing data array.";

  NDArray coo_row = transpose ? coo.col : coo.row;
  NDArray coo_col = transpose ? coo.row : coo.col;

  const IdType* coo_row_data = static_cast<IdType*>(coo_row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo_col->data);
  const DType* coo_data = static_cast<DType*>(coo.data->data);

  std::vector<IdType> ret_row, ret_col;
  std::vector<DType> ret_data;

  IdHashMap<IdType> hashmap(rows);

  for (int64_t i = 0; i < coo_row->shape[0]; ++i) {
    const IdType row_id = coo_row_data[i];
    const IdType col_id = coo_col_data[i];
    const IdType mapped_row_id = hashmap.Map(row_id, -1);
    if (mapped_row_id != -1) {
      ret_row.push_back(mapped_row_id);
      ret_col.push_back(col_id);
      ret_data.push_back(coo_data[i]);
    }
  }

  if (!transpose)
    return COOMatrix{
      rows->shape[0],
      coo.num_cols,
      NDArray::FromVector(ret_row, coo_row->dtype, coo_row->ctx),
      NDArray::FromVector(ret_col, coo_col->dtype, coo_col->ctx),
      NDArray::FromVector(ret_data, coo.data->dtype, coo.data->ctx)};
  else
    return COOMatrix{
      coo.num_rows,
      rows->shape[0],
      NDArray::FromVector(ret_col, coo_col->dtype, coo_col->ctx),
      NDArray::FromVector(ret_row, coo_row->dtype, coo_row->ctx),
      NDArray::FromVector(ret_data, coo.data->dtype, coo.data->ctx)};
}

template COOMatrix COOSliceRows<kDLCPU, int32_t, int32_t>(COOMatrix , NDArray, bool);
template COOMatrix COOSliceRows<kDLCPU, int64_t, int64_t>(COOMatrix , NDArray, bool);

///////////////////////////// CSRSliceMatrix /////////////////////////////

template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix COOSliceMatrix(COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols) {
  CHECK(COOHasData(coo)) << "missing data array.";

  const IdType* coo_row_data = static_cast<IdType*>(coo.row->data);
  const IdType* coo_col_data = static_cast<IdType*>(coo.col->data);
  const DType* coo_data = static_cast<DType*>(coo.data->data);

  IdHashMap<IdType> row_map(rows), col_map(cols);

  std::vector<IdType> ret_row, ret_col;
  std::vector<DType> ret_data;

  for (int64_t i = 0; i < coo.row->shape[0]; ++i) {
    const IdType row_id = coo_row_data[i];
    const IdType col_id = coo_col_data[i];
    const IdType mapped_row_id = row_map.Map(row_id, -1);
    if (mapped_row_id != -1) {
      const IdType mapped_col_id = col_map.Map(col_id, -1);
      if (mapped_col_id != -1) {
        ret_row.push_back(mapped_row_id);
        ret_col.push_back(mapped_col_id);
        ret_data.push_back(coo_data[i]);
      }
    }
  }

  return COOMatrix{
    rows->shape[0],
    cols->shape[0],
    NDArray::FromVector(ret_row, coo.row->dtype, coo.row->ctx),
    NDArray::FromVector(ret_col, coo.col->dtype, coo.col->ctx),
    NDArray::FromVector(ret_data, coo.data->dtype, coo.data->ctx)};
}

template COOMatrix COOSliceMatrix<kDLCPU, int32_t, int32_t>(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);
template COOMatrix COOSliceMatrix<kDLCPU, int64_t, int64_t>(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
