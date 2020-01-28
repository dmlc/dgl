/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/array.h
 * \brief Array types and common array operations required by DGL.
 *
 * Note that this is not meant for a full support of array library such as ATen.
 * Only a limited set of operators required by DGL are implemented.
 */
#ifndef DGL_ARRAY_H_
#define DGL_ARRAY_H_

#include <dgl/runtime/ndarray.h>
#include <algorithm>
#include <vector>
#include <tuple>
#include <utility>

namespace dgl {

typedef uint64_t dgl_id_t;
typedef uint64_t dgl_type_t;

using dgl::runtime::NDArray;

typedef NDArray IdArray;
typedef NDArray DegreeArray;
typedef NDArray BoolArray;
typedef NDArray IntArray;
typedef NDArray FloatArray;
typedef NDArray TypeArray;

namespace aten {

//////////////////////////////////////////////////////////////////////
// ID array
//////////////////////////////////////////////////////////////////////

/*!
 * \brief Create a new id array with given length
 * \param length The array length
 * \param ctx The array context
 * \param nbits The number of integer bits
 * \return id array
 */
IdArray NewIdArray(int64_t length,
                   DLContext ctx = DLContext{kDLCPU, 0},
                   uint8_t nbits = 64);

/*!
 * \brief Create a new id array using the given vector data
 * \param vec The vector data
 * \param nbits The integer bits of the returned array
 * \param ctx The array context
 * \return the id array
 */
template <typename T>
IdArray VecToIdArray(const std::vector<T>& vec,
                     uint8_t nbits = 64,
                     DLContext ctx = DLContext{kDLCPU, 0});

/*!
 * \brief Create a new NDArray using the given vector data
 * \tparam T Type of vector.  The dtype of the returned array is selected accordingly.
 * \param vec The vector data
 * \param ctx The array context
 * \return the NDArray
 */
template <typename T>
NDArray VecToNDArray(const std::vector<T>& vec,
                     DLContext ctx = DLContext{kDLCPU, 0});

/*!
 * \brief Return an array representing a 1D range.
 * \param low Lower bound (inclusive).
 * \param high Higher bound (exclusive).
 * \param nbits result array's bits (32 or 64)
 * \param ctx Device context
 * \return range array
 */
IdArray Range(int64_t low, int64_t high, uint8_t nbits, DLContext ctx);

/*!
 * \brief Return an array full of the given value
 * \param val The value to fill.
 * \param length Number of elements.
 * \param nbits result array's bits (32 or 64)
 * \param ctx Device context
 * \return the result array
 */
IdArray Full(int64_t val, int64_t length, uint8_t nbits, DLContext ctx);

/*! \brief Create a deep copy of the given array */
IdArray Clone(IdArray arr);

/*! \brief Convert the idarray to the given bit width */
IdArray AsNumBits(IdArray arr, uint8_t bits);

/*! \brief Arithmetic functions */
IdArray Add(IdArray lhs, IdArray rhs);
IdArray Sub(IdArray lhs, IdArray rhs);
IdArray Mul(IdArray lhs, IdArray rhs);
IdArray Div(IdArray lhs, IdArray rhs);

IdArray Add(IdArray lhs, dgl_id_t rhs);
IdArray Sub(IdArray lhs, dgl_id_t rhs);
IdArray Mul(IdArray lhs, dgl_id_t rhs);
IdArray Div(IdArray lhs, dgl_id_t rhs);

IdArray Add(dgl_id_t lhs, IdArray rhs);
IdArray Sub(dgl_id_t lhs, IdArray rhs);
IdArray Mul(dgl_id_t lhs, IdArray rhs);
IdArray Div(dgl_id_t lhs, IdArray rhs);

BoolArray LT(IdArray lhs, dgl_id_t rhs);

/*! \brief Stack two arrays (of len L) into a 2*L length array */
IdArray HStack(IdArray arr1, IdArray arr2);

/*!
 * \brief Return the data under the index. In numpy notation, A[I]
 * \tparam ValueType The type of return value.
 */
template<typename ValueType>
ValueType IndexSelect(NDArray array, uint64_t index);
NDArray IndexSelect(NDArray array, IdArray index);

/*!
 * \brief Relabel the given ids to consecutive ids.
 *
 * Relabeling is done inplace. The mapping is created from the union
 * of the give arrays.
 *
 * \param arrays The id arrays to relabel.
 * \return mapping array M from new id to old id.
 */
IdArray Relabel_(const std::vector<IdArray>& arrays);

/*!\brief Return whether the array is a valid 1D int array*/
inline bool IsValidIdArray(const dgl::runtime::NDArray& arr) {
  return arr->ndim == 1 && arr->dtype.code == kDLInt;
}

/*!
 * \brief Pack a padded tensor to a triplet of concatenated tensor, their lengths,
 * and offsets.
 */
template<typename ValueType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, ValueType pad_value);

/*!
 * \brief Concat array[i, 0:lengths[i]] for every i.  Returns the concatenated array
 * and offsets in the concatenated array for each original slice.
 */
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths);

//////////////////////////////////////////////////////////////////////
// Sparse matrix
//////////////////////////////////////////////////////////////////////

/*!
 * \brief Plain CSR matrix
 *
 * The column indices are 0-based and are not necessarily sorted.
 *
 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 */
struct CSRMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows, num_cols;
  /*! \brief CSR index arrays */
  runtime::NDArray indptr, indices;
  /*! \brief data array, could be empty. */
  runtime::NDArray data;
  /*! \brief indicate that the edges are stored in the sorted order. */
  bool sorted;
};

/*!
 * \brief Plain COO structure
 * 
 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 *
 * We call a COO matrix is *coalesced* if its row index is sorted.
 */
struct COOMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows, num_cols;
  /*! \brief COO index arrays */
  runtime::NDArray row, col;
  /*! \brief data array, could be empty. */
  runtime::NDArray data;
};

///////////////////////// CSR routines //////////////////////////

/*! \brief Return true if the value (row, col) is non-zero */
bool CSRIsNonZero(CSRMatrix , int64_t row, int64_t col);
/*!
 * \brief Batched implementation of CSRIsNonZero.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
runtime::NDArray CSRIsNonZero(CSRMatrix, runtime::NDArray row, runtime::NDArray col);

/*! \brief Return the nnz of the given row */
int64_t CSRGetRowNNZ(CSRMatrix , int64_t row);
runtime::NDArray CSRGetRowNNZ(CSRMatrix , runtime::NDArray row);

/*! \brief Return the column index array of the given row */
runtime::NDArray CSRGetRowColumnIndices(CSRMatrix , int64_t row);

/*! \brief Return the data array of the given row */
runtime::NDArray CSRGetRowData(CSRMatrix , int64_t row);

/* \brief Get data. The return type is an ndarray due to possible duplicate entries. */
runtime::NDArray CSRGetData(CSRMatrix , int64_t row, int64_t col);
/*!
 * \brief Batched implementation of CSRGetData.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */

runtime::NDArray CSRGetData(CSRMatrix, runtime::NDArray rows, runtime::NDArray cols);

/*!
 * \brief Get the data and the row,col indices for each returned entries.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    CSRMatrix , runtime::NDArray rows, runtime::NDArray cols);

/*! \brief Return a transposed CSR matrix */
CSRMatrix CSRTranspose(CSRMatrix csr);

/*!
 * \brief Convert CSR matrix to COO matrix.
 * \param csr Input csr matrix
 * \param data_as_order If true, the data array in the input csr matrix contains the order
 *                      by which the resulting COO tuples are stored. In this case, the
 *                      data array of the resulting COO matrix will be empty because it
 *                      is essentially a consecutive range.
 * \return a coo matrix
 */
COOMatrix CSRToCOO(CSRMatrix csr, bool data_as_order);

/*!
 * \brief Slice rows of the given matrix and return.
 * \param csr CSR matrix
 * \param start Start row id (inclusive)
 * \param end End row id (exclusive)
 *
 * Examples:
 * num_rows = 4
 * num_cols = 4
 * indptr = [0, 2, 3, 3, 5]
 * indices = [1, 0, 2, 3, 1]
 *
 *  After CSRSliceRows(csr, 1, 3)
 *
 * num_rows = 2
 * num_cols = 4
 * indptr = [0, 1, 1]
 * indices = [2]
 */
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end);
CSRMatrix CSRSliceRows(CSRMatrix csr, runtime::NDArray rows);

/*!
 * \brief Get the submatrix specified by the row and col ids.
 *
 * In numpy notation, given matrix M, row index array I, col index array J
 * This function returns the submatrix M[I, J].
 *
 * \param csr The input csr matrix
 * \param rows The row index to select
 * \param cols The col index to select
 * \return submatrix
 */
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

/*! \return True if the matrix has duplicate entries */
bool CSRHasDuplicate(CSRMatrix csr);

/*! Sort the columns in each row in the ascending order. */
void CSRSort(CSRMatrix csr);

///////////////////////// COO routines //////////////////////////

/*! \return True if the matrix has duplicate entries */
bool COOHasDuplicate(COOMatrix coo);

/*!
 * \brief Convert COO matrix to CSR matrix.
 *
 * If the input COO matrix does not have data array, the data array of
 * the result CSR matrix stores a shuffle index for how the entries
 * will be reordered in CSR. The i^th entry in the result CSR corresponds
 * to the CSR.data[i] th entry in the input COO.
 */
CSRMatrix COOToCSR(COOMatrix coo);

// inline implementations
template <typename T>
IdArray VecToIdArray(const std::vector<T>& vec,
                     uint8_t nbits,
                     DLContext ctx) {
  IdArray ret = NewIdArray(vec.size(), DLContext{kDLCPU, 0}, nbits);
  if (nbits == 32) {
    std::copy(vec.begin(), vec.end(), static_cast<int32_t*>(ret->data));
  } else if (nbits == 64) {
    std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(ret->data));
  } else {
    LOG(FATAL) << "Only int32 or int64 is supported.";
  }
  return ret.CopyTo(ctx);
}

template<typename T>
NDArray VecToNDArray(const std::vector<T> &vec, DLDataType dtype, DLContext ctx) {
  int64_t size = static_cast<int64_t>(vec.size());
  NDArray ret = NDArray::Empty({size}, dtype, DLContext{kDLCPU, 0});
  std::copy(vec.begin(), vec.end(), static_cast<T *>(ret->data));
  return ret.CopyTo(ctx);
}

///////////////////////// Dispatchers //////////////////////////

/*
 * Dispatch according to device:
 *
 * ATEN_XPU_SWITCH(array->ctx.device_type, XPU, {
 *   // Now XPU is a placeholder for array->ctx.device_type
 *   DeviceSpecificImplementation<XPU>(...);
 * });
 */
#define ATEN_XPU_SWITCH(val, XPU, ...) do {                     \
  if ((val) == kDLCPU) {                                        \
    constexpr auto XPU = kDLCPU;                                \
    {__VA_ARGS__}                                               \
  } else {                                                      \
    LOG(FATAL) << "Device type: " << (val) << " is not supported.";  \
  }                                                             \
} while (0)

/*
 * Dispatch according to integral type (either int32 or int64):
 *
 * ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
 *   // Now IdType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_ID_TYPE_SWITCH(val, IdType, ...) do {            \
  CHECK_EQ((val).code, kDLInt) << "ID must be integer type";  \
  if ((val).bits == 32) {                                     \
    typedef int32_t IdType;                                   \
    {__VA_ARGS__}                                             \
  } else if ((val).bits == 64) {                              \
    typedef int64_t IdType;                                   \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    LOG(FATAL) << "ID can only be int32 or int64";            \
  }                                                           \
} while (0)

/*
 * Dispatch according to float type (either float32 or float64):
 *
 * ATEN_ID_TYPE_SWITCH(array->dtype, FloatType, {
 *   // Now FloatType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   FloatType *data = static_cast<FloatType *>(array->data);
 * });
 */
#define ATEN_FLOAT_TYPE_SWITCH(val, FloatType, val_name, ...) do {  \
  CHECK_EQ((val).code, kDLFloat)                              \
    << (val_name) << " must be float type";                   \
  if ((val).bits == 32) {                                     \
    typedef float FloatType;                                  \
    {__VA_ARGS__}                                             \
  } else if ((val).bits == 64) {                              \
    typedef double FloatType;                                 \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    LOG(FATAL) << (val_name) << " can only be float32 or float64";  \
  }                                                           \
} while (0)

/*
 * Dispatch according to data type (int32, int64, float32 or float64):
 *
 * ATEN_ID_TYPE_SWITCH(array->dtype, DType, {
 *   // Now DType is the type corresponding to data type in array.
 *   // For instance, one can do this for a CPU array:
 *   DType *data = static_cast<DType *>(array->data);
 * });
 */
#define ATEN_DTYPE_SWITCH(val, DType, val_name, ...) do {     \
  if ((val).code == kDLInt && (val).bits == 32) {             \
    typedef int32_t DType;                                    \
    {__VA_ARGS__}                                             \
  } else if ((val).code == kDLInt && (val).bits == 64) {      \
    typedef int64_t DType;                                    \
    {__VA_ARGS__}                                             \
  } else if ((val).code == kDLFloat && (val).bits == 32) {    \
    typedef float DType;                                      \
    {__VA_ARGS__}                                             \
  } else if ((val).code == kDLFloat && (val).bits == 64) {    \
    typedef double DType;                                     \
    {__VA_ARGS__}                                             \
  } else {                                                    \
    LOG(FATAL) << (val_name) << " can only be int32, int64, float32 or float64"; \
  }                                                           \
} while (0)

/*
 * Dispatch according to integral type of CSR graphs.
 * Identical to ATEN_ID_TYPE_SWITCH except for a different error message.
 */
#define ATEN_CSR_DTYPE_SWITCH(val, DType, ...) do {         \
  if ((val).code == kDLInt && (val).bits == 32) {           \
    typedef int32_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else if ((val).code == kDLInt && (val).bits == 64) {    \
    typedef int64_t DType;                                  \
    {__VA_ARGS__}                                           \
  } else {                                                  \
    LOG(FATAL) << "CSR matrix data can only be int32 or int64";  \
  }                                                         \
} while (0)

// Macro to dispatch according to device context, index type and data type
// TODO(minjie): In our current use cases, data type and id type are the
//   same. For example, data array is used to store edge ids.
#define ATEN_CSR_SWITCH(csr, XPU, IdType, DType, ...)       \
  ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, {       \
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {        \
      typedef IdType DType;                                 \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context and index type
#define ATEN_CSR_IDX_SWITCH(csr, XPU, IdType, ...)          \
  ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, {       \
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {        \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context, index type and data type
// TODO(minjie): In our current use cases, data type and id type are the
//   same. For example, data array is used to store edge ids.
#define ATEN_COO_SWITCH(coo, XPU, IdType, DType, ...)       \
  ATEN_XPU_SWITCH(coo.row->ctx.device_type, XPU, {          \
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {           \
      typedef IdType DType;                                 \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context and index type
#define ATEN_COO_IDX_SWITCH(coo, XPU, IdType, ...)          \
  ATEN_XPU_SWITCH(coo.row->ctx.device_type, XPU, {          \
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {           \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

///////////////////////// Array checks //////////////////////////

#define IS_INT32(a)  \
  ((a)->dtype.code == kDLInt && (a)->dtype.bits == 32)
#define IS_INT64(a)  \
  ((a)->dtype.code == kDLInt && (a)->dtype.bits == 64)
#define IS_FLOAT32(a)  \
  ((a)->dtype.code == kDLFloat && (a)->dtype.bits == 32)
#define IS_FLOAT64(a)  \
  ((a)->dtype.code == kDLFloat && (a)->dtype.bits == 64)

#define CHECK_IF(cond, prop, value_name, dtype_name) \
  CHECK(cond) << "Expecting " << (prop) << " of " << (value_name) << " to be " << (dtype_name)

#define CHECK_INT32(value, value_name) \
  CHECK_IF(IS_INT32(value), "dtype", value_name, "int32")
#define CHECK_INT64(value, value_name) \
  CHECK_IF(IS_INT64(value), "dtype", value_name, "int64")
#define CHECK_INT(value, value_name) \
  CHECK_IF(IS_INT32(value) || IS_INT64(value), "dtype", value_name, "int32 or int64")
#define CHECK_FLOAT32(value, value_name) \
  CHECK_IF(IS_FLOAT32(value), "dtype", value_name, "float32")
#define CHECK_FLOAT64(value, value_name) \
  CHECK_IF(IS_FLOAT64(value), "dtype", value_name, "float64")
#define CHECK_FLOAT(value, value_name) \
  CHECK_IF(IS_FLOAT32(value) || IS_FLOAT64(value), "dtype", value_name, "float32 or float64")

#define CHECK_NDIM(value, _ndim, value_name) \
  CHECK_IF((value)->ndim == (_ndim), "ndim", value_name, _ndim)

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_H_
