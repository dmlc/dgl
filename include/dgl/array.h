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
#include <dmlc/io.h>
#include <dmlc/serializer.h>
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
 * \brief Packs a tensor containing padded sequences of variable length.
 *
 * Similar to \c pack_padded_sequence in PyTorch, except that
 *
 * 1. The length for each sequence (before padding) is inferred as the number
 *    of elements before the first occurrence of \c pad_value.
 * 2. It does not sort the sequences by length.
 * 3. Along with the tensor containing the packed sequence, it returns both the
 *    length, as well as the offsets to the packed tensor, of each sequence.
 *
 * \param array The tensor containing sequences padded to the same length
 * \param pad_value The padding value
 * \return A triplet of packed tensor, the length tensor, and the offset tensor
 *
 * \note Example: consider the following array with padding value -1:
 *
 * <code>
 *     [[1, 2, -1, -1],
 *      [3, 4,  5, -1]]
 * </code>
 *
 * The packed tensor would be [1, 2, 3, 4, 5].
 *
 * The length tensor would be [2, 3], i.e. the length of each sequence before padding.
 *
 * The offset tensor would be [0, 2], i.e. the offset to the packed tensor for each
 * sequence (before padding)
 */
template<typename ValueType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, ValueType pad_value);

/*!
 * \brief Batch-slice a 1D or 2D array, and then pack the list of sliced arrays
 * by concatenation.
 *
 * If a 2D array is given, then the function is equivalent to:
 *
 * <code>
 *     def ConcatSlices(array, lengths):
 *         slices = [array[i, :l] for i, l in enumerate(lengths)]
 *         packed = np.concatenate(slices)
 *         offsets = np.cumsum([0] + lengths[:-1])
 *         return packed, offsets
 * </code>
 *
 * If a 1D array is given, then the function is equivalent to
 *
 * <code>
 *     def ConcatSlices(array, lengths):
 *         slices = [array[:l] for l in lengths]
 *         packed = np.concatenate(slices)
 *         offsets = np.cumsum([0] + lengths[:-1])
 *         return packed, offsets
 * </code>
 *
 * \param array A 1D or 2D tensor for slicing
 * \param lengths A 1D tensor indicating the number of elements to slice
 * \return The tensor with packed slices along with the offsets.
 */
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths);

//////////////////////////////////////////////////////////////////////
// Sparse matrix
//////////////////////////////////////////////////////////////////////

/*!
 * \brief Plain CSR matrix
 *
 * The column indices are 0-based and are not necessarily sorted. The data array stores
 * integer ids for reading edge features.
 *
 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 */ 
struct CSRMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows = 0, num_cols = 0;
  /*! \brief CSR index arrays */
  IdArray indptr, indices;
  /*! \brief data index array. When empty, assume it is from 0 to NNZ - 1. */
  IdArray data;
  /*! \brief whether the column indices per row are sorted */
  bool sorted = false;
  /*! \brief default constructor */
  CSRMatrix() = default;
  /*! \brief constructor */
  CSRMatrix(int64_t nrows, int64_t ncols,
            IdArray parr, IdArray iarr, IdArray darr = IdArray(),
            bool sorted_flag = false)
    : num_rows(nrows), num_cols(ncols), indptr(parr), indices(iarr),
      data(darr), sorted(sorted_flag) {}
};

/*!
 * \brief Plain COO structure
 * 
 * The data array stores integer ids for reading edge features.

 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 *
 * We call a COO matrix is *coalesced* if its row index is sorted.
 */
struct COOMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows = 0, num_cols = 0;
  /*! \brief COO index arrays */
  IdArray row, col;
  /*! \brief data index array. When empty, assume it is from 0 to NNZ - 1. */
  IdArray data;
  /*! \brief whether the row indices are sorted */
  bool row_sorted = false;
  /*! \brief whether the column indices per row are sorted */
  bool col_sorted = false;
  /*! \brief default constructor */
  COOMatrix() = default;
  /*! \brief constructor */
  COOMatrix(int64_t nrows, int64_t ncols,
            IdArray rarr, IdArray carr, IdArray darr = IdArray(),
            bool rsorted = false, bool csorted = false)
    : num_rows(nrows), num_cols(ncols), row(rarr), col(carr), data(darr),
      row_sorted(rsorted), col_sorted(csorted) {}
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

/*! \brief Whether the CSR matrix contains data */
inline bool CSRHasData(CSRMatrix csr) {
  return csr.data.defined();
}

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

/*!
 * \brief Sort the column index at each row in the ascending order.
 *
 * Examples:
 * num_rows = 4
 * num_cols = 4
 * indptr = [0, 2, 3, 3, 5]
 * indices = [1, 0, 2, 3, 1]
 *
 *  After CSRSort_(&csr)
 *
 * indptr = [0, 2, 3, 3, 5]
 * indices = [0, 1, 1, 2, 3]
 */
void CSRSort_(CSRMatrix* csr);

/*!
 * \brief Randomly select a fixed number of non-zero entries along each given row independently.
 *
 * The function performs random choices along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than num_samples,
 * all the values are picked.
 *
 * Examples:
 *
 * // csr.num_rows = 4;
 * // csr.num_cols = 4;
 * // csr.indptr = [0, 2, 3, 3, 5]
 * // csr.indices = [0, 1, 1, 2, 3]
 * // csr.data = [2, 3, 0, 1, 4]
 * CSRMatrix csr = ...;
 * IdArray rows = ... ; // [1, 3]
 * COOMatrix sampled = CSRRowWiseSampling(csr, rows, 2, FloatArray(), false);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [1, 3, 3]
 * // sampled.cols = [1, 2, 3]
 * // sampled.data = [3, 0, 4]
 *
 * \param mat Input CSR matrix.
 * \param rows Rows to sample from.
 * \param num_samples Number of samples
 * \param prob Unnormalized probability array. Should be of the same length as the data array.
 *             If an empty array is provided, assume uniform.
 * \param replace True if sample with replacement
 * \return A COOMatrix storing the picked row, col and data indices.
 */
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat,
    IdArray rows,
    int64_t num_samples,
    FloatArray prob = FloatArray(),
    bool replace = true);

/*!
 * \brief Select K non-zero entries with the largest weights along each given row.
 *
 * The function performs top-k selection along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than k,
 * all the values are picked.
 *
 * Examples:
 *
 * // csr.num_rows = 4;
 * // csr.num_cols = 4;
 * // csr.indptr = [0, 2, 3, 3, 5]
 * // csr.indices = [0, 1, 1, 2, 3]
 * // csr.data = [2, 3, 0, 1, 4]
 * CSRMatrix csr = ...;
 * IdArray rows = ... ;  // [0, 1, 3]
 * FloatArray weight = ... ;  // [1., 0., -1., 10., 20.]
 * COOMatrix sampled = CSRRowWiseTopk(csr, rows, 1, weight);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [0, 1, 3]
 * // sampled.cols = [1, 1, 2]
 * // sampled.data = [3, 0, 1]
 *
 * \param mat Input CSR matrix.
 * \param rows Rows to sample from.
 * \param k The K value.
 * \param weight Weight associated with each entry. Should be of the same length as the
 *               data array. If an empty array is provided, assume uniform.
 * \param ascending If true, elements are sorted by ascending order, equivalent to find
 *                 the K smallest values. Otherwise, find K largest values.
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat,
    IdArray rows,
    int64_t k,
    FloatArray weight,
    bool ascending = false);

///////////////////////// COO routines //////////////////////////

/*! \brief Return true if the value (row, col) is non-zero */
bool COOIsNonZero(COOMatrix , int64_t row, int64_t col);
/*!
 * \brief Batched implementation of COOIsNonZero.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
runtime::NDArray COOIsNonZero(COOMatrix, runtime::NDArray row, runtime::NDArray col);

/*! \brief Return the nnz of the given row */
int64_t COOGetRowNNZ(COOMatrix , int64_t row);
runtime::NDArray COOGetRowNNZ(COOMatrix , runtime::NDArray row);

/*! \brief Return the data array of the given row */
std::pair<runtime::NDArray, runtime::NDArray>
COOGetRowDataAndIndices(COOMatrix , int64_t row);

/*! \brief Whether the COO matrix contains data */
inline bool COOHasData(COOMatrix csr) {
  return csr.data.defined();
}

/*! \brief Get data. The return type is an ndarray due to possible duplicate entries. */
runtime::NDArray COOGetData(COOMatrix , int64_t row, int64_t col);

/*!
 * \brief Get the data and the row,col indices for each returned entries.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
std::vector<runtime::NDArray> COOGetDataAndIndices(
    COOMatrix , runtime::NDArray rows, runtime::NDArray cols);

/*! \brief Return a transposed COO matrix */
COOMatrix COOTranspose(COOMatrix coo);

/*!
 * \brief Convert COO matrix to CSR matrix.
 *
 * If the input COO matrix does not have data array, the data array of
 * the result CSR matrix stores a shuffle index for how the entries
 * will be reordered in CSR. The i^th entry in the result CSR corresponds
 * to the CSR.data[i] th entry in the input COO.
 */
CSRMatrix COOToCSR(COOMatrix coo);

/*!
 * \brief Slice rows of the given matrix and return.
 * \param coo COO matrix
 * \param start Start row id (inclusive)
 * \param end End row id (exclusive)
 */
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end);
COOMatrix COOSliceRows(COOMatrix coo, runtime::NDArray rows);

/*!
 * \brief Get the submatrix specified by the row and col ids.
 *
 * In numpy notation, given matrix M, row index array I, col index array J
 * This function returns the submatrix M[I, J].
 *
 * \param coo The input coo matrix
 * \param rows The row index to select
 * \param cols The col index to select
 * \return submatrix
 */
COOMatrix COOSliceMatrix(COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

/*! \return True if the matrix has duplicate entries */
bool COOHasDuplicate(COOMatrix coo);

/*!
 * \brief Sort the indices of a COO matrix.
 *
 * The function sorts row indices in ascending order. If sort_column is true,
 * col indices are sorted in ascending order too. The data array of the returned COOMatrix
 * stores the shuffled index which could be used to fetch edge data.
 *
 * \param mat The input coo matrix
 * \param sort_column True if column index should be sorted too.
 * \return COO matrix with index sorted.
 */
COOMatrix COOSort(COOMatrix mat, bool sort_column = false);

/*!
 * \brief Randomly select a fixed number of non-zero entries along each given row independently.
 *
 * The function performs random choices along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than num_samples,
 * all the values are picked.
 *
 * Examples:
 *
 * // coo.num_rows = 4;
 * // coo.num_cols = 4;
 * // coo.rows = [0, 0, 1, 3, 3]
 * // coo.cols = [0, 1, 1, 2, 3]
 * // coo.data = [2, 3, 0, 1, 4]
 * COOMatrix coo = ...;
 * IdArray rows = ... ; // [1, 3]
 * COOMatrix sampled = COORowWiseSampling(coo, rows, 2, FloatArray(), false);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [1, 3, 3]
 * // sampled.cols = [1, 2, 3]
 * // sampled.data = [3, 0, 4]
 *
 * \param mat Input coo matrix.
 * \param rows Rows to sample from.
 * \param num_samples Number of samples
 * \param prob Unnormalized probability array. Should be of the same length as the data array.
 *             If an empty array is provided, assume uniform.
 * \param replace True if sample with replacement
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix COORowWiseSampling(
    COOMatrix mat,
    IdArray rows,
    int64_t num_samples,
    FloatArray prob = FloatArray(),
    bool replace = true);

/*!
 * \brief Select K non-zero entries with the largest weights along each given row.
 *
 * The function performs top-k selection along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than k,
 * all the values are picked.
 *
 * Examples:
 *
 * // coo.num_rows = 4;
 * // coo.num_cols = 4;
 * // coo.rows = [0, 0, 1, 3, 3]
 * // coo.cols = [0, 1, 1, 2, 3]
 * // coo.data = [2, 3, 0, 1, 4]
 * COOMatrix coo = ...;
 * IdArray rows = ... ;  // [0, 1, 3]
 * FloatArray weight = ... ;  // [1., 0., -1., 10., 20.]
 * COOMatrix sampled = COORowWiseTopk(coo, rows, 1, weight);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [0, 1, 3]
 * // sampled.cols = [1, 1, 2]
 * // sampled.data = [3, 0, 1]
 *
 * \param mat Input COO matrix.
 * \param rows Rows to sample from.
 * \param k The K value.
 * \param weight Weight associated with each entry. Should be of the same length as the
 *               data array. If an empty array is provided, assume uniform.
 * \param ascending If true, elements are sorted by ascending order, equivalent to find
 *                  the K smallest values. Otherwise, find K largest values.
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix COORowWiseTopk(
    COOMatrix mat,
    IdArray rows,
    int64_t k,
    FloatArray weight,
    bool ascending = false);

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
 * ATEN_FLOAT_TYPE_SWITCH(array->dtype, FloatType, {
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
 * ATEN_DTYPE_SWITCH(array->dtype, DType, {
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

// Macro to dispatch according to device context and index type.
#define ATEN_CSR_SWITCH(csr, XPU, IdType, ...)              \
  ATEN_XPU_SWITCH((csr).indptr->ctx.device_type, XPU, {       \
    ATEN_ID_TYPE_SWITCH((csr).indptr->dtype, IdType, {        \
      {__VA_ARGS__}                                         \
    });                                                     \
  });

// Macro to dispatch according to device context and index type.
#define ATEN_COO_SWITCH(coo, XPU, IdType, ...)              \
  ATEN_XPU_SWITCH((coo).row->ctx.device_type, XPU, {          \
    ATEN_ID_TYPE_SWITCH((coo).row->dtype, IdType, {           \
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

namespace dmlc {

namespace serializer {

using dgl::aten::CSRMatrix;

constexpr uint64_t kDGLSerialize_AtenCsrMatrixMagic = 0xDD6cd31205dff127;

template <>
struct Handler<CSRMatrix> {
  inline static void Write(Stream* fs, const CSRMatrix& csr) {
    fs->Write(kDGLSerialize_AtenCsrMatrixMagic);
    fs->Write(csr.num_cols);
    fs->Write(csr.num_rows);
    fs->Write(csr.indptr);
    fs->Write(csr.indices);
    fs->Write(csr.data);
    fs->Write(csr.sorted);
  }
  inline static bool Read(Stream* fs, CSRMatrix* csr) {
    uint64_t magicNum;
    CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
    CHECK_EQ(magicNum, kDGLSerialize_AtenCsrMatrixMagic)
        << "Invalid CSRMatrix Data";
    CHECK(fs->Read(&csr->num_cols)) << "Invalid num_cols";
    CHECK(fs->Read(&csr->num_rows)) << "Invalid num_rows";
    CHECK(fs->Read(&csr->indptr)) << "Invalid indptr";
    CHECK(fs->Read(&csr->indices)) << "Invalid indices";
    CHECK(fs->Read(&csr->data)) << "Invalid data";
    CHECK(fs->Read(&csr->sorted)) << "Invalid sorted";
    return true;
  }
};
}  // namespace serializer
}  // namespace dmlc

#endif  // DGL_ARRAY_H_
