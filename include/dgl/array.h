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
#include <utility>

namespace dgl {

typedef uint64_t dgl_id_t;
typedef uint64_t dgl_type_t;
typedef dgl::runtime::NDArray IdArray;
typedef dgl::runtime::NDArray DegreeArray;
typedef dgl::runtime::NDArray BoolArray;
typedef dgl::runtime::NDArray IntArray;
typedef dgl::runtime::NDArray FloatArray;
typedef dgl::runtime::NDArray TypeArray;

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

/*! \brief Return the data under the index. In numpy notation, A[I] */
int64_t IndexSelect(IdArray array, int64_t index);
IdArray IndexSelect(IdArray array, IdArray index);

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

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_H_
