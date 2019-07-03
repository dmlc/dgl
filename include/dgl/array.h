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
#include <vector>

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
 * \brief Create a new boolean array with given length
 * \note FIXME: the elements are 64-bit.
 * \param length The array length
 * \param ctx The array context
 * \return the bool array
 */
BoolArray NewBoolArray(int64_t length, DLContext ctx = DLContext{kDLCPU, 0});

/*!
 * \brief Create a new id array using the given vector data
 * \param vec The vector data
 * \param ctx The array context
 * \return the id array
 */
IdArray VecToIdArray(const std::vector<int32_t>& vec, DLContext ctx = DLContext{kDLCPU, 0});
IdArray VecToIdArray(const std::vector<int64_t>& vec, DLContext ctx = DLContext{kDLCPU, 0});

/*! \brief Create a copy of the given array */
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

/*! \brief Stack two arrays (of len L) into a 2*L length array */
IdArray HStack(IdArray arr1, IdArray arr2);

/*! \brief Return an array full of the given value */
IdArray Full(int32_t val, int64_t length);
IdArray Full(int64_t val, int64_t length);

/*! \brief Concat the given 1D arrays */
IdArray Concat(const std::vector<IdArray>& arrays);

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
bool CSRIsNonZero(const CSRMatrix& , int64_t row, int64_t col);

/*! \brief Return the nnz of the given row */
int64_t CSRGetRowNNZ(const CSRMatrix& , int64_t row);

/*! \brief Return the column index array of the given row */
runtime::NDArray CSRGetRowColumnIndices(const CSRMatrix& , int64_t row);

/*! \brief Return the data array of the given row */
runtime::NDArray CSRGetRowData(const CSRMatrix& , int64_t row);

/* \brief Get data. The return type is an ndarray due to possible duplicate entries. */
runtime::NDArray CSRGetData(const CSRMatrix& , int64_t row, int64_t col);
runtime::NDArray CSRGetData(const CSRMatrix&, runtime::NDArray rows, runtime::NDArray cols);

/* \brief Get the data and the row,col indices for each returned entries. */
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    const CSRMatrix& , runtime::NDArray rows, runtime::NDArray cols);

/*! \brief Return a transposed CSR matrix */
CSRMatrix CSRTranspose(const CSRMatrix& );

/*! \brief Convert COO matrix to CSR matrix. */
COOMatrix CSRToCOO(const CSRMatrix& );

/*! \brief Slice rows of the given matrix and return. */
CSRMatrix CSRSliceRows(const CSRMatrix& csr, int64_t start, int64_t end);

/*! \brief Get the submatrix specified by the row and col ids. */
CSRMatrix CSRSliceMatrix(const CSRMatrix& csr, runtime::NDArray rows, runtime::NDArray cols);

///////////////////////// COO routines //////////////////////////

/*! \brief Convert COO matrix to CSR matrix. */
CSRMatrix COOToCSR(const COOMatrix& );

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_H_
