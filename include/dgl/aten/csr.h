/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/csr.h
 * \brief Common CSR operations required by DGL.
 */
#ifndef DGL_ATEN_CSR_H_
#define DGL_ATEN_CSR_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include <vector>
#include "./types.h"
#include "./array_ops.h"
#include "./spmat.h"
#include "./macro.h"

namespace dgl {
namespace aten {

struct COOMatrix;

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

constexpr uint64_t kDGLSerialize_AtenCsrMatrixMagic = 0xDD6cd31205dff127;

struct CSRMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows = 0, num_cols = 0;
  /*! \brief CSR index arrays */
  IdArray indptr, indices;
  /*! \brief data index array. When is null, assume it is from 0 to NNZ - 1. */
  IdArray data;
  /*! \brief whether the column indices per row are sorted */
  bool sorted = false;
  /*! \brief default constructor */
  CSRMatrix() = default;
  /*! \brief constructor */
  CSRMatrix(int64_t nrows, int64_t ncols, IdArray parr, IdArray iarr,
            IdArray darr = NullArray(), bool sorted_flag = false)
      : num_rows(nrows),
        num_cols(ncols),
        indptr(parr),
        indices(iarr),
        data(darr),
        sorted(sorted_flag) {
    CheckValidity();
  }

  /*! \brief constructor from SparseMatrix object */
  explicit CSRMatrix(const SparseMatrix& spmat)
      : num_rows(spmat.num_rows),
        num_cols(spmat.num_cols),
        indptr(spmat.indices[0]),
        indices(spmat.indices[1]),
        data(spmat.indices[2]),
        sorted(spmat.flags[0]) {
    CheckValidity();
  }

  // Convert to a SparseMatrix object that can return to python.
  SparseMatrix ToSparseMatrix() const {
    return SparseMatrix(static_cast<int32_t>(SparseFormat::kCSR), num_rows,
                        num_cols, {indptr, indices, data}, {sorted});
  }

  bool Load(dmlc::Stream* fs) {
    uint64_t magicNum;
    CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
    CHECK_EQ(magicNum, kDGLSerialize_AtenCsrMatrixMagic)
        << "Invalid CSRMatrix Data";
    CHECK(fs->Read(&num_cols)) << "Invalid num_cols";
    CHECK(fs->Read(&num_rows)) << "Invalid num_rows";
    CHECK(fs->Read(&indptr)) << "Invalid indptr";
    CHECK(fs->Read(&indices)) << "Invalid indices";
    CHECK(fs->Read(&data)) << "Invalid data";
    CHECK(fs->Read(&sorted)) << "Invalid sorted";
    CheckValidity();
    return true;
  }

  void Save(dmlc::Stream* fs) const {
    fs->Write(kDGLSerialize_AtenCsrMatrixMagic);
    fs->Write(num_cols);
    fs->Write(num_rows);
    fs->Write(indptr);
    fs->Write(indices);
    fs->Write(data);
    fs->Write(sorted);
  }

  inline void CheckValidity() const {
    CHECK_SAME_DTYPE(indptr, indices);
    CHECK_SAME_CONTEXT(indptr, indices);
    if (!aten::IsNullArray(data)) {
      CHECK_SAME_DTYPE(indptr, data);
      CHECK_SAME_CONTEXT(indptr, data);
    }
    CHECK_NO_OVERFLOW(indptr->dtype, num_rows);
    CHECK_NO_OVERFLOW(indptr->dtype, num_cols);
  }
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
  return !IsNullArray(csr.data);
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
 * \brief Reorder the rows and colmns according to the new row and column order.
 * \param csr The input csr matrix.
 * \param new_row_ids the new row Ids (the index is the old row Id)
 * \param new_col_ids the new column Ids (the index is the old col Id).
 */
CSRMatrix CSRReorder(CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

/*!
 * \brief Remove entries from CSR matrix by entry indices (data indices)
 * \return A new CSR matrix as well as a mapping from the new CSR entries to the old CSR
 *         entries.
 */
CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries);

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

}  // namespace aten
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::aten::CSRMatrix, true);
}  // namespace dmlc

#endif  // DGL_ATEN_CSR_H_
