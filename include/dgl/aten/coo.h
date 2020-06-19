
/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/coo.h
 * \brief Common COO operations required by DGL.
 */
#ifndef DGL_ATEN_COO_H_
#define DGL_ATEN_COO_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include <vector>
#include <utility>
#include "./types.h"
#include "./array_ops.h"
#include "./spmat.h"
#include "./macro.h"

namespace dgl {
namespace aten {

struct CSRMatrix;

/*!
 * \brief Plain COO structure
 *
 * The data array stores integer ids for reading edge features.

 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 */

constexpr uint64_t kDGLSerialize_AtenCooMatrixMagic = 0xDD61ffd305dff127;

// TODO(BarclayII): Graph queries on COO formats should support the case where
// data ordered by rows/columns instead of EID.
struct COOMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows = 0, num_cols = 0;
  /*! \brief COO index arrays */
  IdArray row, col;
  /*! \brief data index array. When is null, assume it is from 0 to NNZ - 1. */
  IdArray data;
  /*! \brief whether the row indices are sorted */
  bool row_sorted = false;
  /*! \brief whether the column indices per row are sorted */
  bool col_sorted = false;
  /*! \brief default constructor */
  COOMatrix() = default;
  /*! \brief constructor */
  COOMatrix(int64_t nrows, int64_t ncols, IdArray rarr, IdArray carr,
            IdArray darr = NullArray(), bool rsorted = false,
            bool csorted = false)
      : num_rows(nrows),
        num_cols(ncols),
        row(rarr),
        col(carr),
        data(darr),
        row_sorted(rsorted),
        col_sorted(csorted) {
    CheckValidity();
  }

  /*! \brief constructor from SparseMatrix object */
  explicit COOMatrix(const SparseMatrix& spmat)
      : num_rows(spmat.num_rows),
        num_cols(spmat.num_cols),
        row(spmat.indices[0]),
        col(spmat.indices[1]),
        data(spmat.indices[2]),
        row_sorted(spmat.flags[0]),
        col_sorted(spmat.flags[1]) {
    CheckValidity();
  }

  // Convert to a SparseMatrix object that can return to python.
  SparseMatrix ToSparseMatrix() const {
    return SparseMatrix(static_cast<int32_t>(SparseFormat::kCOO), num_rows,
                        num_cols, {row, col, data}, {row_sorted, col_sorted});
  }

  bool Load(dmlc::Stream* fs) {
    uint64_t magicNum;
    CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
    CHECK_EQ(magicNum, kDGLSerialize_AtenCooMatrixMagic)
        << "Invalid COOMatrix Data";
    CHECK(fs->Read(&num_cols)) << "Invalid num_cols";
    CHECK(fs->Read(&num_rows)) << "Invalid num_rows";
    CHECK(fs->Read(&row)) << "Invalid row";
    CHECK(fs->Read(&col)) << "Invalid col";
    CHECK(fs->Read(&data)) << "Invalid data";
    CHECK(fs->Read(&row_sorted)) << "Invalid row_sorted";
    CHECK(fs->Read(&col_sorted)) << "Invalid col_sorted";
    CheckValidity();
    return true;
  }

  void Save(dmlc::Stream* fs) const {
    fs->Write(kDGLSerialize_AtenCooMatrixMagic);
    fs->Write(num_cols);
    fs->Write(num_rows);
    fs->Write(row);
    fs->Write(col);
    fs->Write(data);
    fs->Write(row_sorted);
    fs->Write(col_sorted);
  }

  inline void CheckValidity() const {
    CHECK_SAME_DTYPE(row, col);
    CHECK_SAME_CONTEXT(row, col);
    if (!aten::IsNullArray(data)) {
      CHECK_SAME_DTYPE(row, data);
      CHECK_SAME_CONTEXT(row, data);
    }
    CHECK_NO_OVERFLOW(row->dtype, num_rows);
    CHECK_NO_OVERFLOW(row->dtype, num_cols);
  }
};

///////////////////////// COO routines //////////////////////////

/*! \brief Return true if the value (row, col) is non-zero */
bool COOIsNonZero(COOMatrix , int64_t row, int64_t col);
/*!
 * \brief Batched implementation of COOIsNonZero.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
runtime::NDArray COOIsNonZero(COOMatrix , runtime::NDArray row, runtime::NDArray col);

/*! \brief Return the nnz of the given row */
int64_t COOGetRowNNZ(COOMatrix , int64_t row);
runtime::NDArray COOGetRowNNZ(COOMatrix , runtime::NDArray row);

/*! \brief Return the data array of the given row */
std::pair<runtime::NDArray, runtime::NDArray>
COOGetRowDataAndIndices(COOMatrix , int64_t row);

/*! \brief Whether the COO matrix contains data */
inline bool COOHasData(COOMatrix csr) {
  return !IsNullArray(csr.data);
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
 * \brief Deduplicate the entries of a sorted COO matrix, replacing the data with the
 * number of occurrences of the row-col coordinates.
 */
std::pair<COOMatrix, IdArray> COOCoalesce(COOMatrix coo);

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
 * \brief Remove entries from COO matrix by entry indices (data indices)
 * \return A new COO matrix as well as a mapping from the new COO entries to the old COO
 *         entries.
 */
COOMatrix COORemove(COOMatrix coo, IdArray entries);

/*!
 * \brief Reorder the rows and colmns according to the new row and column order.
 * \param csr The input coo matrix.
 * \param new_row_ids the new row Ids (the index is the old row Id)
 * \param new_col_ids the new column Ids (the index is the old col Id).
 */
COOMatrix COOReorder(COOMatrix coo, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

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
    NDArray weight,
    bool ascending = false);

}  // namespace aten
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::aten::COOMatrix, true);
}  // namespace dmlc

#endif  // DGL_ATEN_COO_H_
