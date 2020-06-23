/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/spmat.h
 * \brief Sparse matrix definitions
 */
#ifndef DGL_ATEN_SPMAT_H_
#define DGL_ATEN_SPMAT_H_

#include <string>
#include <vector>
#include "./types.h"
#include "../runtime/object.h"

namespace dgl {

/*!
 * \brief Sparse format.
 */
enum class SparseFormat {
  kAny = 0,
  kCOO = 1,
  kCSR = 2,
  kCSC = 3,
  kAuto = 4   // kAuto is a placeholder that indicates it would be materialized later.
};

// Parse sparse format from string.
inline SparseFormat ParseSparseFormat(const std::string& name) {
  if (name == "coo")
    return SparseFormat::kCOO;
  else if (name == "csr")
    return SparseFormat::kCSR;
  else if (name == "csc")
    return SparseFormat::kCSC;
  else if (name == "any")
    return SparseFormat::kAny;
  else if (name == "auto")
    return SparseFormat::kAuto;
  else
    LOG(FATAL) << "Sparse format not recognized";
  return SparseFormat::kAny;
}

// Create string from sparse format.
inline std::string ToStringSparseFormat(SparseFormat sparse_format) {
  if (sparse_format == SparseFormat::kCOO)
    return std::string("coo");
  else if (sparse_format == SparseFormat::kCSR)
    return std::string("csr");
  else if (sparse_format == SparseFormat::kCSC)
    return std::string("csc");
  else if (sparse_format == SparseFormat::kAny)
    return std::string("any");
  else
    return std::string("auto");
}

// Sparse matrix object that is exposed to python API.
struct SparseMatrix : public runtime::Object {
  // Sparse format.
  int32_t format = 0;

  // Shape of this matrix.
  int64_t num_rows = 0, num_cols = 0;

  // Index arrays. For CSR, it is {indptr, indices, data}. For COO, it is {row, col, data}.
  std::vector<IdArray> indices;

  // Boolean flags.
  // TODO(minjie): We might revisit this later to provide a more general solution. Currently,
  //   we only consider aten::COOMatrix and aten::CSRMatrix.
  std::vector<bool> flags;

  SparseMatrix() {}

  SparseMatrix(int32_t fmt, int64_t nrows, int64_t ncols,
               const std::vector<IdArray>& idx,
               const std::vector<bool>& flg)
    : format(fmt), num_rows(nrows), num_cols(ncols), indices(idx), flags(flg) {}

  static constexpr const char* _type_key = "aten.SparseMatrix";
  DGL_DECLARE_OBJECT_TYPE_INFO(SparseMatrix, runtime::Object);
};
// Define SparseMatrixRef
DGL_DEFINE_OBJECT_REF(SparseMatrixRef, SparseMatrix);

}  // namespace dgl

#endif  // DGL_ATEN_SPMAT_H_
