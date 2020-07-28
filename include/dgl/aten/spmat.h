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
  kCOO = 1,
  kCSR = 2,
  kCSC = 3,
};

/*!
 * \brief Sparse format codes
 */
const dgl_format_code_t all_code = 0x7;
const dgl_format_code_t coo_code = 0x1;
const dgl_format_code_t csr_code = 0x2;
const dgl_format_code_t csc_code = 0x4;

// Parse sparse format from string.
inline SparseFormat ParseSparseFormat(const std::string& name) {
  if (name == "coo")
    return SparseFormat::kCOO;
  else if (name == "csr")
    return SparseFormat::kCSR;
  else if (name == "csc")
    return SparseFormat::kCSC;
  else
    LOG(FATAL) << "Sparse format not recognized";
  return SparseFormat::kCOO;
}

// Create string from sparse format.
inline std::string ToStringSparseFormat(SparseFormat sparse_format) {
  if (sparse_format == SparseFormat::kCOO)
    return std::string("coo");
  else if (sparse_format == SparseFormat::kCSR)
    return std::string("csr");
  else
    return std::string("csc");
}

inline std::vector<SparseFormat> CodeToSparseFormats(dgl_format_code_t code) {
  std::vector<SparseFormat> ret;
  if (code & coo_code)
    ret.push_back(SparseFormat::kCOO);
  if (code & csr_code)
    ret.push_back(SparseFormat::kCSR);
  if (code & csc_code)
    ret.push_back(SparseFormat::kCSC);
  return ret;
}

inline dgl_format_code_t
SparseFormatsToCode(const std::vector<SparseFormat> &formats) {
  dgl_format_code_t ret = 0;
  for (auto format : formats) {
    switch (format) {
    case SparseFormat::kCOO:
      ret |= coo_code;
      break;
    case SparseFormat::kCSR:
      ret |= csr_code;
      break;
    case SparseFormat::kCSC:
      ret |= csc_code;
      break;
    default:
      LOG(FATAL) << "Only support COO/CSR/CSC formats.";
    }
  }
  return ret;
}

inline std::string CodeToStr(dgl_format_code_t code) {
  std::string ret = "";
  if (code & coo_code)
    ret += "coo ";
  if (code & csr_code)
    ret += "csr ";
  if (code & csc_code)
    ret += "csc ";
  return ret;
}

inline SparseFormat DecodeFormat(dgl_format_code_t code) {
  if (code & coo_code)
    return SparseFormat::kCOO;
  if (code & csc_code)
    return SparseFormat::kCSC;
  return SparseFormat::kCSR;
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
