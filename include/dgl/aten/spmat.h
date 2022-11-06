/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/aten/spmat.h
 * @brief Sparse matrix definitions
 */
#ifndef DGL_ATEN_SPMAT_H_
#define DGL_ATEN_SPMAT_H_

#include <string>
#include <vector>

#include "../runtime/object.h"
#include "./types.h"

namespace dgl {

/**
 * @brief Sparse format.
 */
enum class SparseFormat {
  kCOO = 1,
  kCSR = 2,
  kCSC = 3,
};

/**
 * @brief Sparse format codes
 */
const dgl_format_code_t ALL_CODE = 0x7;
const dgl_format_code_t ANY_CODE = 0x0;
const dgl_format_code_t COO_CODE = 0x1;
const dgl_format_code_t CSR_CODE = 0x2;
const dgl_format_code_t CSC_CODE = 0x4;

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
  if (code & COO_CODE) ret.push_back(SparseFormat::kCOO);
  if (code & CSR_CODE) ret.push_back(SparseFormat::kCSR);
  if (code & CSC_CODE) ret.push_back(SparseFormat::kCSC);
  return ret;
}

inline dgl_format_code_t SparseFormatsToCode(
    const std::vector<SparseFormat>& formats) {
  dgl_format_code_t ret = 0;
  for (auto format : formats) {
    switch (format) {
      case SparseFormat::kCOO:
        ret |= COO_CODE;
        break;
      case SparseFormat::kCSR:
        ret |= CSR_CODE;
        break;
      case SparseFormat::kCSC:
        ret |= CSC_CODE;
        break;
      default:
        LOG(FATAL) << "Only support COO/CSR/CSC formats.";
    }
  }
  return ret;
}

inline std::string CodeToStr(dgl_format_code_t code) {
  std::string ret = "";
  if (code & COO_CODE) ret += "coo ";
  if (code & CSR_CODE) ret += "csr ";
  if (code & CSC_CODE) ret += "csc ";
  return ret;
}

inline SparseFormat DecodeFormat(dgl_format_code_t code) {
  if (code & COO_CODE) return SparseFormat::kCOO;
  if (code & CSC_CODE) return SparseFormat::kCSC;
  return SparseFormat::kCSR;
}

// Sparse matrix object that is exposed to python API.
struct SparseMatrix : public runtime::Object {
  // Sparse format.
  int32_t format = 0;

  // Shape of this matrix.
  int64_t num_rows = 0, num_cols = 0;

  // Index arrays. For CSR, it is {indptr, indices, data}. For COO, it is {row,
  // col, data}.
  std::vector<IdArray> indices;

  // Boolean flags.
  // TODO(minjie): We might revisit this later to provide a more general
  // solution. Currently, we only consider aten::COOMatrix and aten::CSRMatrix.
  std::vector<bool> flags;

  SparseMatrix() {}

  SparseMatrix(
      int32_t fmt, int64_t nrows, int64_t ncols,
      const std::vector<IdArray>& idx, const std::vector<bool>& flg)
      : format(fmt),
        num_rows(nrows),
        num_cols(ncols),
        indices(idx),
        flags(flg) {}

  static constexpr const char* _type_key = "aten.SparseMatrix";
  DGL_DECLARE_OBJECT_TYPE_INFO(SparseMatrix, runtime::Object);
};
// Define SparseMatrixRef
DGL_DEFINE_OBJECT_REF(SparseMatrixRef, SparseMatrix);

}  // namespace dgl

#endif  // DGL_ATEN_SPMAT_H_
