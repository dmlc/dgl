/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/utils.h
 * \brief DGL C++ sparse API utility
 */
#ifndef DGL_SPARSE_UTILS_H_
#define DGL_SPARSE_UTILS_H_

namespace dgl {
namespace sparse {

/*! \brief Macro to select the sparse format for two sparse matrices. It chooses
 * COO if anyone of the two sparse matrices has COO format. If none of them has
 * COO, it tries CSR and CSC in the same manner. */
#define SPARSE_FORMAT_SELECT_BINARY(A, B, fmt, ...) \
  do {                                              \
    SparseFormat fmt;                               \
    if (A->HasCOO() || B->HasCOO()) {               \
      fmt = SparseFormat::kCOO;                     \
    } else if (A->HasCSR() || B->HasCSR()) {        \
      fmt = SparseFormat::kCSR;                     \
    } else {                                        \
      fmt = SparseFormat::kCSC;                     \
    }                                               \
    { __VA_ARGS__ }                                 \
  } while (0)

}  // namespace sparse
}  // namespace dgl

#endif  // DGL_SPARSE_UTILS_H_