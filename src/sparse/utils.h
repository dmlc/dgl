/*!
 *  Copyright (c) 2022 by Contributors
 * \file sparse/utils.h
 * \brief DGL C++ sparse API utility
 */
#ifndef DGL_SPARSE_UTILS_H_
#define DGL_SPARSE_UTILS_H_

namespace dgl {
namespace sparse {

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