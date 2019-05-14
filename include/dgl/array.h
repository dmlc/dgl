/*!
 *  Copyright (c) 2018 by Contributors
 * \file dgl/array.h
 * \brief Array types and common array operations required by DGL.
 *
 * Note that this is not meant for a full support of array library such as Torch/MXNet.
 * Only a limited set of operators required by DGL are implemented.
 */
#ifndef DGL_ARRAY_H_
#define DGL_ARRAY_H_

#include <dgl/runtime/ndarray.h>
#include <vector>

namespace dgl {

typedef uint64_t dgl_id_t;
typedef dgl::runtime::NDArray IdArray;
typedef dgl::runtime::NDArray DegreeArray;
typedef dgl::runtime::NDArray BoolArray;
typedef dgl::runtime::NDArray IntArray;
typedef dgl::runtime::NDArray FloatArray;

/*! \brief Create a new id array with given length (on CPU) */
IdArray NewIdArray(int64_t length);

/*! \brief Create a new id array with the given vector data (on CPU) */
IdArray VecToIdArray(const std::vector<dgl_id_t>& vec);

/*! \brief Create a copy of the given array */
IdArray Clone(IdArray arr);

/*! \brief Plain CSR matrix */
struct CSRMatrix {
  IdArray indptr, indices, data;
};

/*! \brief Plain COO structure */
struct COOMatrix {
  IdArray row, col, data;
};

/*!
 * \brief Slice rows of the given matrix and return.
 */
CSRMatrix SliceRows(const CSRMatrix& csr, int64_t start, int64_t end);

}  // namespace dgl

#endif  // DGL_ARRAY_H_
