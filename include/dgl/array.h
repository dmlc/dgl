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

/*! \brief Convert the idarray to the given bit width (on CPU) */
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


/*! \brief Plain CSR matrix */
struct CSRMatrix {
  IdArray indptr, indices, data;
};

/*! \brief Plain COO structure */
struct COOMatrix {
  IdArray row, col, data;
};

/*! \brief Slice rows of the given matrix and return. */
CSRMatrix SliceRows(const CSRMatrix& csr, int64_t start, int64_t end);

/*! \brief Convert COO matrix to CSR matrix. */
CSRMatrix ToCSR(const COOMatrix);

/*! \brief Convert COO matrix to CSR matrix. */
COOMatrix ToCOO(const CSRMatrix);

}  // namespace dgl

#endif  // DGL_ARRAY_H_
