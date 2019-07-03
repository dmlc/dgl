/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array_op.h
 * \brief Array operator templates
 */

#include <dgl/array.h>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs);

template <DLDeviceType XPU, typename IdType>
IdArray HStack(IdArray arr1, IdArray arr2);

template <DLDeviceType XPU, typename IdType>
IdArray Full(IdType val, int64_t length);

// sparse arrays

template <DLDeviceType XPU, typename IdType, typename DType>
bool CSRIsNonZero(const CSRMatrix& , int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType, typename DType>
int64_t CSRGetRowNNZ(const CSRMatrix& , int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetRowColumnIndices(const CSRMatrix& , int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetRowData(const CSRMatrix& , int64_t row);


template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(const CSRMatrix& , int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(const CSRMatrix&, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    const CSRMatrix& , runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRTranspose(const CSRMatrix& );

template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRToCOO(const CSRMatrix& );

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceRows(const CSRMatrix& csr, int64_t start, int64_t end);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceMatrix(const CSRMatrix& csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix COOToCSR(const COOMatrix& );

}  // namespace impl
}  // namespace aten
}  // namespace dgl
