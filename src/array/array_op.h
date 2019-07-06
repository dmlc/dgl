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
IdArray Full(IdType val, int64_t length, DLContext ctx);

template <DLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DLContext ctx);

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

// sparse arrays

template <DLDeviceType XPU, typename IdType, typename DType>
bool CSRIsNonZero(CSRMatrix , int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType, typename DType>
int64_t CSRGetRowNNZ(CSRMatrix , int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetRowColumnIndices(CSRMatrix , int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetRowData(CSRMatrix , int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(CSRMatrix , int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(CSRMatrix, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    CSRMatrix , runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRTranspose(CSRMatrix );

// Convert CSR to COO
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix );

// Convert CSR to COO using data array as order
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix );

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix COOToCSR(COOMatrix );

}  // namespace impl
}  // namespace aten
}  // namespace dgl
