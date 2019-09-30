/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array_op.h
 * \brief Array operator templates
 */
#ifndef DGL_ARRAY_ARRAY_OP_H_
#define DGL_ARRAY_ARRAY_OP_H_

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

template <DLDeviceType XPU, typename IdType>
IdArray IndexSelect(IdArray array, IdArray index);

template <DLDeviceType XPU, typename IdType>
int64_t IndexSelect(IdArray array, int64_t index);

template <DLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays);

// sparse arrays

template <DLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRIsNonZero(CSRMatrix csr, runtime::NDArray row, runtime::NDArray col);

template <DLDeviceType XPU, typename IdType>
bool CSRHasDuplicate(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowNNZ(CSRMatrix csr, runtime::NDArray row);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetRowData(CSRMatrix csr, int64_t row);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(CSRMatrix csr, int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRTranspose(CSRMatrix csr);

// Convert CSR to COO
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr);

// Convert CSR to COO using data array as order
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceRows(CSRMatrix csr, runtime::NDArray rows);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType, typename DType>
void CSRSort(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType>
bool COOHasDuplicate(COOMatrix coo);

template <DLDeviceType XPU, typename IdType, typename DType>
CSRMatrix COOToCSR(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_ARRAY_OP_H_
