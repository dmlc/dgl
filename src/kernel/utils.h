/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/utils.h
 * \brief Kernel utilities
 */
#ifndef DGL_KERNEL_UTILS_H_
#define DGL_KERNEL_UTILS_H_

#include <minigun/csr.h>
#include <dlpack/dlpack.h>
#include <dgl/runtime/ndarray.h>

#include <cstdlib>
#include <vector>

namespace dgl {
namespace kernel {
namespace utils {

/*
 * !\brief Find number of threads is smaller than dim and max_nthrs
 * and is also the power of two.
 */
int FindNumThreads(int dim, int max_nthrs);

/*
 * !\brief Compute the total number of feature elements.
 */
int64_t ComputeXLength(runtime::NDArray feat_array);

/*
 * !\brief Compute the total number of elements in the array.
 */
int64_t NElements(const runtime::NDArray& array);

/*
 * !\brief Compute the product of the given vector.
 */
int64_t Prod(const std::vector<int64_t>& vec);

/*
 * !\brief Fill the array with constant value.
 */
template <int XPU, typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val);

/*
 * !\brief Create minigun CSR from two ndarrays.
 */
template <typename Idx>
minigun::Csr<Idx> CreateCsr(runtime::NDArray indptr, runtime::NDArray indices) {
  minigun::Csr<Idx> csr;
  csr.row_offsets.data = static_cast<Idx*>(indptr->data);
  csr.row_offsets.length = indptr->shape[0];
  csr.column_indices.data = static_cast<Idx*>(indices->data);
  csr.column_indices.length = indices->shape[0];
  return csr;
}

}  // namespace utils
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_UTILS_H_
