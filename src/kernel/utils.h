/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/utils.h
 * \brief Kernel utilities
 */
#ifndef DGL_KERNEL_UTILS_H_
#define DGL_KERNEL_UTILS_H_

#include <minigun/spmat.h>
#include <dlpack/dlpack.h>
#include <dgl/array.h>
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
 * !\brief Compute the edge mapping given mapping and edge index tensor.
 */
template <typename Idx>
inline void ComputeEdgeMapping(Idx **cur_mapping, runtime::NDArray cur, runtime::NDArray eids) {
  if (*cur_mapping == nullptr) {
    if (!aten::IsNullArray(eids))
      *cur_mapping = static_cast<Idx*>(eids->data);
  } else {
    runtime::NDArray out_map = aten::MergeIDMapping(eids, cur);
    *cur_mapping = static_cast<Idx*>(out_map->data);
  }
}

template void ComputeEdgeMapping<int>(int **cur_mapping, runtime::NDArray cur, runtime::NDArray eids);
template void ComputeEdgeMapping<long long>(long long **cur_mapping, runtime::NDArray cur, runtime::NDArray eids);

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

/*
 * !\brief Create minigun COO from two ndarrays.
 */
template <typename Idx>
minigun::Coo<Idx> CreateCoo(runtime::NDArray row, runtime::NDArray col) {
  minigun::Coo<Idx> coo;
  coo.row.data = static_cast<Idx*>(row->data);
  coo.row.length = row->shape[0];
  coo.column.data = static_cast<Idx*>(col->data);
  coo.column.length = col->shape[0];
  return coo;
}

}  // namespace utils
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_UTILS_H_
