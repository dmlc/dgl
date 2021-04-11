/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cuda/csr_mask.cu
 * \brief CSR Masking Operation
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {
using dgl::runtime::NDArray;
namespace aten {

template <typename IdType, typename DType>
__global__ void _ComputeValuesKernel(
    const IdType* A_indptr,
    const IdType* A_indices,
    const IdType* A_eids,
    const DType* A_data,
    const IdType* B_indptr,
    const IdType* B_indices,
    const IdType* B_eids,
    DType* C_data,
    int64_t M) {
  IdType tx = blockIdx.x * blockDim.x + threadIdx.x;
  const IdType stride_x = gridDim.x * blockDim.x;
  while (tx < M) {
    for (IdType v = B_indptr[tx]; v < B_indptr[tx + 1]; ++v) {
      // TODO(BarclayII): shared memory?
      IdType vi = B_eids ? B_eids[v] : v;
      C_data[vi] = 0;
      for (IdType u = A_indptr[tx]; u < A_indptr[tx + 1]; ++u) {
        if (A_indices[u] == B_indices[v]) {
          C_data[vi] = A_data[A_eids ? A_eids[u] : u];
          break;
        }
      }
    }
    tx += stride_x;
  }
}

template <int XPU, typename IdType, typename DType>
NDArray CSRMask(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B) {
  CHECK_EQ(A.num_rows, B.num_rows) << "Number of rows must match.";
  CHECK_EQ(A.num_cols, B.num_cols) << "Number of columns must match.";
  const bool A_has_eid = !IsNullArray(A.data);
  const bool B_has_eid = !IsNullArray(B.data);
  const IdType* A_indptr = A.indptr.Ptr<IdType>();
  const IdType* A_indices = A.indices.Ptr<IdType>();
  const IdType* A_eids = A_has_eid ? A.data.Ptr<IdType>() : nullptr;
  const IdType* B_indptr = B.indptr.Ptr<IdType>();
  const IdType* B_indices = B.indices.Ptr<IdType>();
  const IdType* B_eids = B_has_eid ? B.data.Ptr<IdType>() : nullptr;
  const DType* A_data = A_weights.Ptr<DType>();
  const int64_t M = A.num_rows;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  NDArray C_weights = NDArray::Empty({B.indices->shape[0]}, A_weights->dtype, A_weights->ctx);
  DType* C_data = C_weights.Ptr<DType>();
  const int nt = cuda::FindNumThreads(M);
  const int nb = (M + nt - 1) / nt;
  CUDA_KERNEL_CALL(
      _ComputeValuesKernel,
      nb, nt, 0, thr_entry->stream,
      A_indptr, A_indices, A_eids, A_data,
      B_indptr, B_indices, B_eids,
      C_data, M);

  return C_weights;
}

template NDArray CSRMask<kDLGPU, int32_t, float>(const CSRMatrix&, NDArray, const CSRMatrix&);
template NDArray CSRMask<kDLGPU, int64_t, float>(const CSRMatrix&, NDArray, const CSRMatrix&);
template NDArray CSRMask<kDLGPU, int32_t, double>(const CSRMatrix&, NDArray, const CSRMatrix&);
template NDArray CSRMask<kDLGPU, int64_t, double>(const CSRMatrix&, NDArray, const CSRMatrix&);

};  // namespace aten
};  // namespace dgl
