/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/csr_mask.cc
 * \brief CSR Masking Operation
 */

#include <dgl/array.h>
#include <parallel_hashmap/phmap.h>
#include <vector>
#include "array_utils.h"

namespace dgl {

using dgl::runtime::NDArray;

namespace aten {

namespace {

// TODO(BarclayII): avoid using map for sorted CSRs
template <typename IdType, typename DType>
void ComputeValues(
    const IdType* A_indptr,
    const IdType* A_indices,
    const IdType* A_eids,
    const DType* A_data,
    const IdType* B_indptr,
    const IdType* B_indices,
    const IdType* B_eids,
    DType* C_data,
    int64_t M) {
  phmap::flat_hash_map<IdType, DType> map;
#pragma omp parallel for firstprivate(map)
  for (IdType i = 0; i < M; ++i) {
    map.clear();

    for (IdType u = A_indptr[i]; u < A_indptr[i + 1]; ++u) {
      IdType kA = A_indices[u];
      map[kA] = A_data[A_eids ? A_eids[u] : u];
    }

    for (IdType v = B_indptr[i]; v < B_indptr[i + 1]; ++v) {
      IdType kB = B_indices[v];
      auto it = map.find(kB);
      C_data[B_eids ? B_eids[v] : v] = (it != map.end()) ? it->second : 0;
    }
  }
}

};  // namespace

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
  const int64_t N = A.num_cols;

  NDArray C_weights = NDArray::Empty({B.indices->shape[0]}, A_weights->dtype, A_weights->ctx);
  DType* C_data = C_weights.Ptr<DType>();
  ComputeValues(A_indptr, A_indices, A_eids, A_data, B_indptr, B_indices, B_eids, C_data, M);

  return C_weights;
}

template NDArray CSRMask<kDLCPU, int32_t, float>(const CSRMatrix&, NDArray, const CSRMatrix&);
template NDArray CSRMask<kDLCPU, int64_t, float>(const CSRMatrix&, NDArray, const CSRMatrix&);
template NDArray CSRMask<kDLCPU, int32_t, double>(const CSRMatrix&, NDArray, const CSRMatrix&);
template NDArray CSRMask<kDLCPU, int64_t, double>(const CSRMatrix&, NDArray, const CSRMatrix&);

};  // namespace aten
};  // namespace dgl
