/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/csr_mm.cc
 * \brief CSR Matrix Multiplication
 */

#include <dgl/array.h>
#include <parallel_hashmap/phmap.h>
#include <vector>
#include "array_utils.h"

namespace dgl {

using dgl::runtime::NDArray;

namespace aten {

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights) {
  CHECK_EQ(A.num_cols, B.num_rows) << "A's number of columns must equal to B's number of rows";
  const bool A_has_eid = !IsNullArray(A.data);
  const bool B_has_eid = !IsNullArray(B.data);
  const IdType* A_indptr = A.indptr.Ptr<IdType>();
  const IdType* A_indices = A.indices.Ptr<IdType>();
  const IdType* A_eids = A_has_eid ? A.data.Ptr<IdType>() : nullptr;
  const IdType* B_indptr = B.indptr.Ptr<IdType>();
  const IdType* B_indices = B.indices.Ptr<IdType>();
  const IdType* B_eids = B_has_eid ? B.data.Ptr<IdType>() : nullptr;
  const DType* A_data = A_weights.Ptr<DType>();
  const DType* B_data = B_weights.Ptr<DType>();
  const int64_t M = A.num_rows;
  const int64_t P = B.num_cols;

  IdArray C_indptr = IdArray::Empty({M + 1}, A.indptr->dtype, A.indptr->ctx);
  IdType* C_indptr_data = C_indptr.Ptr<IdType>();
  std::vector<IdType> C_indices;
  std::vector<DType> C_weights;
  IdType nnz = 0;
  C_indptr_data[0] = 0;

  std::vector<bool> has_value(P);
  std::vector<DType> values(P);

  for (IdType i = 0; i < M; ++i) {
    for (IdType u = A_indptr[i]; u < A_indptr[i + 1]; ++u) {
      IdType w = A_indices[u];
      DType vA = A_data[A_eids ? A_eids[u] : u];
      for (IdType v = B_indptr[w]; v < B_indptr[w + 1]; ++v) {
        IdType t = B_indices[v];
        DType vB = B_data[B_eids ? B_eids[v] : v];
        has_value[t] = true;
        values[t] += vA * vB;
      }
    }

    for (IdType j = 0; j < P; ++j) {
      if (has_value[j]) {
        C_indices.push_back(j);
        C_weights.push_back(values[j]);
        ++nnz;
        has_value[j] = false;
        values[j] = 0;
      }
    }
    C_indptr_data[i + 1] = nnz;
  }

  return {
      CSRMatrix(M, P, C_indptr, NDArray::FromVector(C_indices), NullArray(), true),
      NDArray::FromVector(C_weights)};
}

template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int32_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int64_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int32_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int64_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);

};  // namespace aten
};  // namespace dgl
