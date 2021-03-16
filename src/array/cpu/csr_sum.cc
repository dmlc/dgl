/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/csr_sum.cc
 * \brief CSR Summation
 */

#include <dgl/array.h>
#include <parallel_hashmap/phmap.h>
#include <vector>
#include "array_utils.h"

namespace dgl {

using dgl::runtime::NDArray;

namespace aten {

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A,
    const std::vector<NDArray>& A_weights) {
  CHECK(A.size() > 0) << "List of matrices can't be empty.";
  CHECK_EQ(A.size(), A_weights.size()) << "List of matrices and weights must have same length";
  const int64_t M = A[0].num_rows;
  const int64_t N = A[0].num_cols;
  const int64_t n = A.size();

  std::vector<bool> A_has_eid(n);
  std::vector<const IdType*> A_indptr(n);
  std::vector<const IdType*> A_indices(n);
  std::vector<const IdType*> A_eids(n);
  std::vector<const DType*> A_data(n);

  for (int64_t i = 0; i < n; ++i) {
    const CSRMatrix& csr = A[i];
    const NDArray& data = A_weights[i];
    A_has_eid[i] = !IsNullArray(csr.data);
    A_indptr[i] = csr.indptr.Ptr<IdType>();
    A_indices[i] = csr.indices.Ptr<IdType>();
    A_eids[i] = A_has_eid[i] ? csr.data.Ptr<IdType>() : nullptr;
    A_data[i] = data.Ptr<DType>();
  }

  IdArray C_indptr = IdArray::Empty({M + 1}, A[0].indptr->dtype, A[0].indptr->ctx);
  IdType* C_indptr_data = C_indptr.Ptr<IdType>();
  std::vector<IdType> C_indices;
  std::vector<DType> C_weights;
  IdType nnz = 0;
  C_indptr_data[0] = 0;

  std::vector<bool> has_value(N);
  std::vector<DType> values(N);

  for (IdType i = 0; i < M; ++i) {
    for (int64_t k = 0; k < n; ++k) {
      for (IdType u = A_indptr[k][i]; u < A_indptr[k][i + 1]; ++u) {
        IdType kA = A_indices[k][u];
        DType vA = A_data[k][A_eids[k] ? A_eids[k][u] : u];
        has_value[kA] = true;
        values[kA] += vA;
      }
    }

    for (IdType j = 0; j < N; ++j) {
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
      CSRMatrix(
        M, N, C_indptr, NDArray::FromVector(C_indices), NullArray(), true),
      NDArray::FromVector(C_weights)};
}

template std::pair<CSRMatrix, NDArray> CSRSum<kDLCPU, int32_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLCPU, int64_t, float>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLCPU, int32_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);
template std::pair<CSRMatrix, NDArray> CSRSum<kDLCPU, int64_t, double>(
    const std::vector<CSRMatrix>&, const std::vector<NDArray>&);

};  // namespace aten
};  // namespace dgl
