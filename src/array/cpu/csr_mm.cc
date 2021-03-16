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
  CHECK_EQ(A.num_cols, B.num_cols) << "A's number of columns must equal to B's number of rows";
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
  const int64_t P = B.num_rows;

  std::vector<IdType> C_indptr, C_indices;
  std::vector<DType> C_weights;

  C_indptr.reserve(M + 1);
  C_indptr.push_back(0);
  IdType nnz = 0;

  phmap::flat_hash_map<IdType, DType> map;

  for (IdType i = 0; i < M; ++i) {
    map.clear();
    map.reserve(A_indptr[i + 1] - A_indptr[i]);
    for (IdType u = A_indptr[i]; u < A_indptr[i + 1]; ++u) {
      IdType kA = A_indices[u];
      map.insert({kA, A_data[A_eids ? A_eids[u] : u]});
    }

    for (IdType j = 0; j < P; ++j) {
      bool has_entry = false;
      DType value = 0;
      for (IdType v = B_indptr[j]; v < B_indptr[j + 1]; ++v) {
        IdType kB = B_indices[v];
        const auto it = map.find(kB);
        if (it != map.end()) {
          has_entry = true;
          DType vA = it->second;
          DType vB = B_data[B_eids ? B_eids[v] : v];
          value += vA * vB;
        }
      }
      if (has_entry) {
        ++nnz;
        C_indices.push_back(j);
        C_weights.push_back(value);
      }
    }
    C_indptr.push_back(nnz);
  }

  return {
      CSRMatrix(
        M, P, NDArray::FromVector(C_indptr), NDArray::FromVector(C_indices),
        NullArray(), true),
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
