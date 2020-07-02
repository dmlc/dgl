/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_line_graph.cc
 * \brief CSR LineGraph
 */

#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <iterator>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRLineGraph(const CSRMatrix &csr, bool backtracking) {
  const int64_t nnz = csr->indices->shape[0];
  const IdType *indptr_data = static_cast<IdType *>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType *>(csr.indices->data);
  std::vector<IdType> new_indptr = {0};
  std::vector<IdType> new_indices;
  int64_t num_edges = 0;

  for (int64_t i = 0; i < csr->num_rows; ++i) {
    for (int64_t j = indptr_data[i]; j < indptr_data[i+1]; ++j) {
      IdType u = i;
      IdType v = indices_data[j];

      for (int64_t k = 0; k < csr->num_rows; ++k) {
        for (int64_t l = indptr_data[k]; l < indptr_data[k+1]; ++l) {
          if (k == i && j == l)
            continue;

          // succ_u == v
          if (v == k && (backtracking || (!backtracking && u != indices_data[l]))) {
            num_edges ++;
            new_indices.push_back(l);
          }
        } // for l
      } // for k
    } // for j

    new_indptr.push_back(num_edges);
  }

  CSRMatrix res = CSRMatrix(nnz, nnz, NDArray::FromVector(new_indptr), NDArray::FromVector(new_indices),
    NullArray(), true};
  return res;
}


template CSRMatrix CSRLineGraph<kDLCPU, int32_t>(const CSRMatrix &csr);
template CSRMatrix CSRLineGraph<kDLCPU, int32_t>(const CSRMatrix &csr);

}  // namespace impl
}  // namespace aten
}  // namespace dgl