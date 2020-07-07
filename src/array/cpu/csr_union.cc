/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_sort.cc
 * \brief COO sorting
 */
#include <dgl/array.h>

#include <numeric>
#include <algorithm>
#include <vector>
#include <iterator>
#include <tuple>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
std::tuple<CSRMatrix, IdArray, IdArray>
UnionCsr(const std::vector<CSRMatrix>& csrs) {
  CHECK_EQ(csrs[0].num_rows, csrs[1].num_rows) <<
    "UnionCsr requires CSRs to have the same shape";
  CHECK_EQ(csrs[0].num_cols, csrs[1].num_cols) <<
    "UnionCsr requires CSRs to have the same shape";
  CSRMatrix csr0 = csrs[0].sorted ? csrs[0] : CSRSort(csrs[0]);
  CSRMatrix csr1 = csrs[1].sorted ? csrs[1] : CSRSort(csrs[1]);

  std::vector<IdType> indptr;
  std::vector<IdType> indices;
  const int64_t nnz0 = csr0.indices->shape[0];
  const int64_t nnz1 = csr1.indices->shape[0];
  const IdType *indptr_data0 = static_cast<IdType*>(csr0.indptr->data);
  const IdType *indptr_data1 = static_cast<IdType*>(csr1.indptr->data);
  const IdType *indices_data0 = static_cast<IdType*>(csr0.indices->data);
  const IdType *indices_data1 = static_cast<IdType*>(csr1.indices->data);

  std::vector<IdType> eid_data_data0;
  std::vector<IdType> eid_data_data1;
  eid_data_data0.resize(csr0.indices->shape[0]);
  eid_data_data1.resize(csr1.indices->shape[0]);
  indptr.resize(csr0.num_rows + 1);
  indices.resize(nnz0 + nnz1);
  indptr[0] = 0;

#pragma omp for
  for (int64_t i = 1; i <= csr0.num_rows; ++i) {
    indptr[i] = indptr_data0[i] + indptr_data1[i];
    int64_t j0 = indptr_data0[i-1];
    int64_t j1 = indptr_data1[i-1];

    while (j0 < indptr_data0[i] || j1 < indptr_data1[i]) {
      if (j0 == indptr_data0[i]) {
        indices[j0 + j1] = indices_data1[j1];
        eid_data_data1[j1] = j0 + j1;
        ++j1;
      } else if (j1 == indptr_data1[i]) {
        indices[j0 + j1] = indices_data0[j0];
        eid_data_data0[j0] = j0 + j1;
        ++j0;
      } else {
        if (indices_data0[j0] <= indices_data1[j1]) {
          indices[j0 + j1] = indices_data0[j0];
          eid_data_data0[j0] = j0 + j1;
          ++j0;
        } else {
          indices[j0 + j1] = indices_data1[j1];
          eid_data_data1[j1] = j0 + j1;
          ++j1;
        }
      }
    }
  }

  IdArray eid_data0 = IsNullArray(csr0.data) ?
                      IdArray::FromVector(eid_data_data0) :
                      Scatter(IdArray::FromVector(eid_data_data0), csr0.data);
  IdArray eid_data1 = IsNullArray(csr1.data) ?
                      IdArray::FromVector(eid_data_data1) :
                      Scatter(IdArray::FromVector(eid_data_data1), csr1.data);

  CSRMatrix ret_csr = CSRMatrix(
    csr0.num_rows,
    csr0.num_cols,
    IdArray::FromVector(indptr),
    IdArray::FromVector(indices),
    NullArray(),
    true);

  return std::make_tuple(ret_csr, eid_data0, eid_data1);
}

template std::tuple<CSRMatrix, IdArray, IdArray> UnionCsr<kDLCPU, int64_t>(const std::vector<CSRMatrix>&);
template std::tuple<CSRMatrix, IdArray, IdArray> UnionCsr<kDLCPU, int32_t>(const std::vector<CSRMatrix>&);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
