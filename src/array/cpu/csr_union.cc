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

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
CSRMatrix UnionCsr(const std::vector<CSRMatrix>& csrs) {
  CHECK_EQ(CSRIsSorted(csrs[0]), true) <<
    "Input CSR matrixes of UnionCsr should be sorted";
  CHECK_EQ(CSRIsSorted(csrs[1]), true) <<
    "Input CSR matrixes of UnionCsr should be sorted";
  std::vector<IdType> indptr;
  std::vector<IdType> indices;
  std::vector<IdType> data;
  const int64_t nnz0 = csrs[0].indices->shape[0];
  const int64_t nnz1 = csrs[1].indices->shape[0];
  const IdType *indptr_data0 = static_cast<IdType*>(csrs[0].indptr->data);
  const IdType *indptr_data1 = static_cast<IdType*>(csrs[1].indptr->data);
  const IdType *indices_data0 = static_cast<IdType*>(csrs[0].indices->data);
  const IdType *indices_data1 = static_cast<IdType*>(csrs[1].indices->data);

  //  eids of csrs[0] remains unchanged
  //  eids of csrs[1] will be increased by number of edges of csrs[0]
  IdArray eid_data0 = IsNullArray(csrs[0].data) ?
                                  Range(0, nnz0,
                                        sizeof(IdType) * 8,
                                        csrs[0].indices->ctx) :
                                  csrs[0].data;
  IdArray eid_data1 = IsNullArray(csrs[1].data) ?
                                  Range(nnz0, nnz0 + nnz1,
                                        sizeof(IdType) * 8,
                                        csrs[1].indices->ctx) :
                                  csrs[1].data + nnz0;
  const IdType *eid_data_data0 = static_cast<IdType*>(eid_data0->data);
  const IdType *eid_data_data1 = static_cast<IdType*>(eid_data1->data);
  indptr.resize(csrs[0].num_rows + 1);
  indices.resize(nnz0 + nnz1);
  data.resize(nnz0 + nnz1);
  indptr[0] = 0;

#pragma omp for
  for (int64_t i = 1; i <= csrs[0].num_rows; ++i) {
    indptr[i] = indptr_data0[i] + indptr_data1[i];
    int64_t j0 = indptr_data0[i-1];
    int64_t j1 = indptr_data1[i-1];

    while (j0 < indptr_data0[i] || j1 < indptr_data1[i]) {
      if (j0 == indptr_data0[i]) {
        indices[j0 + j1] = indices_data1[j1];
        data[j0 + j1] = eid_data_data1[j1];
        ++j1;
      } else if (j1 == indptr_data1[i]) {
        indices[j0 + j1] = indices_data0[j0];
        data[j0 + j1] = eid_data_data0[j0];
        ++j0;
      } else {
        if (indices_data0[j0] <= indices_data1[j1]) {
          indices[j0 + j1] = indices_data0[j0];
          data[j0 + j1] = eid_data_data0[j0];
          ++j0;
        } else {
          indices[j0 + j1] = indices_data1[j1];
          data[j0 + j1] = eid_data_data1[j1];
          ++j1;
        }
      }
    }
  }

  return CSRMatrix(
    csrs[0].num_rows,
    csrs[0].num_cols,
    IdArray::FromVector(indptr),
    IdArray::FromVector(indices),
    IdArray::FromVector(data),
    true);
}

template CSRMatrix UnionCsr<kDLCPU, int64_t>(const std::vector<CSRMatrix>&);
template CSRMatrix UnionCsr<kDLCPU, int32_t>(const std::vector<CSRMatrix>&);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
