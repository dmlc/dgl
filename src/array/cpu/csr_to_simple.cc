/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_to_simple.cc
 * \brief CSR sorting
 */
#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRToSimple(const CSRMatrix& csr) {
  CHECK_EQ(csr.sorted, true) << "Input csr of CSRToSimple should be sorted";

  const IdType *indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType *indices_data = static_cast<IdType*>(csr.indices->data);
  // TODO(xiangsx): FIXME csr.data is dropped.
  std::vector<IdType> indptr;
  std::vector<IdType> indices;
  indptr.resize(csr.indptr->shape[0]);
  indptr[0] = 0;

  for (int64_t i = 1; i < csr.indptr->shape[0]; ++i) {
    if (indptr_data[i-1] == indptr_data[i]) {
      indptr[i] = indptr[i-1];
      continue;
    }
    
    int64_t cnt = 1;
    indices.push_back(indices_data[indptr_data[i-1]]);
    for (int64_t j = indptr_data[i-1]+1; j < indptr_data[i]; ++j) {
      if (indices_data[j-1] == indices_data[j])
        continue;
      indices.push_back(indices_data[j]);
      ++cnt;
    }
    indptr[i] = indptr[i-1] + cnt;
  }

  return CSRMatrix(
    csr.num_rows,
    csr.num_cols,
    IdArray::FromVector(indptr),
    IdArray::FromVector(indices),
    NullArray(),
    true);
}

template CSRMatrix CSRToSimple<kDLCPU, int32_t>(const CSRMatrix&);
template CSRMatrix CSRToSimple<kDLCPU, int64_t>(const CSRMatrix&);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
