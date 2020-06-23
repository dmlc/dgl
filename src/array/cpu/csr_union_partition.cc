/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_union_partition.cc
 * \brief CSR union and partition
 */
#include <dgl/array.h>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
CSRMatrix DisjointUnionCsrGraph(const std::vector<CSRMatrix>& csrs,
                                const std::vector<uint64_t> src_offset,
                                const std::vector<uint64_t> dst_offset) {
  std::vector<int64_t> indices_offset(csrs.size());
  int64_t total_indices = 0;

  for (size_t i = 0; i < csrs.size(); ++i) {
    aten::CSRMatrix csr = csrs[i];
    indices_offset[i] = total_indices;
    total_indices += csr.indices->shape[0];
  }

  std::vector<IdType> result_indptr(src_offset[csrs.size()] + 1);
  std::vector<IdType> result_indices(total_indices);
  result_indptr[0] = 0;

#pragma omp parallel for
  for (size_t i = 0; i < csrs.size(); ++i) {
    aten::CSRMatrix csr = csrs[i];
    const IdType* indptr_data = static_cast<const IdType*>(csr.indptr->data);
    const IdType* indices_data = static_cast<const IdType*>(csr.indices->data);
    CHECK(IsNullArray(csr.data)) <<
        "DisjointUnionCsrGraph does not support union CSRMatrix with eid mapping data";

    // indptr offset start from 1
    for (size_t j = 1; j <= csr.num_rows; ++j) {
        result_indptr[src_offset[i] + j] = indptr_data[j] + indices_offset[i];
    }

    // Loop over all edges
    for (size_t j = 0; j <= csr.indices->shape[0]; ++j) {
        // node_id + node_shift_offset
        result_indices[indices_offset[i] + j] = indices_data[j] + dst_offset[i];
    }
  }

  return CSRMatrix(
    src_offset[csrs.size()], dst_offset[csrs.size()],
    VecToIdArray(result_indptr, sizeof(IdType) * 8),
    VecToIdArray(result_indices, sizeof(IdType) * 8));
}

template CSRMatrix DisjointUnionCsrGraph<kDLCPU, int32_t>(const std::vector<CSRMatrix>& csrs,
                                                          const std::vector<uint64_t> src_offset,
                                                          const std::vector<uint64_t> dst_offset);
template CSRMatrix DisjointUnionCsrGraph<kDLCPU, int64_t>(const std::vector<CSRMatrix>& csrs,
                                                          const std::vector<uint64_t> src_offset,
                                                          const std::vector<uint64_t> dst_offset);
}  // namespace impl
}  // namespace aten
}  // namespace dgl
