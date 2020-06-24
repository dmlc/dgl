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
  std::vector<IdType> result_data(total_indices);
  result_indptr[0] = 0;

#pragma omp parallel for
  for (int64_t i = 0; i < csrs.size(); ++i) {
    aten::CSRMatrix csr = csrs[i];
    const IdType* indptr_data = static_cast<const IdType*>(csr.indptr->data);
    const IdType* indices_data = static_cast<const IdType*>(csr.indices->data);
    const IdType* data = static_cast<const IdType*>(csr.data->data);

    // indptr offset start from 1
    for (int64_t j = 1; j <= csr.num_rows; ++j) {
      result_indptr[src_offset[i] + j] = indptr_data[j] + indices_offset[i];
    }

    // Loop over all edges
    for (int64_t j = 0; j < csr.indices->shape[0]; ++j) {
      // node_id + node_shift_offset
      result_indices[indices_offset[i] + j] = indices_data[j] + dst_offset[i];
      if (data == nullptr) {
        result_data[indices_offset[i] + j] = indices_offset[i] + j;
      } else {
        result_data[indices_offset[i] + j] = indices_offset[i] + data[j];
      }
    }
  }

  return CSRMatrix(
    src_offset[csrs.size()], dst_offset[csrs.size()],
    VecToIdArray(result_indptr, sizeof(IdType) * 8),
    VecToIdArray(result_indices, sizeof(IdType) * 8),
    VecToIdArray(result_data, sizeof(IdType) * 8));
}

template CSRMatrix DisjointUnionCsrGraph<kDLCPU, int32_t>(const std::vector<CSRMatrix>& csrs,
                                                          const std::vector<uint64_t> src_offset,
                                                          const std::vector<uint64_t> dst_offset);
template CSRMatrix DisjointUnionCsrGraph<kDLCPU, int64_t>(const std::vector<CSRMatrix>& csrs,
                                                          const std::vector<uint64_t> src_offset,
                                                          const std::vector<uint64_t> dst_offset);

template <DLDeviceType XPU, typename IdType>
std::vector<CSRMatrix> DisjointPartitionCsrHeteroBySizes(
  const CSRMatrix csr,
  const uint64_t batch_size,
  const std::vector<uint64_t> edge_cumsum,
  const std::vector<uint64_t> src_vertex_cumsum,
  const std::vector<uint64_t> dst_vertex_cumsum) {
  const IdType* indptr_data = static_cast<const IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<const IdType*>(csr.indices->data);
  const IdType* data = static_cast<const IdType*>(csr.data->data);
  std::vector<CSRMatrix> ret;
  ret.resize(batch_size);

#pragma omp parallel for
  for (int64_t g = 0; g < batch_size; ++g) {
    uint64_t num_src = src_vertex_cumsum[g+1]-src_vertex_cumsum[g];
    uint64_t num_edges = edge_cumsum[g+1] - edge_cumsum[g];
    std::vector<IdType> result_indptr(num_src+1);
    std::vector<IdType> result_indices(num_edges);
    std::vector<IdType> result_data;
    if (data != nullptr)
      result_data.resize(num_edges);
    result_indptr[0] = 0;

    for (uint64_t sv = 1; sv <= num_src; ++sv) {
      result_indptr[sv] = indptr_data[src_vertex_cumsum[g] + sv] - edge_cumsum[g];
    }

    for (uint64_t dv = 0; dv < num_edges; ++dv) {
      // dst idx = global_dnidx - dnidx_offset
      result_indices[dv] = indices_data[edge_cumsum[g] + dv] - dst_vertex_cumsum[g];

      if (data != nullptr) {
        // eid = global_eid - eid_offset
        result_data[dv] = data[edge_cumsum[g] + dv] - edge_cumsum[g];
      }
    }

    auto data_array = NullArray();
    if (data != nullptr)
      data_array = VecToIdArray(result_data, sizeof(IdType) * 8);
    CSRMatrix sub_csr = CSRMatrix(
      num_src,
      dst_vertex_cumsum[g+1]-dst_vertex_cumsum[g],
      VecToIdArray(result_indptr, sizeof(IdType) * 8),
      VecToIdArray(result_indices, sizeof(IdType) * 8),
      data_array);
    ret[g] = sub_csr;
  }

  return ret;
}

template std::vector<CSRMatrix> DisjointPartitionCsrHeteroBySizes <kDLCPU, int32_t>(
  const CSRMatrix,
  const uint64_t,
  const std::vector<uint64_t>,
  const std::vector<uint64_t>,
  const std::vector<uint64_t>);
template std::vector<CSRMatrix> DisjointPartitionCsrHeteroBySizes <kDLCPU, int64_t>(
  const CSRMatrix,
  const uint64_t,
  const std::vector<uint64_t>,
  const std::vector<uint64_t>,
  const std::vector<uint64_t>);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
