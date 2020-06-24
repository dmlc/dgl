/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_union_partition.cc
 * \brief COO union and partition
 */
#include <dgl/array.h>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
COOMatrix DisjointUnionCooGraph(const std::vector<COOMatrix>& coos,
                                const std::vector<uint64_t> src_offset,
                                const std::vector<uint64_t> dst_offset) {
  std::vector<int64_t> edge_offset(coos.size());
  int64_t total_edges = 0;
  bool has_data = false;

  for (size_t i = 0; i < coos.size(); ++i) {
      aten::COOMatrix coo = coos[i];
      edge_offset[i] = total_edges;
      total_edges += coo.row->shape[0];

      if (IsNullArray(coo.data) == false)
        has_data = true;
  }

  std::vector<IdType> result_src(total_edges);
  std::vector<IdType> result_dst(total_edges);
  std::vector<IdType> result_data;
  if (has_data)
    result_data.resize(total_edges);

#pragma omp parallel for
  for (int64_t i = 0; i < coos.size(); ++i) {
    aten::COOMatrix coo = coos[i];
    int64_t num_edges = coo.row->shape[0];

    const IdType* edges_src_data = static_cast<const IdType*>(coo.row->data);
    const IdType* edges_dst_data = static_cast<const IdType*>(coo.col->data);
    const IdType* data = static_cast<const IdType*>(coo.data->data);

    // Loop over all edges
    for (int64_t j = 0; j < num_edges; ++j) {
      result_src[edge_offset[i] + j] = edges_src_data[j] + src_offset[i];
      result_dst[edge_offset[i] + j] = edges_dst_data[j] + dst_offset[i];

      if (has_data) {
        if (data == nullptr) {
          result_data[edge_offset[i] + j] = edge_offset[i] + j;
        } else {
          result_data[edge_offset[i] + j] = edge_offset[i] + data[j];
        }
      }
    }
  }

  return COOMatrix(
    src_offset[coos.size()], dst_offset[coos.size()],
    VecToIdArray(result_src, sizeof(IdType) * 8),
    VecToIdArray(result_dst, sizeof(IdType) * 8));
}

template COOMatrix DisjointUnionCooGraph<kDLCPU, int32_t>(const std::vector<COOMatrix>&,
                                                          const std::vector<uint64_t>,
                                                          const std::vector<uint64_t>);
template COOMatrix DisjointUnionCooGraph<kDLCPU, int64_t>(const std::vector<COOMatrix>&,
                                                          const std::vector<uint64_t>,
                                                          const std::vector<uint64_t>);

template <DLDeviceType XPU, typename IdType>
std::vector<COOMatrix> DisjointPartitionHeteroBySizes(
  const COOMatrix coo,
  const uint64_t batch_size,
  const std::vector<uint64_t> edge_cumsum,
  const std::vector<uint64_t> src_vertex_cumsum,
  const std::vector<uint64_t> dst_vertex_cumsum) {
  const IdType* edges_src_data = static_cast<const IdType*>(coo.row->data);
  const IdType* edges_dst_data = static_cast<const IdType*>(coo.col->data);
  CHECK(IsNullArray(coo.data)) <<
        "DisjointPartitionHeteroBySizes does not support input COOMatrix with eid mapping data";
  std::vector<COOMatrix> ret;
  ret.resize(batch_size);

#pragma omp parallel for
  for (int64_t g = 0; g < batch_size; ++g) {
    std::vector<IdType> result_src, result_dst;
    for (uint64_t e = edge_cumsum[g]; e < edge_cumsum[g]; ++e) {
      result_src.push_back(edges_src_data[e] - src_vertex_cumsum[g]);
      result_dst.push_back(edges_dst_data[e] - dst_vertex_cumsum[g]);
    }

    COOMatrix sub_coo = COOMatrix(
        src_vertex_cumsum[batch_size], dst_vertex_cumsum[batch_size],
        VecToIdArray(result_src, sizeof(IdType) * 8),
        VecToIdArray(result_dst, sizeof(IdType) * 8));
    ret[g] = sub_coo;
  }

  return ret;
}

template std::vector<COOMatrix>
    DisjointPartitionHeteroBySizes<kDLCPU, int32_t>(const COOMatrix,
                                                    const uint64_t,
                                                    const std::vector<uint64_t>,
                                                    const std::vector<uint64_t>,
                                                    const std::vector<uint64_t>);

template std::vector<COOMatrix>
    DisjointPartitionHeteroBySizes<kDLCPU, int64_t>(const COOMatrix,
                                                    const uint64_t,
                                                    const std::vector<uint64_t>,
                                                    const std::vector<uint64_t>,
                                                    const std::vector<uint64_t>);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
