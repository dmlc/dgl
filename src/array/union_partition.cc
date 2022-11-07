/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/coo_union_partition.cc
 * @brief COO union and partition
 */
#include <dgl/array.h>

#include <vector>

namespace dgl {
namespace aten {
///////////////////////// COO Based Operations/////////////////////////
std::vector<COOMatrix> DisjointPartitionCooBySizes(
    const COOMatrix &coo, const uint64_t batch_size,
    const std::vector<uint64_t> &edge_cumsum,
    const std::vector<uint64_t> &src_vertex_cumsum,
    const std::vector<uint64_t> &dst_vertex_cumsum) {
  CHECK_EQ(edge_cumsum.size(), batch_size + 1);
  CHECK_EQ(src_vertex_cumsum.size(), batch_size + 1);
  CHECK_EQ(dst_vertex_cumsum.size(), batch_size + 1);
  std::vector<COOMatrix> ret;
  ret.resize(batch_size);

  for (size_t g = 0; g < batch_size; ++g) {
    IdArray result_src =
        IndexSelect(coo.row, edge_cumsum[g], edge_cumsum[g + 1]) -
        src_vertex_cumsum[g];
    IdArray result_dst =
        IndexSelect(coo.col, edge_cumsum[g], edge_cumsum[g + 1]) -
        dst_vertex_cumsum[g];
    IdArray result_data = NullArray();
    // has data index array
    if (COOHasData(coo)) {
      result_data = IndexSelect(coo.data, edge_cumsum[g], edge_cumsum[g + 1]) -
                    edge_cumsum[g];
    }

    COOMatrix sub_coo = COOMatrix(
        src_vertex_cumsum[g + 1] - src_vertex_cumsum[g],
        dst_vertex_cumsum[g + 1] - dst_vertex_cumsum[g], result_src, result_dst,
        result_data, coo.row_sorted, coo.col_sorted);
    ret[g] = sub_coo;
  }

  return ret;
}

COOMatrix COOSliceContiguousChunk(
    const COOMatrix &coo, const std::vector<uint64_t> &edge_range,
    const std::vector<uint64_t> &src_vertex_range,
    const std::vector<uint64_t> &dst_vertex_range) {
  IdArray result_src = NullArray(coo.row->dtype, coo.row->ctx);
  IdArray result_dst = NullArray(coo.row->dtype, coo.row->ctx);
  if (edge_range[1] != edge_range[0]) {
    // The chunk has edges
    result_src = IndexSelect(coo.row, edge_range[0], edge_range[1]) -
                 src_vertex_range[0];
    result_dst = IndexSelect(coo.col, edge_range[0], edge_range[1]) -
                 dst_vertex_range[0];
  }

  IdArray result_data = NullArray();
  // has data index array
  if (COOHasData(coo)) {
    result_data =
        IndexSelect(coo.data, edge_range[0], edge_range[1]) - edge_range[0];
  }

  COOMatrix sub_coo = COOMatrix(
      src_vertex_range[1] - src_vertex_range[0],
      dst_vertex_range[1] - dst_vertex_range[0], result_src, result_dst,
      result_data, coo.row_sorted, coo.col_sorted);

  return sub_coo;
}

///////////////////////// CSR Based Operations/////////////////////////
CSRMatrix DisjointUnionCsr(const std::vector<CSRMatrix> &csrs) {
  uint64_t src_offset = 0, dst_offset = 0;
  int64_t indices_offset = 0;
  bool has_data = false;
  bool sorted = true;

  // check if data index array
  for (size_t i = 0; i < csrs.size(); ++i) {
    CHECK_SAME_DTYPE(csrs[0].indptr, csrs[i].indptr);
    CHECK_SAME_CONTEXT(csrs[0].indices, csrs[i].indices);
    has_data |= CSRHasData(csrs[i]);
  }

  std::vector<IdArray> res_indptr;
  std::vector<IdArray> res_indices;
  std::vector<IdArray> res_data;
  res_indptr.resize(csrs.size());
  res_indices.resize(csrs.size());

  for (size_t i = 0; i < csrs.size(); ++i) {
    const aten::CSRMatrix &csr = csrs[i];
    sorted &= csr.sorted;
    IdArray indptr = csr.indptr + indices_offset;
    IdArray indices = csr.indices + dst_offset;
    if (i > 0) indptr = IndexSelect(indptr, 1, indptr->shape[0]);
    res_indptr[i] = indptr;
    res_indices[i] = indices;
    src_offset += csr.num_rows;
    dst_offset += csr.num_cols;

    // any one of input csr has data index array
    if (has_data) {
      IdArray edges_data;
      if (CSRHasData(csr) == false) {
        edges_data = Range(
            indices_offset, indices_offset + csr.indices->shape[0],
            csr.indices->dtype.bits, csr.indices->ctx);
      } else {
        edges_data = csr.data + indices_offset;
      }
      res_data.push_back(edges_data);
      indices_offset += csr.indices->shape[0];
    }
  }

  IdArray result_indptr = Concat(res_indptr);
  IdArray result_indices = Concat(res_indices);
  IdArray result_data = has_data ? Concat(res_data) : NullArray();

  return CSRMatrix(
      src_offset, dst_offset, result_indptr, result_indices, result_data,
      sorted);
}

std::vector<CSRMatrix> DisjointPartitionCsrBySizes(
    const CSRMatrix &csr, const uint64_t batch_size,
    const std::vector<uint64_t> &edge_cumsum,
    const std::vector<uint64_t> &src_vertex_cumsum,
    const std::vector<uint64_t> &dst_vertex_cumsum) {
  CHECK_EQ(edge_cumsum.size(), batch_size + 1);
  CHECK_EQ(src_vertex_cumsum.size(), batch_size + 1);
  CHECK_EQ(dst_vertex_cumsum.size(), batch_size + 1);
  std::vector<CSRMatrix> ret;
  ret.resize(batch_size);

  for (size_t g = 0; g < batch_size; ++g) {
    uint64_t num_src = src_vertex_cumsum[g + 1] - src_vertex_cumsum[g];
    IdArray result_indptr;
    if (g == 0) {
      result_indptr =
          IndexSelect(csr.indptr, 0, src_vertex_cumsum[1] + 1) - edge_cumsum[0];
    } else {
      result_indptr =
          IndexSelect(
              csr.indptr, src_vertex_cumsum[g], src_vertex_cumsum[g + 1] + 1) -
          edge_cumsum[g];
    }

    IdArray result_indices =
        IndexSelect(csr.indices, edge_cumsum[g], edge_cumsum[g + 1]) -
        dst_vertex_cumsum[g];

    IdArray result_data = NullArray();
    // has data index array
    if (CSRHasData(csr)) {
      result_data = IndexSelect(csr.data, edge_cumsum[g], edge_cumsum[g + 1]) -
                    edge_cumsum[g];
    }

    CSRMatrix sub_csr = CSRMatrix(
        num_src, dst_vertex_cumsum[g + 1] - dst_vertex_cumsum[g], result_indptr,
        result_indices, result_data, csr.sorted);
    ret[g] = sub_csr;
  }

  return ret;
}

CSRMatrix CSRSliceContiguousChunk(
    const CSRMatrix &csr, const std::vector<uint64_t> &edge_range,
    const std::vector<uint64_t> &src_vertex_range,
    const std::vector<uint64_t> &dst_vertex_range) {
  int64_t indptr_len = src_vertex_range[1] - src_vertex_range[0] + 1;
  IdArray result_indptr =
      Full(0, indptr_len, csr.indptr->dtype.bits, csr.indptr->ctx);
  IdArray result_indices = NullArray(csr.indptr->dtype, csr.indptr->ctx);
  IdArray result_data = NullArray();
  if (edge_range[1] != edge_range[0]) {
    // The chunk has edges
    result_indptr =
        IndexSelect(csr.indptr, src_vertex_range[0], src_vertex_range[1] + 1) -
        edge_range[0];
    result_indices = IndexSelect(csr.indices, edge_range[0], edge_range[1]) -
                     dst_vertex_range[0];
    if (CSRHasData(csr)) {
      result_data =
          IndexSelect(csr.data, edge_range[0], edge_range[1]) - edge_range[0];
    }
  }

  CSRMatrix sub_csr = CSRMatrix(
      src_vertex_range[1] - src_vertex_range[0],
      dst_vertex_range[1] - dst_vertex_range[0], result_indptr, result_indices,
      result_data, csr.sorted);

  return sub_csr;
}

}  // namespace aten
}  // namespace dgl
