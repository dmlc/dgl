/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/coo_union_partition.cc
 * \brief COO union and partition
 */
#include <dgl/array.h>
#include <vector>

namespace dgl {
namespace aten {
///////////////////////// COO Based Operations/////////////////////////
COOMatrix DisjointUnionCoo(const std::vector<COOMatrix>& coos) {
  uint64_t src_offset = 0, dst_offset = 0;
  int64_t edge_data_offset = 0;
  bool has_data = false;
  bool row_sorted = true;
  bool col_sorted = true;

  // check if data index array
  for (size_t i = 0; i < coos.size(); ++i) {
    CHECK_SAME_DTYPE(coos[0].row, coos[i].row);
    CHECK_SAME_CONTEXT(coos[0].row, coos[i].row);
    has_data |= COOHasData(coos[i]);
  }

  std::vector<IdArray> res_src;
  std::vector<IdArray> res_dst;
  std::vector<IdArray> res_data;
  res_src.resize(coos.size());
  res_dst.resize(coos.size());

  for (size_t i = 0; i < coos.size(); ++i) {
    const aten::COOMatrix &coo = coos[i];
    row_sorted &= coo.row_sorted;
    col_sorted &= coo.col_sorted;
    IdArray edges_src = coo.row + src_offset;
    IdArray edges_dst = coo.col + dst_offset;
    res_src[i] = edges_src;
    res_dst[i] = edges_dst;
    src_offset += coo.num_rows;
    dst_offset += coo.num_cols;

    // any one of input coo has data index array
    if (has_data) {
      IdArray edges_data;
      if (COOHasData(coo) == false) {
        edges_data = Range(edge_data_offset,
                           edge_data_offset + coo.row->shape[0],
                           coo.row->dtype.bits,
                           coo.row->ctx);
      } else {
        edges_data = coo.data + edge_data_offset;
      }
      res_data.push_back(edges_data);
      edge_data_offset += coo.row->shape[0];
    }
  }

  IdArray result_src = Concat(res_src);
  IdArray result_dst = Concat(res_dst);
  IdArray result_data = has_data ? Concat(res_data) : NullArray();

  return COOMatrix(
    src_offset, dst_offset,
    result_src,
    result_dst,
    result_data,
    row_sorted,
    col_sorted);
}

std::vector<COOMatrix> DisjointPartitionCooBySizes(
  const COOMatrix &coo,
  const uint64_t batch_size,
  const std::vector<uint64_t> &edge_cumsum,
  const std::vector<uint64_t> &src_vertex_cumsum,
  const std::vector<uint64_t> &dst_vertex_cumsum) {
  CHECK_EQ(edge_cumsum.size(), batch_size + 1);
  CHECK_EQ(src_vertex_cumsum.size(), batch_size + 1);
  CHECK_EQ(dst_vertex_cumsum.size(), batch_size + 1);
  std::vector<COOMatrix> ret;
  ret.resize(batch_size);

  for (size_t g = 0; g < batch_size; ++g) {
    IdArray result_src = IndexSelect(coo.row,
                                     edge_cumsum[g],
                                     edge_cumsum[g + 1]) - src_vertex_cumsum[g];
    IdArray result_dst = IndexSelect(coo.col,
                                     edge_cumsum[g],
                                     edge_cumsum[g + 1]) - dst_vertex_cumsum[g];
    IdArray result_data = NullArray();
    // has data index array
    if (COOHasData(coo)) {
      result_data = IndexSelect(coo.data,
                                edge_cumsum[g],
                                edge_cumsum[g + 1]) - edge_cumsum[g];
    }

    COOMatrix sub_coo = COOMatrix(
        src_vertex_cumsum[g+1]-src_vertex_cumsum[g],
        dst_vertex_cumsum[g+1]-dst_vertex_cumsum[g],
        result_src,
        result_dst,
        result_data,
        coo.row_sorted,
        coo.col_sorted);
    ret[g] = sub_coo;
  }

  return ret;
}

///////////////////////// CSR Based Operations/////////////////////////
CSRMatrix DisjointUnionCsr(const std::vector<CSRMatrix>& csrs) {
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
    if (i > 0)
      indptr = IndexSelect(indptr,
                           1,
                           indptr->shape[0]);
    res_indptr[i] = indptr;
    res_indices[i] = indices;
    src_offset += csr.num_rows;
    dst_offset += csr.num_cols;

    // any one of input csr has data index array
    if (has_data) {
      IdArray edges_data;
      if (CSRHasData(csr) == false) {
        edges_data = Range(indices_offset,
                           indices_offset + csr.indices->shape[0],
                           csr.indices->dtype.bits,
                           csr.indices->ctx);
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
    src_offset, dst_offset,
    result_indptr,
    result_indices,
    result_data,
    sorted);
}

std::vector<CSRMatrix> DisjointPartitionCsrBySizes(
  const CSRMatrix &csr,
  const uint64_t batch_size,
  const std::vector<uint64_t> &edge_cumsum,
  const std::vector<uint64_t> &src_vertex_cumsum,
  const std::vector<uint64_t> &dst_vertex_cumsum) {
  CHECK_EQ(edge_cumsum.size(), batch_size + 1);
  CHECK_EQ(src_vertex_cumsum.size(), batch_size + 1);
  CHECK_EQ(dst_vertex_cumsum.size(), batch_size + 1);
  std::vector<CSRMatrix> ret;
  ret.resize(batch_size);

  for (size_t g = 0; g < batch_size; ++g) {
    uint64_t num_src = src_vertex_cumsum[g+1]-src_vertex_cumsum[g];
    IdArray result_indptr;
    if (g == 0) {
      result_indptr = IndexSelect(csr.indptr,
                                  0,
                                  src_vertex_cumsum[1] + 1) - edge_cumsum[0];
    } else {
      result_indptr = IndexSelect(csr.indptr,
                                  src_vertex_cumsum[g],
                                  src_vertex_cumsum[g+1] + 1) - edge_cumsum[g];
    }

    IdArray result_indices = IndexSelect(csr.indices,
                                         edge_cumsum[g],
                                         edge_cumsum[g+1]) - dst_vertex_cumsum[g];

    IdArray result_data = NullArray();
    // has data index array
    if (CSRHasData(csr)) {
      result_data = IndexSelect(csr.data,
                                edge_cumsum[g],
                                edge_cumsum[g+1]) - edge_cumsum[g];
    }

    CSRMatrix sub_csr = CSRMatrix(
      num_src,
      dst_vertex_cumsum[g+1]-dst_vertex_cumsum[g],
      result_indptr,
      result_indices,
      result_data,
      csr.sorted);
    ret[g] = sub_csr;
  }

  return ret;
}

}  // namespace aten
}  // namespace dgl
