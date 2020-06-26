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
COOMatrix DisjointUnionCooGraph(const std::vector<COOMatrix>& coos) {
  uint64_t src_offset = 0, dst_offset = 0;
  int64_t edge_data_offset = 0;
  bool has_data = false;

  // check if data index array
  for (size_t i = 0; i < coos.size(); ++i) {
      if (IsNullArray(coos[i].data) == false)
        has_data = true;
  }

  std::vector<IdArray> res_src;
  std::vector<IdArray> res_dst;
  std::vector<IdArray> res_data;
  for (size_t i = 0; i < coos.size(); ++i) {
    aten::COOMatrix coo = coos[i];
    IdArray edges_src = coo.row + src_offset;
    IdArray edges_dst = coo.col + dst_offset;
    res_src.push_back(edges_src);
    res_dst.push_back(edges_dst);
    src_offset += coo.num_rows;
    dst_offset += coo.num_cols;

    // any one of input coo has data index array
    if (has_data) {
      IdArray edges_data;
      if (IsNullArray(coo.data)) {
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
  IdArray result_data = NullArray();
  if (has_data)
    result_data = Concat(res_data);

  return COOMatrix(
    src_offset, dst_offset,
    result_src,
    result_dst,
    result_data);
}

std::vector<COOMatrix> DisjointPartitionCooBySizes(
  const COOMatrix coo,
  const uint64_t batch_size,
  const std::vector<uint64_t> edge_cumsum,
  const std::vector<uint64_t> src_vertex_cumsum,
  const std::vector<uint64_t> dst_vertex_cumsum) {
  CHECK(IsNullArray(coo.data)) <<
        "DisjointPartitionHeteroBySizes does not support input COOMatrix with eid mapping data";
  std::vector<COOMatrix> ret;
  ret.resize(batch_size);

  for (size_t g = 0; g < batch_size; ++g) {
    IdArray result_src = IndexSelect(coo.row,
                                     Range(edge_cumsum[g],
                                           edge_cumsum[g + 1],
                                           coo.row->dtype.bits,
                                           coo.row->ctx)) - src_vertex_cumsum[g];
    IdArray result_dst = IndexSelect(coo.col,
                                     Range(edge_cumsum[g],
                                           edge_cumsum[g + 1],
                                           coo.col->dtype.bits,
                                           coo.col->ctx)) - dst_vertex_cumsum[g];
    IdArray result_data = NullArray();
    // has data index array
    if (IsNullArray(coo.data) == false) {
      result_data = IndexSelect(coo.data,
                                Range(edge_cumsum[g],
                                      edge_cumsum[g + 1],
                                      coo.data->dtype.bits,
                                      coo.data->ctx)) - edge_cumsum[g];
    }

    COOMatrix sub_coo = COOMatrix(
        src_vertex_cumsum[g+1]-src_vertex_cumsum[g],
        dst_vertex_cumsum[g+1]-dst_vertex_cumsum[g],
        result_src,
        result_dst,
        result_data);
    ret[g] = sub_coo;
  }

  return ret;
}

///////////////////////// CSR Based Operations/////////////////////////
CSRMatrix DisjointUnionCsrGraph(const std::vector<CSRMatrix>& csrs) {
  uint64_t src_offset = 0, dst_offset = 0;
  int64_t indices_offset = 0;
  bool has_data = false;

  // check if data index array
  for (size_t i = 0; i < csrs.size(); ++i) {
      if (IsNullArray(csrs[i].data) == false)
        has_data = true;
  }

  std::vector<IdArray> res_indptr;
  std::vector<IdArray> res_indices;
  std::vector<IdArray> res_data;

  for (size_t i = 0; i < csrs.size(); ++i) {
    aten::CSRMatrix csr = csrs[i];
    IdArray indptr = csr.indptr + src_offset;
    IdArray indices = csr.indices + dst_offset;
    if (i > 0)
      indptr = IndexSelect(indptr,
                           Range(1,
                                 indptr->shape[0],
                                 indptr->dtype.bits,
                                 indptr->ctx));
    res_indptr.push_back(indptr);
    res_indices.push_back(indices);
    src_offset += csr.num_rows;
    dst_offset += csr.num_cols;

    // any one of input csr has data index array
    if (has_data) {
      IdArray edges_data;
      if (IsNullArray(csr.data)) {
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
  IdArray result_data = NullArray();
  if (has_data)
    result_data = Concat(res_data);

  return CSRMatrix(
    src_offset, dst_offset,
    result_indptr,
    result_indices,
    result_data);
}

std::vector<CSRMatrix> DisjointPartitionCsrBySizes(
  const CSRMatrix csr,
  const uint64_t batch_size,
  const std::vector<uint64_t> edge_cumsum,
  const std::vector<uint64_t> src_vertex_cumsum,
  const std::vector<uint64_t> dst_vertex_cumsum) {
  std::vector<CSRMatrix> ret;
  ret.resize(batch_size);

  for (size_t g = 0; g < batch_size; ++g) {
    uint64_t num_src = src_vertex_cumsum[g+1]-src_vertex_cumsum[g];
    IdArray result_indptr;
    if (g == 0) {
      result_indptr = IndexSelect(csr.indptr,
                                  Range(0,
                                        src_vertex_cumsum[g+1] + 1,
                                        csr.indptr->dtype.bits,
                                        csr.indptr->ctx)) + edge_cumsum[g];
    } else {
      IdArray result_indptr_0 = Full(0, 1,
                                    csr.indptr->dtype.bits,
                                    csr.indptr->ctx);
      result_indptr = IndexSelect(csr.indptr,
                                  Range(src_vertex_cumsum[g] + 1,
                                        src_vertex_cumsum[g+1] + 1,
                                        csr.indptr->dtype.bits,
                                        csr.indptr->ctx)) + edge_cumsum[g];
      result_indptr = Concat(std::vector<IdArray>({result_indptr_0, result_indptr}));
    }

    IdArray result_indices = IndexSelect(csr.indices,
                                         Range(edge_cumsum[g],
                                               edge_cumsum[g+1],
                                               csr.indices->dtype.bits,
                                               csr.indices->ctx)) - dst_vertex_cumsum[g];

    IdArray result_data = NullArray();
    // has data index array
    if (IsNullArray(csr.data) == false) {
      result_data = IndexSelect(csr.data,
                                Range(edge_cumsum[g],
                                      edge_cumsum[g + 1],
                                      csr.data->dtype.bits,
                                      csr.data->ctx)) - edge_cumsum[g];
    }

    CSRMatrix sub_csr = CSRMatrix(
      num_src,
      dst_vertex_cumsum[g+1]-dst_vertex_cumsum[g],
      result_indptr,
      result_indices,
      result_data);
    ret[g] = sub_csr;
  }

  return ret;
}

}  // namespace aten
}  // namespace dgl
