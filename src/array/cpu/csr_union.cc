/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/coo_sort.cc
 * @brief COO sorting
 */
#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix UnionCsr(const std::vector<CSRMatrix> &csrs) {
  std::vector<IdType> res_indptr;
  std::vector<IdType> res_indices;
  std::vector<IdType> res_data;

  // some preprocess
  // we assume the number of csrs is not large in common cases
  std::vector<IdArray> data;
  std::vector<IdType *> data_data;
  std::vector<IdType *> indptr_data;
  std::vector<IdType *> indices_data;
  int64_t num_edges = 0;
  bool sorted = true;
  for (size_t i = 0; i < csrs.size(); ++i) {
    //  eids of csrs[0] remains unchanged
    //  eids of csrs[1] will be increased by number of edges of csrs[0], etc.
    data.push_back(
        CSRHasData(csrs[i])
            ? csrs[i].data + num_edges
            : Range(
                  num_edges, num_edges + csrs[i].indices->shape[0],
                  csrs[i].indptr->dtype.bits, csrs[i].indptr->ctx));
    data_data.push_back(data[i].Ptr<IdType>());
    indptr_data.push_back(csrs[i].indptr.Ptr<IdType>());
    indices_data.push_back(csrs[i].indices.Ptr<IdType>());
    num_edges += csrs[i].indices->shape[0];
    sorted &= csrs[i].sorted;
  }

  res_indptr.resize(csrs[0].num_rows + 1);
  res_indices.resize(num_edges);
  res_data.resize(num_edges);
  res_indptr[0] = 0;

  if (sorted) {  // all csrs are sorted
#pragma omp for
    for (int64_t i = 1; i <= csrs[0].num_rows; ++i) {
      std::vector<int64_t> indices_off;
      res_indptr[i] = indptr_data[0][i];

      indices_off.push_back(indptr_data[0][i - 1]);
      for (size_t j = 1; j < csrs.size(); ++j) {
        res_indptr[i] += indptr_data[j][i];
        indices_off.push_back(indptr_data[j][i - 1]);
      }

      IdType off = res_indptr[i - 1];
      while (off < res_indptr[i]) {
        IdType min = csrs[0].num_cols + 1;
        int64_t min_idx = -1;
        for (size_t j = 0; j < csrs.size(); ++j) {
          if (indices_off[j] < indptr_data[j][i]) {
            if (min <= indices_data[j][indices_off[j]]) {
              continue;
            } else {
              min = indices_data[j][indices_off[j]];
              min_idx = j;
            }
          }  // for check out of bound
        }    // for
        res_indices[off] = min;
        res_data[off] = data_data[min_idx][indices_off[min_idx]];
        indices_off[min_idx] += 1;
        ++off;
      }     // while
    }       // omp for
  } else {  // some csrs are not sorted
#pragma omp for
    for (int64_t i = 1; i <= csrs[0].num_rows; ++i) {
      IdType off = res_indptr[i - 1];
      res_indptr[i] = 0;

      for (size_t j = 0; j < csrs.size(); ++j) {
        std::memcpy(
            &res_indices[off], &indices_data[j][indptr_data[j][i - 1]],
            sizeof(IdType) * (indptr_data[j][i] - indptr_data[j][i - 1]));
        std::memcpy(
            &res_data[off], &data_data[j][indptr_data[j][i - 1]],
            sizeof(IdType) * (indptr_data[j][i] - indptr_data[j][i - 1]));
        off += indptr_data[j][i] - indptr_data[j][i - 1];
      }
      res_indptr[i] = off;
    }  // omp for
  }

  return CSRMatrix(
      csrs[0].num_rows, csrs[0].num_cols, IdArray::FromVector(res_indptr),
      IdArray::FromVector(res_indices), IdArray::FromVector(res_data), sorted);
}

template CSRMatrix UnionCsr<kDGLCPU, int64_t>(const std::vector<CSRMatrix> &);
template CSRMatrix UnionCsr<kDGLCPU, int32_t>(const std::vector<CSRMatrix> &);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
