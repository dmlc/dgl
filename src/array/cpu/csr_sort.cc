/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/csr_sort.cc
 * @brief CSR sorting
 */
#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

///////////////////////////// CSRIsSorted /////////////////////////////
template <DGLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr) {
  const IdType *indptr = csr.indptr.Ptr<IdType>();
  const IdType *indices = csr.indices.Ptr<IdType>();
  return runtime::parallel_reduce(
      0, csr.num_rows, 1, 1,
      [indptr, indices](size_t b, size_t e, bool ident) {
        for (size_t row = b; row < e; ++row) {
          for (IdType i = indptr[row] + 1; i < indptr[row + 1]; ++i) {
            if (indices[i - 1] > indices[i]) return false;
          }
        }
        return ident;
      },
      [](bool a, bool b) { return a && b; });
}

template bool CSRIsSorted<kDGLCPU, int64_t>(CSRMatrix csr);
template bool CSRIsSorted<kDGLCPU, int32_t>(CSRMatrix csr);

///////////////////////////// CSRSort /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix *csr) {
  typedef std::pair<IdType, IdType> ShufflePair;
  const int64_t num_rows = csr->num_rows;
  const int64_t nnz = csr->indices->shape[0];
  const IdType *indptr_data = static_cast<IdType *>(csr->indptr->data);
  IdType *indices_data = static_cast<IdType *>(csr->indices->data);

  if (CSRIsSorted(*csr)) {
    csr->sorted = true;
    return;
  }

  if (!CSRHasData(*csr)) {
    csr->data = aten::Range(0, nnz, csr->indptr->dtype.bits, csr->indptr->ctx);
  }
  IdType *eid_data = static_cast<IdType *>(csr->data->data);

  runtime::parallel_for(0, num_rows, [=](size_t b, size_t e) {
    for (auto row = b; row < e; ++row) {
      const int64_t num_cols = indptr_data[row + 1] - indptr_data[row];
      std::vector<ShufflePair> reorder_vec(num_cols);
      IdType *col = indices_data + indptr_data[row];
      IdType *eid = eid_data + indptr_data[row];

      for (int64_t i = 0; i < num_cols; i++) {
        reorder_vec[i].first = col[i];
        reorder_vec[i].second = eid[i];
      }
      std::sort(
          reorder_vec.begin(), reorder_vec.end(),
          [](const ShufflePair &e1, const ShufflePair &e2) {
            return e1.first < e2.first;
          });
      for (int64_t i = 0; i < num_cols; i++) {
        col[i] = reorder_vec[i].first;
        eid[i] = reorder_vec[i].second;
      }
    }
  });

  csr->sorted = true;
}

template void CSRSort_<kDGLCPU, int64_t>(CSRMatrix *csr);
template void CSRSort_<kDGLCPU, int32_t>(CSRMatrix *csr);

template <DGLDeviceType XPU, typename IdType, typename TagType>
std::pair<CSRMatrix, NDArray> CSRSortByTag(
    const CSRMatrix &csr, const IdArray tag_array, int64_t num_tags) {
  const auto indptr_data = static_cast<const IdType *>(csr.indptr->data);
  const auto indices_data = static_cast<const IdType *>(csr.indices->data);
  const auto eid_data = aten::CSRHasData(csr)
                            ? static_cast<const IdType *>(csr.data->data)
                            : nullptr;
  const auto tag_data = static_cast<const TagType *>(tag_array->data);
  const int64_t num_rows = csr.num_rows;

  NDArray tag_pos = NDArray::Empty(
      {csr.num_rows, num_tags + 1}, csr.indptr->dtype, csr.indptr->ctx);
  auto tag_pos_data = static_cast<IdType *>(tag_pos->data);
  std::fill(tag_pos_data, tag_pos_data + csr.num_rows * (num_tags + 1), 0);

  aten::CSRMatrix output(
      csr.num_rows, csr.num_cols, csr.indptr.Clone(), csr.indices.Clone(),
      NDArray::Empty(
          {csr.indices->shape[0]}, csr.indices->dtype, csr.indices->ctx),
      csr.sorted);

  auto out_indices_data = static_cast<IdType *>(output.indices->data);
  auto out_eid_data = static_cast<IdType *>(output.data->data);

  runtime::parallel_for(0, num_rows, [&](size_t b, size_t e) {
    for (auto src = b; src < e; ++src) {
      const IdType start = indptr_data[src];
      const IdType end = indptr_data[src + 1];

      auto tag_pos_row = tag_pos_data + src * (num_tags + 1);
      std::vector<IdType> pointer(num_tags, 0);

      for (IdType ptr = start; ptr < end; ++ptr) {
        const IdType eid = eid_data ? eid_data[ptr] : ptr;
        const TagType tag = tag_data[eid];
        CHECK_LT(tag, num_tags);
        ++tag_pos_row[tag + 1];
      }  // count

      for (TagType tag = 1; tag <= num_tags; ++tag) {
        tag_pos_row[tag] += tag_pos_row[tag - 1];
      }  // cumulate

      for (IdType ptr = start; ptr < end; ++ptr) {
        const IdType dst = indices_data[ptr];
        const IdType eid = eid_data ? eid_data[ptr] : ptr;
        const TagType tag = tag_data[eid];
        const IdType offset = tag_pos_row[tag] + pointer[tag];
        CHECK_LT(offset, tag_pos_row[tag + 1]);
        ++pointer[tag];

        out_indices_data[start + offset] = dst;
        out_eid_data[start + offset] = eid;
      }
    }
  });
  output.sorted = false;
  return std::make_pair(output, tag_pos);
}

template std::pair<CSRMatrix, NDArray> CSRSortByTag<kDGLCPU, int64_t, int64_t>(
    const CSRMatrix &csr, const IdArray tag, int64_t num_tags);
template std::pair<CSRMatrix, NDArray> CSRSortByTag<kDGLCPU, int64_t, int32_t>(
    const CSRMatrix &csr, const IdArray tag, int64_t num_tags);
template std::pair<CSRMatrix, NDArray> CSRSortByTag<kDGLCPU, int32_t, int64_t>(
    const CSRMatrix &csr, const IdArray tag, int64_t num_tags);
template std::pair<CSRMatrix, NDArray> CSRSortByTag<kDGLCPU, int32_t, int32_t>(
    const CSRMatrix &csr, const IdArray tag, int64_t num_tags);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
