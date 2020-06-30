/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_sort.cc
 * \brief CSR sorting
 */
#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

///////////////////////////// CSRIsSorted /////////////////////////////
template <DLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr) {
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  bool ret = true;
#pragma omp parallel for shared(ret)
  for (int64_t row = 0; row < csr.num_rows; ++row) {
    if (!ret)
      continue;
    for (IdType i = indptr[row] + 1; i < indptr[row + 1]; ++i) {
      if (indices[i - 1] > indices[i]) {
        ret = false;
        break;
      }
    }
  }
  return ret;
}

template bool CSRIsSorted<kDLCPU, int64_t>(CSRMatrix csr);
template bool CSRIsSorted<kDLCPU, int32_t>(CSRMatrix csr);

///////////////////////////// CSRSort /////////////////////////////

template <DLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix* csr) {
  typedef std::pair<IdType, IdType> ShufflePair;
  const int64_t num_rows = csr->num_rows;
  const int64_t nnz = csr->indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr->indptr->data);
  IdType* indices_data = static_cast<IdType*>(csr->indices->data);
  if (!CSRHasData(*csr)) {
    csr->data = aten::Range(0, nnz, csr->indptr->dtype.bits, csr->indptr->ctx);
  }
  IdType* eid_data = static_cast<IdType*>(csr->data->data);
#pragma omp parallel
  {
    std::vector<ShufflePair> reorder_vec;
#pragma omp for
    for (int64_t row = 0; row < num_rows; row++) {
      const int64_t num_cols = indptr_data[row + 1] - indptr_data[row];
      IdType *col = indices_data + indptr_data[row];
      IdType *eid = eid_data + indptr_data[row];

      reorder_vec.resize(num_cols);
      for (int64_t i = 0; i < num_cols; i++) {
        reorder_vec[i].first = col[i];
        reorder_vec[i].second = eid[i];
      }
      std::sort(reorder_vec.begin(), reorder_vec.end(),
                [](const ShufflePair &e1, const ShufflePair &e2) {
                  return e1.first < e2.first;
                });
      for (int64_t i = 0; i < num_cols; i++) {
        col[i] = reorder_vec[i].first;
        eid[i] = reorder_vec[i].second;
      }
    }
  }
  csr->sorted = true;
}

template void CSRSort_<kDLCPU, int64_t>(CSRMatrix* csr);
template void CSRSort_<kDLCPU, int32_t>(CSRMatrix* csr);

template <DLDeviceType XPU, typename IdType, typename TagType>
NDArray CSRSortByTag_(CSRMatrix* csr, IdArray tag, int64_t num_tags) {
  auto indptr_data = static_cast<const IdType *>(csr->indptr->data);
  auto indices_data = static_cast<IdType *>(csr->indices->data);
  if (!aten::CSRHasData(*csr))
    csr->data = aten::Range(0, csr->indices->shape[0], csr->indptr->dtype.bits, csr->indptr->ctx);
  auto eid_data = static_cast<IdType *>(csr->data->data);
  auto tag_data = static_cast<const TagType *>(tag->data);
  int64_t num_rows = csr->num_rows;

  NDArray split = NDArray::Empty({csr->num_rows, num_tags - 1},
      csr->indptr->dtype, csr->indptr->ctx);
  auto split_data = static_cast<IdType *>(split->data);

#pragma omp parallel for
  for (IdType src = 0 ; src < num_rows ; ++src) {
    IdType start = indptr_data[src];
    IdType end = indptr_data[src + 1];

    std::vector<std::vector<IdType>> dst_arr(num_tags);
    std::vector<std::vector<IdType>> eid_arr(num_tags);

    for (IdType ptr = start ; ptr < end ; ++ptr) {
      IdType dst = indices_data[ptr];
      IdType eid = eid_data[ptr];
      TagType t = tag_data[dst];
      CHECK_LT(t, num_tags);
      dst_arr[t].push_back(dst);
      eid_arr[t].push_back(eid);
    }

    IdType *indices_ptr = indices_data + start;
    IdType *eid_ptr = eid_data + start;
    IdType split_ptr = 0;
    for (TagType t = 0 ; t < num_tags ; ++t) {
      std::copy(dst_arr[t].begin(), dst_arr[t].end(), indices_ptr);
      std::copy(eid_arr[t].begin(), eid_arr[t].end(), eid_ptr);
      indices_ptr += dst_arr[t].size();
      eid_ptr += eid_arr[t].size();
      split_ptr += dst_arr[t].size();
      if (t < num_tags - 1)
        split_data[src * (num_tags - 1) + t] = split_ptr;
    }
  }
  csr->sorted = false;
  return split;
}

template IdArray CSRSortByTag_<kDLCPU, int64_t, int64_t>(
    CSRMatrix* csr, IdArray tag, int64_t num_tags);
template IdArray CSRSortByTag_<kDLCPU, int64_t, int32_t>(
    CSRMatrix* csr, IdArray tag, int64_t num_tags);
template IdArray CSRSortByTag_<kDLCPU, int32_t, int64_t>(
    CSRMatrix* csr, IdArray tag, int64_t num_tags);
template IdArray CSRSortByTag_<kDLCPU, int32_t, int32_t>(
    CSRMatrix* csr, IdArray tag, int64_t num_tags);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
