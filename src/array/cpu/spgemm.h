#ifndef DGL_ARRAY_CPU_SPGEMM_H_
#define DGL_ARRAY_CPU_SPGEMM_H_

namespace dgl {
namespace aten {
namespace cpu {

/*!
 * \brief Sparse GEMM:
 *
 * alpha * A * B.T + beta * C
 */
template <typename IdType, typename DType>
void SpSpMMCsr(
    const CSRMatrix& a,
    const NDArray* a_weights,
    const CSRMatrix& b,
    const NDArray* b_weights,
    CSRMatrix* c,
    NDArray* c_weights) {
  int64_t n = a.num_rows;
  int64_t m = a.num_cols;
  int64_t k = b.num_cols;

  std::vector<IdType> c_indices;
  std::vector<DType> c_data;

  auto ctx = a.indptr->ctx;

  const IdType* a_indptr = a.indptr.Ptr<IdType>();
  const IdType* a_indices = a.indices.Ptr<IdType>();
  const IdType* a_eids = CSRHasData(a) ? a.data.Ptr<IdType>() : nullptr;
  const IdType* b_indptr = b.indptr.Ptr<IdType>();
  const IdType* b_indices = b.indices.Ptr<IdType>();
  const IdType* b_eids = CSRHasData(b) ? b.data.Ptr<IdType>() : nullptr;

  const DType* a_data = a_weights ? a_weights->Ptr<DType>() : nullptr;
  const DType* b_data = b_weights ? b_weights->Ptr<DType>() : nullptr;

  c->indptr = NewIdArray(n + 1, ctx, a.indptr->dtype.bits);
  IdType* c_indptr = c.indptr.Ptr<IdType>();

  phmap::flat_hash_map<IdType, DType> set;
  c_indptr[0] = 0;
  int64_t nnz = 0;

  for (IdType a_i = 0; a_i < n; ++a_i) {
    set.clear();

    for (IdType a_off = a_indptr[a_i]; a_off < a_indptr[a_i + 1]; ++a_off) {
      a_j = a_indices[a_off];
      for (IdType b_off = b_indptr[b_j]; b_off < b_indptr[b_j + 1]; ++b_off) {
        b_j = b_indices[b_off];

        a_val = (!a_data) ? 1 : (a_eids ? a_data[a_eids[a_j]] : a_data[a_j]);
        b_val = (!b_data) ? 1 : (b_eids ? b_data[b_eids[b_j]] : b_data[b_j]);

        set[b_j] += a_val * b_val;
      }
    }

    nnz += set.size();
    c_indptr[a_i + 1] = nnz;
    c_indices.reserve(nnz);
    if (c_weights)
      c_data.reserve(nnz);
    for (auto it : set) {
      c_indices.push_back(it.first);
      if (c_weights)
        c_data.push_back(it.second);
    }
  }

  c->indices = NDArray::FromVector(c_indices, ctx);
  if (c_weights)
    *c_weights = NDArray::FromVector(c_data, ctx);
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPGEMM_H_
