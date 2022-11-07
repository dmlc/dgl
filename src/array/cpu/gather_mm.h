/**
 *  Copyright (c) 2022 by Contributors
 * @file array/cpu/gather_mm.h
 * @brief GATHER_MM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_GATHER_MM_H_
#define DGL_ARRAY_CPU_GATHER_MM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>

#include <utility>

namespace dgl {
namespace aten {
namespace cpu {

template <typename DType>
void transpose(const DType *in, DType *out, const int N, const int M) {
#pragma omp parallel for
  for (int n = 0; n < N * M; n++) {
    int i = n / N;
    int j = n % N;
    out[n] = in[M * j + i];
  }
}

template <typename DType>
void matmul(
    const DType *A, const DType *B, DType *C, const int M, const int N,
    const int K) {
#pragma omp parallel
  {
    int i, j, k;
#pragma omp for
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        DType local_accum = 0;
        for (k = 0; k < K; k++) {
          local_accum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = local_accum;
      }
    }
  }
}

/**
 * @brief CPU kernel of Gather_mm. The input matrix A is expected to be
 *        sorted according to relation type.
 * @param A The input dense matrix of dimension m x k
 * @param B The input dense matrix of dimension k x n
 * @param C The output dense matrix od dimension m x n
 * @param A_dim1_per_rel The number of rows in each relation in A
 * @param B_dim1_per_rel The number of rows in each relation in B
 * @param a_trans Matrix A to be transposed
 * @param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, typename DType>
void gatherMM_SortedEtype(
    const NDArray A, const NDArray B, NDArray C, const NDArray A_dim1_per_rel,
    const NDArray B_dim1_per_rel, bool a_trans, bool b_trans) {
  assert(A_dim1_per_rel.NumElements() == B_dim1_per_rel.NumElements());
  int64_t num_rel = A_dim1_per_rel.NumElements();
  const DType *A_data = A.Ptr<DType>();
  const DType *B_data = B.Ptr<DType>();
  const IdType *A_rel_data = A_dim1_per_rel.Ptr<IdType>();
  const IdType *B_rel_data = B_dim1_per_rel.Ptr<IdType>();
  DType *C_data = C.Ptr<DType>();

  int64_t A_offset = 0, B_offset = 0, C_offset = 0;
  int64_t m, n, k, h_col, w_row;
  for (int etype = 0; etype < num_rel; ++etype) {
    assert(
        (a_trans)                  ? A_rel_data[etype]
        : A->shape[1] == (b_trans) ? B->shape[1]
                                   : B_rel_data[etype]);
    m = A_rel_data[etype];  // rows of A
    n = B->shape[1];        // cols of B
    k = B_rel_data[etype];  // rows of B == cols of A

    NDArray A_trans, B_trans;
    if (a_trans) {
      A_trans = NDArray::Empty({m * k}, A->dtype, A->ctx);
      transpose<DType>(
          A_data + A_offset, static_cast<DType *>(A_trans->data), m, k);
    }
    if (b_trans) {
      B_trans = NDArray::Empty({k * n}, B->dtype, B->ctx);
      transpose<DType>(
          B_data + B_offset, static_cast<DType *>(B_trans->data), k, n);
    }
    if (a_trans || b_trans) {
      int64_t tmp = k;
      if (a_trans) std::swap(m, k);
      if (b_trans) {
        k = tmp;
        std::swap(n, k);
      }
    }
    matmul<DType>(
        (a_trans) ? static_cast<DType *>(A_trans->data) : A_data + A_offset,
        (b_trans) ? static_cast<DType *>(B_trans->data) : B_data + B_offset,
        C_data + C_offset, m, n, k);
    A_offset += m * k;
    B_offset += k * n;
    C_offset += m * n;
  }
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_GATHER_MM_H_
