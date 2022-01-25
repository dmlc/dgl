/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/gather_mm.h
 * \brief SPMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_GATHER_MM_H_
#define DGL_ARRAY_CPU_GATHER_MM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>

namespace dgl {
namespace aten {
namespace cpu {

template <typename DType>
void transpose(const DType *in, DType *out, const int N, const int M) {
#pragma omp parallel for
    for(int n = 0; n < N * M; n++) {
        int i = n / N;
        int j = n % N;
        out[n] = in[M * j + i];
    }
}

template <typename DType>
void matmul(const DType *A, const DType *B,
    DType *C, const int M, const int N, const int K) {
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
                C[i * N + j ] = local_accum;
            }
        }
    }
}

template <int XPU, typename IdType, typename DType>
void gatherMM_SortedEtype(const NDArray h,
              const NDArray w,
              NDArray out,
              const NDArray h_per_rel,
              const NDArray w_per_rel,
              bool H_trans, bool W_trans) {
    // auto device = runtime::DeviceAPI::Get(h->ctx);
    assert(h_per_rel.NumElements() == w_per_rel.NumElements());
    int64_t num_rel = h_per_rel.NumElements();
    const DType *h_data = h.Ptr<DType>();
    DType *w_data = w.Ptr<DType>();
    const IdType* h_per_rel_data = h_per_rel.Ptr<IdType>();
    const IdType* w_per_rel_data = w_per_rel.Ptr<IdType>();
    DType *out_data = out.Ptr<DType>();

    int64_t h_offset = 0, w_offset = 0, out_offset = 0;
    int64_t m, n, k, h_col, w_row;
    for (int etype = 0; etype < num_rel; ++etype) {
        assert((H_trans) ? h_per_rel_data[etype] : h->shape[1] ==  \
            (W_trans) ? w->shape[1] : w_per_rel_data[etype]);
        m = h_per_rel_data[etype];  // rows of A
        n = w->shape[1];  // cols of B
        k = w_per_rel_data[etype];

        NDArray h_trans, w_trans; // = nullptr;

        if (H_trans) {
            h_trans = NDArray::Empty({m * k}, h->dtype, h->ctx);
            transpose<DType>(h_data + h_offset, static_cast<DType *>(h_trans->data), m, k);
        }
        if (W_trans) {
            w_trans = NDArray::Empty({k * n}, w->dtype, w->ctx);
            transpose<DType>(w_data + w_offset, static_cast<DType *>(w_trans->data), k, n);
        }
        if (H_trans || W_trans) {
            int64_t tmp = k;
            if (H_trans)
                std::swap(m, k);
            if (W_trans)  {
                k = tmp;
                std::swap(n, k);
            }
        }
        matmul<DType>(
            (H_trans) ? static_cast<DType *>(h_trans->data) : h_data + h_offset,
            (W_trans) ? static_cast<DType *>(w_trans->data) : w_data + w_offset,
            out_data + out_offset, m, n, k);
        h_offset += m * k;
        w_offset += k * n;
        out_offset += m * n;
    }
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SPMM_H_





