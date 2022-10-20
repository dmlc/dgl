/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/spmm_cpu.h
 * \brief SPMM CPU kernel function header.
 */
#ifndef DGL_SPARSE_CPU_SPMM_CPU_H_
#define DGL_SPARSE_CPU_SPMM_CPU_H_

#include <ATen/Parallel.h>
#include <torch/torch.h>

namespace cpu {

template <typename IdType, typename DType>
torch::Tensor SpMMSumCsrNaive(torch::Tensor rowptr, torch::Tensor cols,
                              torch::Tensor val, torch::Tensor matB,
                              torch::Tensor tmp) {

  const IdType *rowptr_data = rowptr.data_ptr<IdType>();
  const IdType *col_data = cols.data_ptr<IdType>();
  const DType *val_data = val.data_ptr<DType>();
  const DType *B_data = matB.data_ptr<DType>();

  const int64_t M = rowptr.numel() - 1;
  const int64_t N = matB.size(-2);
  const int64_t K = matB.size(-1);
  const int64_t nnz = cols.numel();
  torch::Tensor out = torch::empty(nnz);
  DType *out_data = out.data_ptr<DType>();

  // at::parallel_for(0, M, [&](size_t start, size_t nrows) {
  //   for (auto row = start; row < nrows; ++row) {
  //     for (IdType idx = rowptr[row]; idx < rowptr[row + 1]; ++idx) {
  //       for (int64_t k = 0; k < K; ++k) {
  //           out_data[row * K + k] += val_data[idx] * B_data[col_data[idx] * K
  //           + k];
  //       }
  //     }
  //   }
  // });
  return out;
}

template <typename IdType, typename DType>
torch::Tensor SpMMSumCsrNaiveV2(const CSR &csr, torch::Tensor matB,
                                torch::Tensor tmp) {
  /* SpMM implementation */
}

} // namespace cpu

#endif // DGL_SPARSE_CPU_SPMM_CPU_H_
