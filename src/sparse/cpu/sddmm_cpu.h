/*!
 *  Copyright (c) 2022 by Contributors
 * \file array/cpu/sddmm_cpu.h
 * \brief SDDMM CPU kernel function header.
 */
#ifndef DGL_SPARSE_CPU_SDDMM_CPU_H_
#define DGL_SPARSE_CPU_SDDMM_CPU_H_

#include <torch/torch.h>





namespace cpu {

/*!
 * \brief CPU kernel of SDDMM on COO format.
 * \param bcast Broadcast information.
 * \param row The COO matrix.
 * \param col The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 * \note it uses edge parallel strategy, different threads are responsible
 *       for the computation of different edges.
 */
template <typename IdType, typename DType>
torch::Tensor SDDMMCoo(torch::Tensor row, torch::Tensor col,
              torch::Tensor val, torch::Tensor matB,
              torch::Tensor matC) {

  const IdType* row_data = row.data_ptr<IdType>();
  const IdType* col_data = col.data_ptr<IdType>();
  const DType* val_data = val.data_ptr<DType>();
  const DType* B_data = matB.data_ptr<DType>();
  const DType* C_data = matC.data_ptr<DType>();

  const int64_t M = matB.size(-2);
  const int64_t N = matC.size(-1);
  const int64_t K = matB.size(-1);
  const int64_t nnz = row.numel();
  torch::Tensor out_val = torch::empty(nnz);
  DType* out_val_data = out_val.data_ptr<DType>();

#pragma omp parallel for
  for (int64_t i = 0; i < nnz; ++i) {
    const IdType rid = row_data[i];
    const IdType cid = col_data[i];
    DType out_val = val_data[i];
    for (int64_t k = 0; k < K; ++k) {
      out_val += B_data[rid * K + k] * C_data[k * N + cid];
    }
    out_val_data[i] = out_val;
  }
  return out_val;
}

}  // namespace cpu

// }  // namespace aten
// }  // namespace dgl


#endif  // DGL_ARRAY_CPU_SDDMM_H_
