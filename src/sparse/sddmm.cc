#include <torch/script.h>
// #include "cpu/sddmm_cpu.h"

// TODO(Israt): Move code to cpu/sddmm_cpu.h
/*! \brief Generalized SDDMM on Coo format. */
template <int XPU, typename IdType, typename DType>
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


struct SDDMM : public torch::autograd::Function<SDDMM> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor row,
                                 torch::Tensor col,
                                 torch::Tensor val,
                                 torch::Tensor matB,
                                 torch::Tensor matC) {
        auto ret = SDDMMCoo<0, int64_t, float>(row, col, val, matB, matC);
        return ret;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_input) {
        // TODO(Israt): Add backward operator
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor sddmm(torch::Tensor row, torch::Tensor col, torch::Tensor val,
    torch::Tensor matB, torch::Tensor matC) {
    auto result = SDDMM::apply(row, col, val, matB, matC);
    return result;
}

TORCH_LIBRARY(dgl_sparse, m) {
  m.def("SDDMM", sddmm);
}
