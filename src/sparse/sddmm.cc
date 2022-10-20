#include <torch/script.h>

#include "cpu/sddmm_cpu.h"
#include "cpu/spmm_cpu.h"

struct SDDMM : public torch::autograd::Function<SDDMM> {
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor row,
                               torch::optional<torch::Tensor> rowptr,
                               torch::Tensor col, torch::Tensor val,
                               torch::Tensor matB, torch::Tensor matC) {
    auto ret = cpu::SDDMMCoo<int64_t, float>(row, col, val, matB, matC);
    ctx->save_for_backward({row, col, val, matB, matC});
    return {ret};
  }
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_input) {
    auto saved = ctx->get_saved_variables();
    auto row = saved[0];
    auto rowptr = saved[1];
    auto col = saved[2];
    auto val = saved[3];
    auto matB = saved[4];
    auto matC = saved[5];

    // TODO(Israt): transpose adj, grad_input[0]
    auto dX = cpu::SpMMSumCsrNaive<int64_t, float>(rowptr, col, val, matB,
                                                   grad_input[0]);
    auto dY = cpu::SpMMSumCsrNaive<int64_t, float>(rowptr, col, val, matC,
                                                   grad_input[0]);

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

struct SDDMMV2 : public torch::autograd::Function<SDDMMV2> {
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               c10::intrusive_ptr<SparseMatrix> m, torch::Tensor matB,
                               torch::Tensor matC) {
    auto ret = cpu::SDDMMCooV2<int64_t, float>(*m->COO(), matB, matC);
    ctx->save_for_backward({m, matB, matC});
    return {ret};
  }
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_input) {
    auto saved = ctx->get_saved_variables();
    auto m = saved[0];
    auto matB = saved[1];
    auto matC = saved[2];

    // TODO(Israt): transpose adj, grad_input[0]
    auto dX =
        cpu::SpMMSumCsrNaiveV2<int64_t, float>(*m->CSR(), matB, grad_input[0]);
    auto dY =
        cpu::SpMMSumCsrNaiveV2<int64_t, float>(*m->CSR(), matC, grad_input[0]);

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }
};

torch::Tensor sddmm(torch::Tensor row, torch::optional<torch::Tensor> rowptr,
                    torch::Tensor col, torch::Tensor val, torch::Tensor matB,
                    torch::Tensor matC) {
  auto result = SDDMM::apply(row, rowptr, col, val, matB, matC);
  return result;
}

torch::Tensor sddmmV2(c10::intrusive_ptr<SparseMatrix> m, torch::Tensor matB, torch::Tensor matC) {
  auto result = SDDMMV2::apply(m, matB, matC);
  return result;
}

TORCH_LIBRARY(dgl_sparse, m) { m.def("SDDMM", sddmm); }
TORCH_LIBRARY(dgl_sparse, m) { m.def("SDDMMV2", sddmmV2); }
