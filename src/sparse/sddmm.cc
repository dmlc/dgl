#include <torch/script.h>
#include "cpu/sddmm_cpu.h"


struct SDDMM : public torch::autograd::Function<SDDMM> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor row,
                                 torch::optional<torch::Tensor> rowptr,
                                 torch::Tensor col,
                                 torch::Tensor val,
                                 torch::Tensor matB,
                                 torch::Tensor matC) {
        auto ret = cpu::SDDMMCoo<int64_t, float>(row, col, val, matB, matC);
        return ret;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_input) {
        // TODO(Israt): Add backward operator
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor()};
    }
};


torch::Tensor sddmm(torch::Tensor row,
                    torch::optional<torch::Tensor> rowptr,
                    torch::Tensor col, torch::Tensor val,
                    torch::Tensor matB, torch::Tensor matC) {
    auto result = SDDMM::apply(row, rowptr, col, val, matB, matC);
    return result;
}


TORCH_LIBRARY(dgl_sparse, m) {
  m.def("SDDMM", sddmm);
}
