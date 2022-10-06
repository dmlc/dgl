#include <torch/script.h>

// TODO(Israt): Replace dummy code
torch::Tensor _sddmm(torch::Tensor row,
                        torch::Tensor col) {
	return col;
}

struct SDDMM : public torch::autograd::Function<SDDMM> {
    static torch::Tensor forward(
                        torch::autograd::AutogradContext* ctx,
                        torch::Tensor row,
                        torch::Tensor col) {
        std::cout << "SDDMM->forward" << std::endl;
        auto rst = _sddmm(row, col);
        return rst;
    }
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                torch::autograd::variable_list grad_input) {
        std::cout << "SDDMM-backward" << std::endl;
        return {torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor sddmm(torch::Tensor row, torch::Tensor col) {
    auto result = SDDMM::apply(row, col);
    return result;
}

TORCH_LIBRARY(dgl_sparse, m) {
  m.def("SDDMM", sddmm);
}
