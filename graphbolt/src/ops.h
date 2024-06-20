#include <torch/script.h>

namespace dgl {
namespace graphbolt {

template <c10::DeviceType XPU>
torch::Tensor AddOneImpl(torch::Tensor tensor);

torch::Tensor AddOne(torch::Tensor tensor);

}  // namespace graphbolt
}  // namespace dgl