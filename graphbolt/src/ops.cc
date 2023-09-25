#include "./ops.h"

namespace dgl {
namespace graphbolt {

torch::Tensor AddOne(torch::Tensor tensor) {
  return AddOneImpl<torch::DeviceType::CUDA>(tensor);
}

}  // namespace graphbolt
}  // namespace dgl