#pragma once

#include <cstdint>

// forward declaration
namespace minigun {
struct Csr;
namespace advance {
struct RuntimeConfig;
}  // namespace advance
}  // namespace minigun

namespace dgl {
namespace kernel {
namespace cuda {

template <typename DType>
struct GData {
  // length along x dimension
  int64_t x_length;
  // input data
  DType *lhs_data, *rhs_data;
  // input id mappings
  int64_t *lhs_mapping, *rhs_mapping;
  // output data
  DType *out_data;
  // output id mapping
  int64_t *out_mapping;
};

template <typename DType, typename IdGetter,
          typename OutSelector, typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    GData<DType>* gdata);

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl
