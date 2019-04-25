#ifndef DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_H_

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

// Binary reduce

template <typename DType>
struct GData {
  // length along x dimension
  int64_t x_length{0};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr};
  // input id mappings
  int64_t *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output data
  DType *out_data{nullptr};
  // output id mapping
  int64_t *out_mapping{nullptr};
};

template <typename DType, typename IdGetter,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    GData<DType>* gdata);

// Binary reduce with broadcasting

template <typename DType>
struct BcastGData {
  int ndim{0};
  // input shape and stride
  int64_t lhs_len{0}, rhs_len{0};
  int64_t *lhs_shape{nullptr}, *lhs_stride{nullptr};
  int64_t *rhs_shape{nullptr}, *rhs_stride{nullptr};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr};
  // input id mappings
  int64_t *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output shape and stride
  int64_t out_len{0};  // output total length;
  int64_t *out_shape{nullptr}, *out_stride{nullptr};
  // output data
  DType *out_data{nullptr};
  // output id mapping
  int64_t *out_mapping{nullptr};
};

template <typename DType, typename IdGetter,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const minigun::Csr& csr,
    BcastGData<DType>* gdata);

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_BINARY_REDUCE_IMPL_H_
