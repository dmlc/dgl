/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce_impl_decl.h
 * \brief Data structure and function declarations for implementations.
 */
#ifndef DGL_KERNEL_BINARY_REDUCE_IMPL_DECL_H_
#define DGL_KERNEL_BINARY_REDUCE_IMPL_DECL_H_

#include <dgl/runtime/ndarray.h>

#include <string>

#include "./binary_reduce_common.h"

namespace minigun {
namespace advance {
struct RuntimeConfig;
}  // namespace advance
}  // namespace minigun

namespace dgl {

class ImmutableGraph;

namespace kernel {

struct BcastInfo;

template <typename Idx, typename DType>
struct GData {
  // length along x(feature) dimension
  int64_t x_length{0};
  // number of rows of the output tensor
  int64_t out_size{0};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr};
  // output data
  DType *out_data{nullptr};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output id mapping
  Idx *out_mapping{nullptr};
};

template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    GData<Idx, DType>* gdata);

template <int XPU>
void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping);

template <typename Idx, typename DType>
struct BackwardGData {
  // length along x(feature) dimension
  int64_t x_length{0};
  // number of rows of the output tensor
  int64_t out_size{0};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr}, *out_data{nullptr};
  DType *grad_out_data{nullptr};
  // output data
  DType *grad_lhs_data{nullptr}, *grad_rhs_data{nullptr};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output id mapping
  Idx *out_mapping{nullptr};
};

template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardGData<Idx, DType>* gdata);

template <int XPU>
void BackwardBinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data);

// Binary reduce with broadcasting

/*
 * !\brief Data and auxiliary information for binary broadcasting op.
 *
 * Note that all the shapes and strides are for the feature dimensions.
 */
template <int NDim, typename Idx, typename DType>
struct BcastGData {
  int ndim{0};
  // input shape and stride
  int64_t lhs_len{0}, rhs_len{0};
  int64_t lhs_shape[NDim]{0}, lhs_stride[NDim]{0};
  int64_t rhs_shape[NDim]{0}, rhs_stride[NDim]{0};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output shape and stride
  int64_t out_len{0};  // output total length;
  int64_t out_shape[NDim]{0}, out_stride[NDim]{0};
  // output data
  DType *out_data{nullptr};
  // output id mapping
  Idx *out_mapping{nullptr};
};

template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BcastGData<NDim, Idx, DType>* gdata);

template <int XPU>
void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

/*
 * !\brief Data and auxiliary information for backward binary broadcasting op.
 *
 * Note that all the shapes and strides are for the feature dimensions.
 *
 * The gradients of the broadcasting dimensions are not reduced. As a result,
 * The grad_lhs and grad_rhs have the same shape as grad_out.
 */
template <int NDim, typename Idx, typename DType>
struct BackwardBcastGData {
  int ndim{0};
  // input shape and stride
  int64_t lhs_len{0}, rhs_len{0}, out_len{0};
  int64_t lhs_shape[NDim]{0}, lhs_stride[NDim]{0};
  int64_t rhs_shape[NDim]{0}, rhs_stride[NDim]{0};
  int64_t out_shape[NDim]{0}, out_stride[NDim]{0};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr}, *out_mapping{nullptr};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr}, *out_data{nullptr};
  DType *grad_out_data{nullptr};
  // output data
  DType *grad_lhs_data{nullptr}, *grad_rhs_data{nullptr};
};

template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata);

template <int XPU>
void BackwardBinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_IMPL_DECL_H_
