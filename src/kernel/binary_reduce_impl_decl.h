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
#include "./csr_interface.h"

namespace minigun {
namespace advance {
// forward declaration
struct RuntimeConfig;
}  // namespace advance
}  // namespace minigun

namespace dgl {

namespace kernel {

// forward declaration
struct BcastInfo;

///////////////////////////////////////////////////////////////////////////////
// BinaryReduce declarations
///////////////////////////////////////////////////////////////////////////////

/*!\brief Data structure used by computing BinaryOpReduce in Minigun. */
template <typename Idx, typename DType>
struct GData {
  // length along x(feature) dimension
  int64_t x_length{0};
  // size of data, can be single value or a vector
  int64_t data_len{0};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr};
  // output data
  DType *out_data{nullptr};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output id mapping
  Idx *out_mapping{nullptr};
};

/*!
 * \brief Template declaration for BinaryReduce operator.
 *
 * LeftSelector and RightSelector must be one of the four operand target
 * categories.
 *
 * BinaryOp must be one of the binary operator types.
 *
 * Reducer must be one of the reducer types.
 *
 * The implementation of this template is device-dependent
 * (see kernel/xpu/binary_reduce_impl.(cu)h).
 *
 * See definitions in binary_reduce_common.h
 *
 * \tparam XPU the device flag
 * \tparam Idx type of node/edge index (e.g. int32_t, int64_t)
 * \tparam DType type of the feature data (e.g. float32)
 * \tparam LeftSelect lhs category type
 * \tparam RightSelect rhs category type
 * \tparam BinaryOp Binary operator type
 * \tparam Reducer Reducer type
 * \param rtcfg Runtime configuration used by miningun
 * \param graph The graph object.
 * \param gdata The feature and mapping data used by the computation.
 */
template <int XPU, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    GData<Idx, DType>* gdata);

/*!
 * \brief Template declaration for common logics shared by different devices.
 *
 * \tparam XPU the device flag
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param op The type of the binary operator ("mul", "add").
 * \param graph The graph object.
 * \param lhs The lhs target (src, dst, edge)
 * \param rhs The rhs target (src, dst, edge)
 * \param lhs_data The lhs feature tensor.
 * \param rhs_data The rhs feature tensor.
 * \param out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 * \param lhs_mapping An optional int64 id mapping array.
 * \param rhs_mapping An optional int64 id mapping array.
 * \param out_mapping An optional int64 id mapping array.
 */
template <int XPU>
void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping);

///////////////////////////////////////////////////////////////////////////////
// BackwardBinaryReduce declarations
///////////////////////////////////////////////////////////////////////////////

/*!\brief Data structure used by computing BackwardBinaryReduce in Minigun. */
template <typename Idx, typename DType>
struct BackwardGData {
  // length along x(feature) dimension
  int64_t x_length{0};
  // size of data, can be single value or a vector
  int64_t data_len{0};
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

/*!
 * \brief Template declaration for BackwardBinaryReduce operator.
 *
 * Mode must be one of the enum code in binary_op::BackwardMode.
 *
 * LeftSelector and RightSelector must be one of the four operand target
 * categories.
 *
 * BinaryOp must be one of the binary operator types.
 *
 * Reducer must be one of the reducer types.
 *
 * The implementation of this template is device-dependent
 * (see kernel/xpu/backward_binary_reduce_impl.(cu)h).
 *
 * See definitions in binary_reduce_common.h
 *
 * \tparam XPU the device flag
 * \tparam Mode the backward mode code
 * \tparam Idx type of node/edge index (e.g. int32_t, int64_t)
 * \tparam DType type of the feature data (e.g. float32)
 * \tparam LeftSelect lhs category type
 * \tparam RightSelect rhs category type
 * \tparam BinaryOp Binary operator type
 * \tparam Reducer Reducer type
 * \param rtcfg Runtime configuration used by miningun
 * \param graph The graph object.
 * \param gdata The feature and mapping data used by the computation.
 */
template <int XPU, int Mode, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduce(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<Idx, DType>* gdata);

/*!
 * \brief Template declaration for common logics shared by different devices.
 *
 * \tparam XPU the device flag
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param op The type of the binary operator ("mul", "add").
 * \param graph The graph object.
 * \param lhs The lhs target (src, dst, edge)
 * \param rhs The rhs target (src, dst, edge)
 * \param lhs_mapping An optional int64 id mapping array.
 * \param rhs_mapping An optional int64 id mapping array.
 * \param out_mapping An optional int64 id mapping array.
 * \param lhs_data The lhs feature tensor.
 * \param rhs_data The rhs feature tensor.
 * \param out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 * \param grad_out_data The gradient output tensor.
 * \param grad_lhs_data The gradient lhs tensor.
 */
template <int XPU>
void BackwardBinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data);

///////////////////////////////////////////////////////////////////////////////
// BinaryReduce with broadcasting declarations
///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Data structure used by computing BinaryOp with broadcasting in Minigun.
 *
 * Note that all the shapes and strides are for the feature dimensions.
 *
 * \tparam NDim maximum number of feature dimensions
 * \tparam Idx id index type
 * \tparam DType feature data type
 */
template <int NDim, typename Idx, typename DType>
struct BcastGData {
  // actual number of feature dimensions
  int ndim{0};
  // input feature shape and stride
  int64_t lhs_len{0}, rhs_len{0};
  int64_t lhs_shape[NDim]{0}, lhs_stride[NDim]{0};
  int64_t rhs_shape[NDim]{0}, rhs_stride[NDim]{0};
  // size of data, can be single value or a vector
  int64_t data_len{0};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr};
  // output feature shape and stride
  int64_t out_len{0};  // output total feature length (equal to prod(out_shape));
  int64_t out_shape[NDim]{0}, out_stride[NDim]{0};
  // output data
  DType *out_data{nullptr};
  // output id mapping
  Idx *out_mapping{nullptr};
};

/*!
 * \brief Template declaration for BinaryReduce with broadcasting operator.
 *
 * LeftSelector and RightSelector must be one of the four operand target
 * categories.
 *
 * BinaryOp must be one of the binary operator types.
 *
 * Reducer must be one of the reducer types.
 *
 * The implementation of this template is device-dependent
 * (see kernel/xpu/binary_reduce_impl.(cu)h).
 *
 * See definitions in binary_reduce_common.h
 *
 * \tparam XPU the device flag
 * \tparam NDim maximum number of feature dimensions
 * \tparam Idx type of node/edge index (e.g. int32_t, int64_t)
 * \tparam DType type of the feature data (e.g. float32)
 * \tparam LeftSelect lhs category type
 * \tparam RightSelect rhs category type
 * \tparam BinaryOp rinary operator type
 * \tparam Reducer reducer type
 * \param rtcfg runtime configuration used by miningun
 * \param graph The graph object.
 * \param gdata The feature and mapping data used by the computation.
 */
template <int XPU, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BcastGData<NDim, Idx, DType>* gdata);

/*!
 * \brief Template declaration for common logics shared by different devices.
 *
 * \tparam XPU the device flag
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param op The type of the binary operator ("mul", "add").
 * \param graph The graph object.
 * \param lhs The lhs target (src, dst, edge)
 * \param rhs The rhs target (src, dst, edge)
 * \param lhs_data The lhs feature tensor.
 * \param rhs_data The rhs feature tensor.
 * \param out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 * \param lhs_mapping An optional int64 id mapping array.
 * \param rhs_mapping An optional int64 id mapping array.
 * \param out_mapping An optional int64 id mapping array.
 */
template <int XPU>
void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

///////////////////////////////////////////////////////////////////////////////
// BackwardBinaryReduce with broadcasting declarations
///////////////////////////////////////////////////////////////////////////////

/*!
 * \brief Data and auxiliary information for backward binary broadcasting op.
 *
 * Note that all the shapes and strides are for the feature dimensions.
 *
 * The gradients of the broadcasting dimensions are not reduced. As a result,
 * The grad_lhs and grad_rhs have the same shape as grad_out.
 *
 * \tparam NDim maximum number of feature dimensions
 * \tparam Idx id index type
 * \tparam DType feature data type
 */
template <int NDim, typename Idx, typename DType>
struct BackwardBcastGData {
  // actual number of feature dimensions
  int ndim{0};
  // input shape and stride
  int64_t lhs_len{0}, rhs_len{0}, out_len{0};
  int64_t lhs_shape[NDim]{0}, lhs_stride[NDim]{0};
  int64_t rhs_shape[NDim]{0}, rhs_stride[NDim]{0};
  int64_t out_shape[NDim]{0}, out_stride[NDim]{0};
  // size of data, can be single value or a vector
  int64_t data_len{0};
  // input id mappings
  Idx *lhs_mapping{nullptr}, *rhs_mapping{nullptr}, *out_mapping{nullptr};
  // input data
  DType *lhs_data{nullptr}, *rhs_data{nullptr}, *out_data{nullptr};
  DType *grad_out_data{nullptr};
  // output data
  DType *grad_lhs_data{nullptr}, *grad_rhs_data{nullptr};
};

/*!
 * \brief Template declaration for BackwardBinaryReduce with broadcasting operator.
 *
 * LeftSelector and RightSelector must be one of the four operand target
 * categories.
 *
 * BinaryOp must be one of the binary operator types.
 *
 * Reducer must be one of the reducer types.
 *
 * The implementation of this template is device-dependent
 * (see kernel/xpu/binary_reduce_impl.(cu)h).
 *
 * See definitions in binary_reduce_common.h
 *
 * \tparam XPU the device flag
 * \tparam Mode the backward mode code
 * \tparam NDim maximum number of feature dimensions
 * \tparam Idx type of node/edge index (e.g. int32_t, int64_t)
 * \tparam DType type of the feature data (e.g. float32)
 * \tparam LeftSelect lhs category type
 * \tparam RightSelect rhs category type
 * \tparam BinaryOp rinary operator type
 * \tparam Reducer reducer type
 * \param rtcfg runtime configuration used by miningun
 * \param graph The graph object.
 * \param gdata The feature and mapping data used by the computation.
 */
template <int XPU, int Mode, int NDim, typename Idx, typename DType,
          typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
void CallBackwardBinaryReduceBcast(
    const minigun::advance::RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardBcastGData<NDim, Idx, DType>* gdata);

/*!
 * \brief Template declaration for common logics shared by different devices.
 *
 * \tparam XPU the device flag
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param op The type of the binary operator ("mul", "add").
 * \param graph The graph object.
 * \param lhs The lhs target (src, dst, edge)
 * \param rhs The rhs target (src, dst, edge)
 * \param lhs_mapping An optional int64 id mapping array.
 * \param rhs_mapping An optional int64 id mapping array.
 * \param out_mapping An optional int64 id mapping array.
 * \param lhs_data The lhs feature tensor.
 * \param rhs_data The rhs feature tensor.
 * \param out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 * \param grad_out_data The gradient output tensor.
 * \param grad_lhs_data The gradient lhs tensor.
 */
template <int XPU>
void BackwardBinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const CSRWrapper& graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_IMPL_DECL_H_
