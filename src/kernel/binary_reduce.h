/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce.h
 * \brief Binary reduce function C++ header.
 */
#ifndef DGL_KERNEL_BINARY_REDUCE_H_
#define DGL_KERNEL_BINARY_REDUCE_H_

#include <dgl/runtime/ndarray.h>
#include <dgl/immutable_graph.h>

#include <vector>
#include <string>

#include "./binary_reduce_common.h"

namespace dgl {
namespace kernel {

// Structure for broadcasting shapes
struct BcastInfo {
  // inferred output shape
  std::vector<int64_t> real_out_shape;
  // Following shapes here have been preprocessed, so that:
  //  - The first dimension (for graph) is removed. Shapes here are only for features.
  //  - They have the same number of dimensions.
  //    e.g. (4,) and (3, 4) become (1, 4) and (3, 4)
  //  - Continuous non-broadcasting dimenions are flattened.
  //    e.g. (4, 1, 3, 3) and (4, 5, 3, 3) become (4, 1, 9) and (4, 5, 9)
  std::vector<int64_t> lhs_shape, lhs_stride;
  std::vector<int64_t> rhs_shape, rhs_stride;
  std::vector<int64_t> out_shape, out_stride;
};

/*
 * !\brief Compute the feature shape after binary reduce computation.
 */
std::vector<int64_t> InferBinaryFeatureShape(
    runtime::NDArray lhs,
    runtime::NDArray rhs);

/*
 * !\brief Perform binary operation between the given data and reduce by the graph.
 *
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
void BinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping);

/*
 * !\brief Compute the lhs gradient of BinaryOpReduce
 *
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
void BackwardLhsBinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data);

/*
 * !\brief Compute the rhs gradient of BinaryOpReduce
 *
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
 * \param grad_rhs_data The gradient rhs tensor.
 */
void BackwardRhsBinaryOpReduce(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_rhs_data);

/*
 * !\brief Copy the target data and reduce by graph structure.
 *
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param graph The graph object.
 * \param target The nput target (src, edge)
 * \param in_data The input feature tensor.
 * \param out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 * \param in_mapping An optional int64 id mapping array.
 * \param out_mapping An optional int64 id mapping array.
 */
void CopyReduce(
    const std::string& reducer,
    const ImmutableGraph* graph,
    binary_op::Target target,
    runtime::NDArray in_data, runtime::NDArray out_data,
    runtime::NDArray in_mapping, runtime::NDArray out_mapping);

/*
 * !\brief Compute backward of the CopyReduce
 *
 * \param reducer The type of the reducer ("sum", "max", "mean", "min", "none").
 *                If the reducer is "none", the output is an edge feature tensor.
 *                Otherwise, a node feature tensor is returned.
 * \param graph The graph object.
 * \param target The nput target (src, edge)
 * \param in_mapping An optional int64 id mapping array.
 * \param out_mapping An optional int64 id mapping array.
 * \param in_data The input feature tensor.
 * \param out_data The output tensor. Could be either node or edge feature
 *                  tensor depending on the reducer.
 * \param grad_out_data The gradient output tensor.
 * \param grad_in_data The gradient input tensor.
 */
void BackwardCopyReduce(
    const std::string& reducer,
    const ImmutableGraph* graph,
    binary_op::Target target,
    runtime::NDArray in_mapping,
    runtime::NDArray out_mapping,
    runtime::NDArray in_data,
    runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_in_data);

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_H_
