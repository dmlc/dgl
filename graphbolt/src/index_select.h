/**
 *  Copyright (c) 2023 by Contributors
 * @file index_select.h
 * @brief Index select operators.
 */
#ifndef GRAPHBOLT_INDEX_SELECT_H_
#define GRAPHBOLT_INDEX_SELECT_H_

#include <torch/script.h>

namespace graphbolt {
namespace ops {

/**
 * @brief Select columns for a sparse matrix in a CSC format according to nodes
 * tensor.
 *
 * NOTE:
 * 1. The shape of all tensors must be 1-D.
 * 2. If indices is on pinned memory and nodes is on pinned memory or GPU
 * memory, then UVAIndexSelectCSCImpl will be called. If indices is on GPU
 * memory, then IndexSelectCSCImpl will be called. Otherwise,
 * FusedCSCSamplingGraph::InSubgraph will be called.
 *
 * @param indptr Indptr tensor containing offsets with shape (N,).
 * @param indices Indices tensor with edge information of shape (indptr[N],).
 * @param nodes Nodes tensor with shape (M,).
 * @return (torch::Tensor, torch::Tensor) Output indptr and indices tensors of
 * shapes (M + 1,) and ((indptr[nodes + 1] - indptr[nodes]).sum(),).
 */
std::tuple<torch::Tensor, torch::Tensor> IndexSelectCSC(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nodes);

/**
 * @brief Select rows from input tensor according to index tensor.
 *
 * NOTE:
 * 1. The shape of input tensor can be multi-dimensional, but the index tensor
 * must be 1-D.
 * 2. If input is on pinned memory and index is on pinned memory or GPU memory,
 * then UVAIndexSelectImpl will be called. Otherwise, torch::index_select will
 * be called.
 *
 * @param input Input tensor with shape (N, ...).
 * @param index Index tensor with shape (M,).
 * @return torch::Tensor Output tensor with shape (M, ...).
 */
torch::Tensor IndexSelect(torch::Tensor input, torch::Tensor index);

/**
 * @brief Select rows from disk numpy file according to index tensor.
 *
 * NOTE:
 * 1. The shape of input tensor can be multi-dimensional, but the index tensor
 * must be 1-D.
 * 2. The disk feature file should be numpy format and index should be on pinned
 * memory.
 *
 * @param path Input numpy file path (string).
 * @param index Index tensor with shape (M,).
 * @return torch::Tensor Output tensor with shape (M, ...).
 */
torch::Tensor DiskIndexSelect(std::string path, torch::Tensor index);

/**
 * @brief Return the shape of disk numpy feature.
 *
 * @param path Input numpy file path (string).
 * @return torch::Tensor Output tensor shape.
 */
torch::Tensor DiskFeatureShape(std::string path);

}  // namespace ops
}  // namespace graphbolt

#endif  // GRAPHBOLT_INDEX_SELECT_H_
