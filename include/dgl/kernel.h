/**
 *  Copyright (c) 2020 by Contributors
 * @file dgl/aten/kernel.h
 * @brief Sparse matrix operators.
 */
#ifndef DGL_KERNEL_H_
#define DGL_KERNEL_H_

#include <string>
#include <utility>
#include <vector>

#include "./base_heterograph.h"
#include "./bcast.h"
#include "array.h"

namespace dgl {
namespace aten {

/**
 * @brief Generalized Sparse Matrix-Matrix Multiplication.
 * @param op The binary operator, could be `add`, `sub', `mul`, 'div',
 *        `copy_u`, `copy_e'.
 * @param op The reduce operator, could be `sum`, `min`, `max'.
 * @param graph The graph we apply SpMM on.
 * @param ufeat The source node feature.
 * @param efeat The edge feature.
 * @param out The output feature on destination nodes.
 * @param out_aux A list of NDArray's that contains auxiliary information such
 *        as the argmax on source nodes and edges for reduce operators such as
 *        `min` and `max`.
 */
void SpMM(
    const std::string& op, const std::string& reduce, HeteroGraphPtr graph,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

void SpMM(
    const char* op, const char* reduce, HeteroGraphPtr graph, NDArray ufeat,
    NDArray efeat, NDArray out, std::vector<NDArray> out_aux);

/**
 * @brief Generalized Sampled Dense-Dense Matrix Multiplication.
 * @param op The binary operator, could be `add`, `sub', `mul`, 'div',
 *        `dot`, `copy_u`, `copy_e'.
 * @param graph The graph we apply SpMM on.
 * @param ufeat The source node feature.
 * @param vfeat The destination node feature.
 * @param out The output feature on edge.
 * @param lhs_target Type of `ufeat`. (0: source; 1: edge; 2: destination)
 * @param rhs_target Type of `vfeat`. (0: source; 1: edge; 2: destination)
 */
void SDDMM(
    const std::string& op, HeteroGraphPtr graph, NDArray ufeat, NDArray vfeat,
    NDArray out, int lhs_target, int rhs_target);

void SDDMM(
    const char* op, HeteroGraphPtr graph, NDArray ufeat, NDArray vfeat,
    NDArray out, int lhs_target, int rhs_target);

/**
 * @brief Sparse-sparse matrix multiplication.
 *
 * The sparse matrices must have scalar weights (i.e. \a A_weights and \a
 * B_weights are 1D vectors.)
 */
std::pair<CSRMatrix, NDArray> CSRMM(
    CSRMatrix A, NDArray A_weights, CSRMatrix B, NDArray B_weights);

/**
 * @brief Summing up a list of sparse matrices.
 *
 * The sparse matrices must have scalar weights (i.e. the arrays in \a A_weights
 * are 1D vectors.)
 */
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A, const std::vector<NDArray>& A_weights);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_KERNEL_H_
