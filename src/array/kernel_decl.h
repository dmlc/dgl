/**
 *  Copyright (c) 2020 by Contributors
 * @file array/kernel_decl.h
 * @brief Sparse matrix format-specific operator declarations.
 */
#ifndef DGL_ARRAY_KERNEL_DECL_H_
#define DGL_ARRAY_KERNEL_DECL_H_

#include <dgl/base_heterograph.h>
#include <dgl/bcast.h>
#include <dgl/runtime/ndarray.h>

#include <string>
#include <utility>
#include <vector>

namespace dgl {
namespace aten {

/**
 * @brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const aten::CSRMatrix& csr, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

/**
 * @brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format
 * with heterograph support.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsrHetero(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_eid,
    const std::vector<dgl_type_t>& out_eid);
/**
 * @brief Generalized Sparse Matrix Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCoo(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const aten::COOMatrix& coo, NDArray ufeat, NDArray efeat, NDArray out,
    std::vector<NDArray> out_aux);

/**
 * @brief Generalized Sampled Dense-Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCsr(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);
/**
 * @brief Generalized Sampled Dense-Dense Matrix Multiplication on Csr format
 * with heterograph support.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCsrHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& ufeat_eid,
    const std::vector<dgl_type_t>& out_eid);

/**
 * @brief Generalized Sampled Dense-Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCoo(
    const std::string& op, const BcastOff& bcast, const aten::COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out, int lhs_target, int rhs_target);

/**
 * @brief Generalized Sampled Dense-Dense Matrix Multiplication on Coo format
 * with heterograph support.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCooHetero(
    const std::string& op, const BcastOff& bcast,
    const std::vector<COOMatrix>& vec_coo, const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs, std::vector<NDArray> vec_out,
    int lhs_target, int rhs_target, const std::vector<dgl_type_t>& lhs_eid,
    const std::vector<dgl_type_t>& rhs_eid);

/**
 * @brief Generalized Dense Matrix-Matrix Multiplication according to relation
 * types.
 */
template <int XPU, typename IdType, typename DType>
void GatherMM(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b);

/**
 * @brief Generalized Dense Matrix-Matrix Multiplication according to relation
 * types.
 */
template <int XPU, typename IdType, typename DType>
void GatherMMScatter(
    const NDArray A, const NDArray B, NDArray out, const NDArray idx_a,
    const NDArray idx_b, const NDArray idx_c);

/**
 * @brief Generalized segmented dense Matrix-Matrix Multiplication.
 */
template <int XPU, typename IdType, typename DType>
void SegmentMM(
    const NDArray A, const NDArray B, NDArray out, const NDArray seglen_A,
    bool a_trans, bool b_trans);

template <int XPU, typename IdType, typename DType>
void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen);

/**
 * @brief Segment reduce.
 */
template <int XPU, typename IdType, typename DType>
void SegmentReduce(
    const std::string& op, NDArray feat, NDArray offsets, NDArray out,
    NDArray arg);

/**
 * @brief Scatter Add on first dimension.
 */
template <int XPU, typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out);

/**
 * @brief Update gradients for reduce operator max and min on first dimension.
 */
template <int XPU, typename IdType, typename DType>
void UpdateGradMinMax_hetero(
    const HeteroGraphPtr& g, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out);

/**
 * @brief Backward function of segment cmp.
 */
template <int XPU, typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out);

/**
 * @brief Sparse-sparse matrix multiplication
 *
 * @param A The left operand.
 * @param A_weights The weights of matrix as a 1D tensor.
 * @param B The right operand.
 * @param B_weights The weights of matrix as a 1D tensor.
 *
 * @note GPU implementation will cast the indices to 32 bit.
 * @note The zero entries in the result are not removed.
 * @note The CSR matrix should not have duplicate entries.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A, NDArray A_weights, const CSRMatrix& B,
    NDArray B_weights);

/**
 * @brief Sparse-sparse matrix summation.
 *
 * @param A The sparse matrices with the same size.
 * @param A_weights The weights of each sparse matrix as a 1D tensor.
 *
 * @note GPU implementation will cast the indices to 32 bit.
 * @note The zero entries in the result are not removed.
 * @note The CSR matrix should not have duplicate entries.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A, const std::vector<NDArray>& A_weights);

/**
 * @brief Edge_softmax_csr forward function on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void Edge_softmax_csr_forward(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
/**
 * @brief Edge_softmax_csr backward function on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void Edge_softmax_csr_backward(
    const std::string& op, const BcastOff& bcast, const aten::CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out);
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_KERNEL_DECL_H_
