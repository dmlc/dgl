/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel_decl.h
 * \brief Sparse matrix format-specific operator declarations.
 */
#ifndef DGL_ARRAY_KERNEL_DECL_H_
#define DGL_ARRAY_KERNEL_DECL_H_

#include <dgl/bcast.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>

#include <string>
#include <vector>
#include <utility>

namespace dgl {
namespace aten {

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target);

/*!
 * \brief Segment reduce.
 */
template <int XPU, typename IdType, int bits>
void SegmentReduce(const std::string& op,
                   NDArray feat,
                   NDArray offsets,
                   NDArray out,
                   NDArray arg);

/*!
 * \brief Scatter Add on first dimension.
 */
template <int XPU, typename IdType, int bits>
void ScatterAdd(NDArray feat,
                NDArray idx,
                NDArray out);

/*!
 * \brief Backward function of segment cmp.
 */
template <int XPU, typename IdType, int bits>
void BackwardSegmentCmp(NDArray feat,
                        NDArray arg,
                        NDArray out);

/*!
 * \brief Sparse-sparse matrix multiplication
 *
 * \note B is transposed (i.e. in CSC format).
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights);

/*!
 * \brief Sparse-sparse matrix summation.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A,
    const std::vector<NDArray>& A_weights);

/*!
 * \brief Return a sparse matrix with the values of A but nonzero entry locations of B.
 */
template <int XPU, typename IdType, typename DType>
NDArray CSRMask(const CSRMatrix& A, NDArray A_weights, const CSRMatrix& B);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_KERNEL_DECL_H_
