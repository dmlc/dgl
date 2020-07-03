/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cu
 * \brief SDDMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./sddmm.cuh"
#include "./functor.cuh"

namespace dgl {
namespace aten {

#define SWITCH_OP(op, Op, ...)                                      \
  do {                                                              \
    if ((op) == "add") {                                            \
      typedef cuda::binary::Add<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "sub") {                                     \
      typedef cuda::binary::Sub<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "mul") {                                     \
      typedef cuda::binary::Mul<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "div") {                                     \
      typedef cuda::binary::Div<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_u") {                                  \
      typedef cuda::binary::CopyU<DType> Op;                        \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "copy_e") {                                  \
      typedef cuda::binary::CopyE<DType> Op;                        \
      { __VA_ARGS__ }                                               \
    } else if ((op) == "dot") {                                     \
      typedef cuda::binary::Dot<DType> Op;                          \
      { __VA_ARGS__ }                                               \
    } else {                                                        \
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op;     \
    }                                                               \
  } while (0)


#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...)\
  do {                                                                  \
    if ((lhs_target) == 0) {                                            \
      constexpr int LhsTarget = 0;                                      \
      if ((rhs_target) == 0) {                                          \
        constexpr int RhsTarget = 0;                                    \
        { __VA_ARGS__ }                                                 \
      } else {                                                          \
        constexpr int RhsTarget = 1;                                    \
        { __VA_ARGS__ }                                                 \
      }                                                                 \
    } else {                                                            \
      constexpr int LhsTarget = 1;                                      \
      if ((rhs_target) == 0) {                                          \
        constexpr int RhsTarget = 0;                                    \
        { __VA_ARGS__ }                                                 \
      } else {                                                          \
        constexpr int RhsTarget = 1;                                    \
        { __VA_ARGS__ }                                                 \
      }                                                                 \
    }                                                                   \
  } while (0)


/*!
 * \brief CUDA implementation of g-SDDMM on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const CSRMatrix& csr,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              int lhs_target,
              int rhs_target) {
  SWITCH_OP(op, Op,
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      cuda::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, csr, ufeat, vfeat, out);
  }));
}

/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const COOMatrix& coo,
              NDArray ufeat,
              NDArray vfeat,
              NDArray out,
              int lhs_target,
              int rhs_target) {
  SWITCH_OP(op, Op,
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        cuda::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, coo, ufeat, vfeat, out);
  }));
}

template void SDDMMCsr<kDLGPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLGPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);

template void SDDMMCoo<kDLGPU, int32_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int64_t, float>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int32_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLGPU, int64_t, double>(
    const std::string& op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray vfeat, NDArray out,
    int lhs_target, int rhs_target);

}  // namespace aten
}  // namespace dgl
