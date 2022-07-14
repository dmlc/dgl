/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cu
 * \brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>
#include "./spmm.cuh"
#include "./ge_spmm.cuh"
#include "./functor.cuh"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using namespace cuda;

namespace aten {

/*!
 * \brief Determine whether cusparse SpMM function is applicable.
 */
template <int bits, typename IdType>
inline bool cusparse_available(bool more_nnz_than_matrix_size) {
#if CUDART_VERSION < 11000
  if (std::is_same<IdType, int>::value)
    if (bits > 16)
      return true;
  return false;
#else
  if (bits == 16)
    return false;  // cusparse's SpMM on fp16 is slow, temporally disabled.
  // If the CSR matrix has more NNZ than matrix size, we should not use cuSPARSE 11.1.
  return !more_nnz_than_matrix_size;
#endif
}

/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  bool is_scalar_efeat = efeat.NumElements() == csr.indices->shape[0];
  bool use_efeat = op != "copy_lhs";

  if (reduce == "sum") {
    bool more_nnz = (csr.indices->shape[0] > csr.num_rows * csr.num_cols);
    if (op == "copy_lhs" && cusparse_available<bits, IdType>(more_nnz)) {
      // cusparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
      SWITCH_BITS(bits, DType, {
        CusparseCsrmm2<DType, IdType>(
            ufeat->ctx, csr,
            static_cast<DType*>(ufeat->data),
            nullptr,
            static_cast<DType*>(out->data),
            x_length);
      });
    } else if (op == "mul" && is_scalar_efeat && cusparse_available<bits, IdType>(more_nnz)) {
      // cusparse
      int64_t x_length = 1;
      for (int i = 1; i < ufeat->ndim; ++i)
        x_length *= ufeat->shape[i];
      if (!IsNullArray(csr.data)) {
        SWITCH_BITS(bits, DType, {
          efeat = _IndexSelect<DType, IdType>(efeat, csr.data);
        });
      }
      SWITCH_BITS(bits, DType, {
        CusparseCsrmm2<DType, IdType>(
            ufeat->ctx, csr,
            static_cast<DType*>(ufeat->data),
            static_cast<DType*>(efeat->data),
            static_cast<DType*>(out->data),
            x_length);
      });
    } else {  // general kernel
      SWITCH_BITS(bits, DType, {
        SWITCH_OP(op, Op, {
          cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType> >(
              bcast, csr, ufeat, efeat, out, NullArray(), NullArray());
        });
      });
    }
  } else if (reduce == "max") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Max<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else if (reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Min<IdType, DType> >(
            bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}


/*!
 * \brief CUDA implementation of g-SpMM on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Sum<IdType, DType, true> > (
            bcast, coo, ufeat, efeat, out, NullArray(), NullArray());
      });
    });
  } else if (reduce == "max") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> > (
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  }  else if (reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cuda::SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> > (
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

template void SpMMCsr<kDLGPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLGPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


template void SpMMCoo<kDLGPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLGPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


}  // namespace aten
}  // namespace dgl
