/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cuda/spmm.cu
 * @brief SPMM C APIs and definitions.
 */
#include <dgl/array.h>

#include <cstdlib>

#include "../../runtime/cuda/cuda_common.h"
#include "./functor.cuh"
#include "./ge_spmm.cuh"
#include "./spmm.cuh"

namespace dgl {

using namespace cuda;

namespace aten {

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @note use cusparse if the reduce operator is `sum` and there is
 *       no broadcast, use dgl's kernel in other cases.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsrHetero(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& vec_ufeat,
    const std::vector<NDArray>& vec_efeat, std::vector<NDArray>* vec_out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,  // ufeat node type id
    const std::vector<dgl_type_t>& out_ntids) {  // output node type id
  bool is_scalar_efeat =
      vec_efeat[0].NumElements() == vec_csr[0].indices->shape[0];
  bool use_efeat = op != "copy_lhs";
  auto device = runtime::DeviceAPI::Get(vec_csr[0].indptr->ctx);
  std::vector<DType*> trans_out((*vec_out).size(), NULL);
  bool use_deterministic_alg_only = false;
  if (NULL != std::getenv("USE_DETERMINISTIC_ALG"))
    use_deterministic_alg_only = true;

  bool use_legacy_cusparsemm =
      (CUDART_VERSION < 11000) && (reduce == "sum") &&
      // legacy cuSPARSE does not care about NNZ, hence the argument "false".
      ((op == "copy_lhs" && cusparse_available<DType, IdType>(false)) ||
       (op == "mul" && is_scalar_efeat &&
        cusparse_available<DType, IdType>(false)));
  // Create temporary output buffer to store non-transposed output
  if (use_legacy_cusparsemm) {
    for (dgl_type_t ntype = 0; ntype < (*vec_out).size(); ++ntype) {
      const int m = (*vec_out)[ntype]->shape[0];
      const int n = (*vec_out)[ntype]->shape[1];
      if (m == 0) continue;
      DType* out = static_cast<DType*>(device->AllocWorkspace(
          vec_csr[0].indptr->ctx, m * n * sizeof(DType)));
      CUDA_CALL(cudaMemset(out, 0, m * n * sizeof(DType)));
      trans_out[ntype] = out;
    }
  }
  // Check shape of ufeat for all relation type and compute feature size
  int64_t x_length = 1;
  for (dgl_type_t etype = 0; etype < (ufeat_ntids.size() - 1); ++etype) {
    NDArray ufeat = vec_ufeat[ufeat_ntids[etype]];
    NDArray next_ufeat = vec_ufeat[ufeat_ntids[etype + 1]];
    CHECK_EQ(ufeat->ndim, next_ufeat->ndim)
        << "Input features have different shapes";
    for (int i = 1; i < ufeat->ndim; ++i) {
      if (ufeat->shape[i] != next_ufeat->shape[i]) {
        if (ufeat->shape[i] == 1 || next_ufeat->shape[i] == 1)
          LOG(FATAL) << "Homogenized message passing on heterogeneous graphs "
                        "does not support "
                     << "automatic broadcasting.  Please manually broadcast it "
                        "before calling "
                     << "message passing functions.";
        else
          LOG(FATAL) << "Input features have different shapes.";
        return;
      }

      if (etype == 0) x_length *= ufeat->shape[i];
    }
  }
  // TODO(Israt): Can python do the following initializations while creating the
  // tensors?
  if (reduce == "max" || reduce == "min") {
    const int64_t dim = bcast.out_len;
    std::vector<bool> updated((*vec_out).size(), false);
    for (dgl_type_t etype = 0; etype < ufeat_ntids.size(); ++etype) {
      DType* out_off = (*vec_out)[out_ntids[etype]].Ptr<DType>();
      if (reduce == "max")
        _Fill(
            out_off, vec_csr[etype].num_rows * dim,
            cuda::reduce::Max<IdType, DType>::zero());
      else  // min
        _Fill(
            out_off, vec_csr[etype].num_rows * dim,
            cuda::reduce::Min<IdType, DType>::zero());
      const dgl_type_t dst_id = out_ntids[etype];
      if (!updated[dst_id]) {
        updated[dst_id] = true;
        if (op == "copy_lhs") {
          IdType* argu_ntype = (*out_aux)[2][dst_id].Ptr<IdType>();
          _Fill(
              argu_ntype, vec_csr[etype].num_rows * dim,
              static_cast<IdType>(-1));
        }
        if (op == "copy_rhs") {
          IdType* arge_etype = (*out_aux)[3][dst_id].Ptr<IdType>();
          _Fill(
              arge_etype, vec_csr[etype].num_rows * dim,
              static_cast<IdType>(-1));
        }
      }
    }
  }

  cudaStream_t stream = runtime::getCurrentCUDAStream();
  for (dgl_type_t etype = 0; etype < ufeat_ntids.size(); ++etype) {
    const dgl_type_t src_id = ufeat_ntids[etype];
    const dgl_type_t dst_id = out_ntids[etype];
    CSRMatrix csr = vec_csr[etype];
    if (reduce == "sum") {
      bool more_nnz = (csr.indices->shape[0] > csr.num_rows * csr.num_cols);
      /* Call  SpMM for each relation type */
      if (op == "copy_lhs" &&
          cusparse_available<DType, IdType>(more_nnz)) {  // cusparse
        /* If CUDA is less than 11.0, put the output in trans_out for later
         * transposition */
        DType* out = (CUDART_VERSION < 11000)
                         ? trans_out[dst_id]
                         : static_cast<DType*>((*vec_out)[dst_id]->data);
        CusparseCsrmm2Hetero<DType, IdType>(
            csr.indptr->ctx, csr, static_cast<DType*>(vec_ufeat[src_id]->data),
            nullptr, out, x_length, stream, use_deterministic_alg_only);
      } else if (
          op == "mul" && is_scalar_efeat &&
          cusparse_available<DType, IdType>(more_nnz)) {  // cusparse
        NDArray efeat = vec_efeat[etype];
        if (!IsNullArray(csr.data)) efeat = IndexSelect(efeat, csr.data);
        CusparseCsrmm2Hetero<DType, IdType>(
            csr.indptr->ctx, csr, static_cast<DType*>(vec_ufeat[src_id]->data),
            static_cast<DType*>(efeat->data),
            // TODO(Israt): Change (*vec_out) to trans_out to support CUDA
            // version < 11
            static_cast<DType*>((*vec_out)[dst_id]->data), x_length, stream,
            use_deterministic_alg_only);
      } else {  // general kernel
        NDArray ufeat =
            (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
        NDArray efeat =
            (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
        SWITCH_OP(op, Op, {
          cuda::SpMMCsr<IdType, DType, Op, cuda::reduce::Sum<IdType, DType>>(
              bcast, csr, ufeat, efeat, (*vec_out)[dst_id], NullArray(),
              NullArray());
        });
      }
    } else if (reduce == "max") {
      SWITCH_OP(op, Op, {
        NDArray ufeat =
            (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
        NDArray efeat =
            (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
        cuda::SpMMCmpCsrHetero<
            IdType, DType, Op, cuda::reduce::Max<IdType, DType>>(
            bcast, csr, ufeat, efeat, (*vec_out)[dst_id], (*out_aux)[0][dst_id],
            (*out_aux)[1][dst_id], (*out_aux)[2][dst_id], (*out_aux)[3][dst_id],
            src_id, etype);
      });
    } else if (reduce == "min") {
      SWITCH_OP(op, Op, {
        NDArray ufeat =
            (vec_ufeat.size() == 0) ? NullArray() : vec_ufeat[src_id];
        NDArray efeat =
            (vec_efeat.size() == 0) ? NullArray() : vec_efeat[etype];
        cuda::SpMMCmpCsrHetero<
            IdType, DType, Op, cuda::reduce::Min<IdType, DType>>(
            bcast, csr, ufeat, efeat, (*vec_out)[dst_id], (*out_aux)[0][dst_id],
            (*out_aux)[1][dst_id], (*out_aux)[2][dst_id], (*out_aux)[3][dst_id],
            src_id, etype);
      });
    } else {
      LOG(FATAL) << "Not implemented";
    }
  }

  if (use_legacy_cusparsemm) {
    // transpose output
    for (dgl_type_t ntype = 0; ntype < (*vec_out).size(); ++ntype) {
      const int m = (*vec_out)[ntype]->shape[0];
      const int n = (*vec_out)[ntype]->shape[1];
      if (m == 0) continue;
      DType* C_data = static_cast<DType*>((*vec_out)[ntype]->data);
      _Transpose(trans_out[ntype], C_data, n, m);
      device->FreeWorkspace(vec_csr[0].indptr->ctx, trans_out[ntype]);
    }
  }
}

template void SpMMCsrHetero<kDGLCUDA, int32_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
template void SpMMCsrHetero<kDGLCUDA, int64_t, __half>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
#if BF16_ENABLED
template void SpMMCsrHetero<kDGLCUDA, int32_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
template void SpMMCsrHetero<kDGLCUDA, int64_t, __nv_bfloat16>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
#endif  // BF16_ENABLED
template void SpMMCsrHetero<kDGLCUDA, int32_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
template void SpMMCsrHetero<kDGLCUDA, int64_t, float>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
template void SpMMCsrHetero<kDGLCUDA, int32_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);
template void SpMMCsrHetero<kDGLCUDA, int64_t, double>(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::vector<CSRMatrix>& csr, const std::vector<NDArray>& ufeat,
    const std::vector<NDArray>& efeat, std::vector<NDArray>* out,
    std::vector<std::vector<NDArray>>* out_aux,
    const std::vector<dgl_type_t>& ufeat_ntids,
    const std::vector<dgl_type_t>& out_ntids);

}  // namespace aten
}  // namespace dgl
