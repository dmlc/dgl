/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/binary_reduce_sum.cu
 * \brief CUDA kernels for binary reduce sum
 */
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"
#include "../utils.h"

using minigun::advance::RuntimeConfig;
using Csr = minigun::Csr<int32_t>;

namespace dgl {
namespace kernel {
namespace cuda {
// specialization for cusparse

template <typename DType>
cusparseStatus_t Xcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const DType* alpha, const cusparseMatDescr_t descrA,
    const DType* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const DType* B, int ldb, const DType* beta, DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <>
cusparseStatus_t Xcsrmm2<float>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const float* alpha, const cusparseMatDescr_t descrA,
    const float* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
  return cusparseScsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <>
cusparseStatus_t Xcsrmm2<double>(cusparseHandle_t handle, cusparseOperation_t transA,
    cusparseOperation_t transB, int m, int n, int k, int nnz,
    const double* alpha, const cusparseMatDescr_t descrA,
    const double* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
  return cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz,
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t Xgeam(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const DType* alpha, const DType* A, int lda,
    const DType* beta, const DType* B, int ldb,
    DType* C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t Xgeam<float>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb,
    float* C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <>
cublasStatus_t Xgeam<double>(cublasHandle_t handle, cublasOperation_t transa,
    cublasOperation_t transb, int m, int n,
    const double* alpha, const double* A, int lda,
    const double* beta, const double* B, int ldb,
    double* C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda,
      beta, B, ldb, C, ldc);
}

template <typename DType>
void CusparseCsrmm2(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    const DType* B_data, DType* C_data,
    int out_size, int x_length) {
  // We use csrmm2 to perform following operation:
  // C = A x B, where A is a sparse matrix in csr format, B is the dense matrix for node
  // feature tensor. However, since cusparse only supports column-major, while our tensor
  // is stored in row-major, the actual computation is:
  // C = trans(A x trans(B)).
  // Currently, we use cublasXgeam to implement transposition and allocate intermediate
  // workspace memory for this.
  // TODO(minjie): The given CSR could potentially represent a bipartite graph (e.g. in the
  //   case of nodeflow). Currently, we don't have bipartite graph support. Here is a small
  //   hack. In the python side, we create a CSR that includes both the source and destination
  //   nodes in the bipartite graph (so it is still square matrix). Here, when multiplying
  //   this sparse matrix, we specify the number of rows (the `m` here) to be equal to the
  //   number of rows of the output tensor (i.e, the `out_size`).
  //   In the future, we should make sure the number of rows of the given csr is equal
  //   to out_size (a.k.a the given csr is a rectangle matrix).
  const int m = out_size;
  const int k = csr.row_offsets.length - 1;
  const int n = x_length;
  const int nnz = csr.column_indices.length;
  const DType alpha = 1.0;
  const DType beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(rtcfg.ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  CUSPARSE_CALL(cusparseSetStream(thr_entry->cusparse_handle, rtcfg.stream));
  // allocate matrix for temporary transposed output
  DType* trans_out = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, m * n * sizeof(DType)));
  // all one data array
  DType* valptr = static_cast<DType*>(device->AllocWorkspace(rtcfg.ctx, nnz * sizeof(DType)));
  utils::Fill<kDLGPU>(rtcfg.ctx, valptr, nnz, static_cast<DType>(1.));
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(Xcsrmm2<DType>(
      thr_entry->cusparse_handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_TRANSPOSE,
      m, n, k, nnz, &alpha,
      descr, valptr, csr.row_offsets.data, csr.column_indices.data,
      B_data, n, &beta, trans_out, m));
  device->FreeWorkspace(rtcfg.ctx, valptr);
  // transpose the output matrix
  if (!thr_entry->cublas_handle) {
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  }
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, rtcfg.stream));
  CUBLAS_CALL(Xgeam<DType>(
      thr_entry->cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n, m,
      &alpha, trans_out, m,
      &beta, nullptr, n,
      C_data, n));
  device->FreeWorkspace(rtcfg.ctx, trans_out);
}

// forward

template <typename DType>
void FallbackCallBinaryReduce(
    const RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    GData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  typedef cuda::FunctorsTempl<Idx, DType, LeftSelector,
                        RightSelector, BinaryOp, Reducer>
          Functors;
  typedef cuda::BinaryReduce<Idx, DType, Functors> UDF;
  // csr
  auto outcsr = graph->GetOutCSR();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr->indptr(), outcsr->indices());
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(outcsr->edge_ids()->data);
  }
  if (RightSelector::target == binary_op::kEdge && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(outcsr->edge_ids()->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(outcsr->edge_ids()->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, GData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

template <typename DType>
void FallbackCallBackwardBinaryReduce(
    const RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardGData<int32_t, DType>* gdata) {
  constexpr int XPU = kDLGPU;
  constexpr int Mode = binary_op::kGradLhs;
  typedef int32_t Idx;
  typedef SelectSrc LeftSelector;
  typedef SelectNone RightSelector;
  typedef BinaryUseLhs<DType> BinaryOp;
  typedef ReduceSum<kDLGPU, DType> Reducer;
  // For backward computation, we use reverse csr and switch dst and src.
  // This benefits the most common src_op_edge or copy_src case, because the
  // gradients of src are now aggregated into destination buffer to reduce
  // competition of atomic add.
  auto incsr = graph->GetInCSR();
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr->indptr(), incsr->indices());
  typedef cuda::BackwardFunctorsTempl<Idx, DType,
          typename SwitchSrcDst<LeftSelector>::Type,
          typename SwitchSrcDst<RightSelector>::Type,
          BinaryOp, Reducer> Functors;
  typedef cuda::BackwardBinaryReduce<Mode, Idx, DType, Functors> UDF;
  // If the user-given mapping is none and the target is edge data, we need to
  // replace the mapping by the edge ids in the csr graph so that the edge
  // data is correctly read/written.
  if (LeftSelector::target == binary_op::kEdge
      && gdata->lhs_mapping == nullptr) {
    gdata->lhs_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  if (RightSelector::target == binary_op::kEdge
      && gdata->rhs_mapping == nullptr) {
    gdata->rhs_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge
      && gdata->out_mapping == nullptr) {
    gdata->out_mapping = static_cast<Idx*>(incsr->edge_ids()->data);
  }
  // TODO(minjie): allocator
  minigun::advance::Advance<XPU, Idx, cuda::AdvanceConfig, BackwardGData<Idx, DType>, UDF>(
        rtcfg, csr, gdata, minigun::IntArray1D<Idx>());
}

}  // namespace cuda

template <>
void CallBinaryReduce<kDLGPU, int32_t, float, SelectSrc, SelectNone,
                      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    GData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<float>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto incsr = graph->GetInCSR();
    Csr csr = utils::CreateCsr<int32_t>(incsr->indptr(), incsr->indices());
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        gdata->out_size, gdata->x_length);
  }
}

template <>
void CallBinaryReduce<kDLGPU, int32_t, double, SelectSrc, SelectNone,
                      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    GData<int32_t, double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBinaryReduce<double>(rtcfg, graph, gdata);
  } else {
    // cusparse use rev csr for csrmm
    auto incsr = graph->GetInCSR();
    Csr csr = utils::CreateCsr<int32_t>(incsr->indptr(), incsr->indices());
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->lhs_data, gdata->out_data,
        gdata->out_size, gdata->x_length);
  }
}

// backward

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, int32_t, float,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardGData<int32_t, float>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<float>(rtcfg, graph, gdata);
  } else {
    auto outcsr = graph->GetOutCSR();
    Csr csr = utils::CreateCsr<int32_t>(outcsr->indptr(), outcsr->indices());
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
        gdata->out_size, gdata->x_length);
  }
}

template <>
void CallBackwardBinaryReduce<kDLGPU, binary_op::kGradLhs, int32_t, double,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const ImmutableGraph* graph,
    BackwardGData<int32_t, double>* gdata) {
  if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
    cuda::FallbackCallBackwardBinaryReduce<double>(rtcfg, graph, gdata);
  } else {
    auto outcsr = graph->GetOutCSR();
    Csr csr = utils::CreateCsr<int32_t>(outcsr->indptr(), outcsr->indices());
    cuda::CusparseCsrmm2(rtcfg, csr, gdata->grad_out_data, gdata->grad_lhs_data,
        gdata->out_size, gdata->x_length);
  }
}

// generate definitions

#define REDUCER ReduceSum
#define XPU kDLGPU
#define IDX int32_t

EVAL(GEN_DTYPE, GEN_OP_TARGET, GEN_DEFINE);
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_OP_TARGET, GEN_BACKWARD_DEFINE);

}  // namespace kernel
}  // namespace dgl
