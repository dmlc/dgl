#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "./binary_reduce_impl.cuh"
#include "./backward_binary_reduce_impl.cuh"

using minigun::Csr;
using minigun::advance::RuntimeConfig;

namespace dgl {
namespace kernel {
namespace cuda {

// specialization for cusparse

void CusparseCsrmm2(const RuntimeConfig& rtcfg, const Csr& csr, GData<float>* gdata) {
  const int m = csr.row_offsets.length - 1;
  const int k = csr.row_offsets.length - 1;
  const int n = gdata->x_length;
  const int nnz = csr.column_indices.length;
  float alpha = 1.0;
  float beta = 0.0;
  // device
  auto device = runtime::DeviceAPI::Get(rtcfg.ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // allocate cusparse handle if needed
  if (!thr_entry->cusparse_handle) {
    CUSPARSE_CALL(cusparseCreate(&(thr_entry->cusparse_handle)));
  }
  cusparseMatDescr_t descr;
  CUSPARSE_CALL(cusparseCreateMatDescr(&descr));
  CUSPARSE_CALL(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CALL(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  //cusparseScsrmm(
      //thr_entry->cusparse_handle,
      //CUSPARSE_OPERATION_NON_TRANSPOSE,
      //CUSPARSE_OPERATION_TRANSPOSE,
      //m, n, k, nnz, alpha,
  LOG(FATAL) << "NOT IMPLEMENTED";
}

void CusparseCsrmm2(const RuntimeConfig& rtcfg, const Csr& csr, GData<double>* gdata) {
  LOG(FATAL) << "NOT IMPLEMENTED";
}

template <>
void CallBinaryReduce<float, DirectId<kDLGPU, int64_t>,
                      SelectSrc, SelectDst,
                      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    GData<float>* gdata) {
  CusparseCsrmm2(rtcfg, csr, gdata);
}

template <>
void CallBinaryReduce<float, DirectId<kDLGPU, int64_t>,
                      SelectSrc, SelectEdge,
                      BinaryUseLhs<float>, ReduceSum<kDLGPU, float>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    GData<float>* gdata) {
  CusparseCsrmm2(rtcfg, csr, gdata);
}

template <>
void CallBinaryReduce<double, DirectId<kDLGPU, int64_t>,
                      SelectSrc, SelectDst,
                      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    GData<double>* gdata) {
  CusparseCsrmm2(rtcfg, csr, gdata);
}

template <>
void CallBinaryReduce<double, DirectId<kDLGPU, int64_t>,
                      SelectSrc, SelectEdge,
                      BinaryUseLhs<double>, ReduceSum<kDLGPU, double>>(
    const RuntimeConfig& rtcfg,
    const Csr& csr,
    GData<double>* gdata) {
  CusparseCsrmm2(rtcfg, csr, gdata);
}

// generate definitions

#define REDUCER ReduceSum
#define XPU kDLGPU
#define GETID DirectId

EVAL(GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_DEFINE)
EVAL(GEN_BACKWARD_MODE, GEN_DTYPE, GEN_TARGET, GEN_BINARY_OP, GEN_BACKWARD_DEFINE)

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl
