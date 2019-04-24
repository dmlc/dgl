#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <dlpack/dlpack.h>
#include <minigun/minigun.h>
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../binary_reduce_common.h"
#include "./binary_reduce_impl.h"

using dgl::runtime::NDArray;
using minigun::Csr;
using minigun::IntArray1D;

namespace dgl {
namespace kernel {
namespace cuda {
namespace {
inline int64_t ComputeXLength(NDArray feat_array) {
  int64_t ret = 1;
  for (int i = 1; i < feat_array->ndim; ++i) {
    ret *= feat_array->shape[i];
  }
  return ret;
}

// Find the number of threads that is:
//  - power of two
//  - smaller or equal to dim
//  - smaller or equal to max_nthrs
inline int FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

inline int64_t NElements(NDArray array) {
  if (array->ndim == 0) {
    return 0;
  } else {
    int64_t ret = 1;
    for (int i = 0; i < array->ndim; ++i) {
      ret *= array->shape[i];
    }
    return ret;
  }
}
}  // namespace

template <typename DType, typename Reducer>
GData<DType>* AllocGData(cudaStream_t stream, int64_t x_len,
                         NDArray lhs_mapping, NDArray rhs_mapping,
                         NDArray lhs_data, NDArray rhs_data,
                         NDArray out_mapping, NDArray out_data) {
  // GData
  GData<DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (lhs_mapping->ndim != 0) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (rhs_mapping->ndim != 0) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (out_mapping->ndim != 0) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // fill out data with zero values
  thrust::device_ptr<DType> out_ptr = thrust::device_pointer_cast(gdata.out_data);
  thrust::fill(thrust::cuda::par.on(stream),
      out_ptr, out_ptr + NElements(out_data), Zero<Reducer>::value);
  // device GData
  GData<DType>* d_gdata;
  CUDA_CALL(cudaMalloc(&d_gdata, sizeof(GData<DType>)));
  CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(GData<DType>),
        cudaMemcpyHostToDevice));
  return d_gdata;
}

void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    NDArray indptr,
    NDArray indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    NDArray lhs_mapping,
    NDArray rhs_mapping,
    NDArray lhs_data,
    NDArray rhs_data,
    NDArray out_mapping,
    NDArray out_data) {
  // device
  auto device = runtime::DeviceAPI::Get(out_data->ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // Graph
  Csr csr;
  csr.row_offsets.data = static_cast<int64_t*>(indptr->data);
  csr.row_offsets.length = indptr->shape[0];
  csr.column_indices.data = static_cast<int64_t*>(indices->data);
  csr.column_indices.length = indices->shape[0];
  const int64_t x_len = ComputeXLength(out_data);
  LOG(INFO) << "csr.rowlen=" << csr.row_offsets.length
    << " csr.collen=" << csr.column_indices.length
    << " xlen=" << x_len;

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
  rtcfg.stream = thr_entry->stream;
  const int nt = FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);

  const DLDataType& dtype = lhs_data->dtype;
  const bool has_indirect =
    (lhs_mapping->ndim != 0 || rhs_mapping->ndim != 0 || out_mapping->ndim != 0);
  LOG(INFO) << "has_indirect? " << has_indirect;
  LOG(INFO) << "nt=" << nt << " nb=" << rtcfg.data_num_blocks;
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, kDLGPU, DType, Reducer, {
      GData<DType>* gdata = AllocGData<DType, Reducer>(
          rtcfg.stream, x_len, lhs_mapping, rhs_mapping,
          lhs_data, rhs_data, out_mapping, out_data);
      BINARY_OP_SWITCH(op, DType, BinaryOp, {
        TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
          if (has_indirect) {
            typedef IndirectId<kDLGPU, int64_t> IdGetter;
            CallBinaryReduce<DType, IdGetter, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, gdata);
          } else {
            typedef DirectId<kDLGPU, int64_t> IdGetter;
            CallBinaryReduce<DType, IdGetter, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, gdata);
          }
        });
      });
      // free device GData
      CUDA_CALL(cudaFree(gdata));
    });
    if (reducer == binary_op::kReduceMean) {
      // TODO(minjie): divide
      LOG(FATAL) << "reduce mean is not supported.";
    }
  });
}

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl
