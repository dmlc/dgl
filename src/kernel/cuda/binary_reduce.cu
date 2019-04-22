#include <dlpack/dlpack.h>
#include <minigun/minigun.h>
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../binary_reduce.h"
#include "./binary_reduce.cuh"

using dgl::runtime::NDArray;
using minigun::Csr;
using minigun::IntArray1D;

namespace dgl {
namespace kernel {
namespace {
int64_t ComputeXLength(NDArray feat_array) {
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
int FindNumThreads(int dim, int max_nthrs) {
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}
}  // namespace

template <typename DType, typename EidGetter,
          typename OutSelector, typename LeftSelector, typename RightSelector,
          typename BinaryOp, typename Reducer>
struct BinaryReduceExecutor<kDLGPU, DType, EidGetter,
                            OutSelector, LeftSelector, RightSelector,
                            BinaryOp, Reducer> {
  static void Run(NDArray indptr,
                  NDArray indices,
                  NDArray edge_ids,
                  NDArray src_data,
                  NDArray edge_data,
                  NDArray dst_data,
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
    // GData
    cuda::GData<DType> gdata;
    gdata.x_length = x_len;
    gdata.src_data = static_cast<DType*>(src_data->data);
    gdata.edge_data = static_cast<DType*>(edge_data->data);
    gdata.dst_data = static_cast<DType*>(dst_data->data);
    gdata.edge_ids = static_cast<int64_t*>(edge_ids->data);
    gdata.out_data = static_cast<DType*>(out_data->data);
    // device GData
    cuda::GData<DType>* d_gdata;
    CUDA_CALL(cudaMalloc(&d_gdata, sizeof(cuda::GData<DType>)));
    CUDA_CALL(cudaMemcpy(d_gdata, &gdata, sizeof(cuda::GData<DType>),
          cudaMemcpyHostToDevice));
    // call advance
    minigun::advance::RuntimeConfig rtcfg;
    rtcfg.ctx = out_data->ctx;
    rtcfg.stream = thr_entry->stream;
    const int nt = FindNumThreads(x_len, 64);
    rtcfg.data_num_threads = nt;
    // XXX(minjie): hard-code to let each thread compute two elements to increase
    //              instruction level parallelism
    rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);

    typedef minigun::advance::Config<true, minigun::advance::kV2N> Config;
    minigun::advance::Advance<kDLGPU, Config, cuda::GData<DType>, Functor, Alloc>(
        rtcfg, csr, d_gdata, IntArray1D(), IntArray1D());

    // free device GData
    CUDA_CALL(cudaFree(d_gdata));
  }
};

}  // namespace kernel
}  // namespace dgl
