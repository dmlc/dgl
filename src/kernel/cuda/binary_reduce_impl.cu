#include <dlpack/dlpack.h>
#include <minigun/minigun.h>
#include <dgl/runtime/device_api.h>

#include "../../runtime/cuda/cuda_common.h"
#include "../binary_reduce.h"
#include "./binary_reduce_impl.h"
#include "./utils.cuh"

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

inline int64_t NElements(NDArray array) {
  if (IsNoneArray(array)) {
    return 0;
  } else {
    int64_t ret = 1;
    for (int i = 0; i < array->ndim; ++i) {
      ret *= array->shape[i];
    }
    return ret;
  }
}
inline int64_t Prod(const std::vector<int64_t>& vec) {
  int64_t ret = 1;
  for (int64_t v : vec) {
    ret *= v;
  }
  return ret;
}
}  // namespace

template <typename DType, typename Reducer>
GData<DType> AllocGData(cudaStream_t stream, int64_t x_len,
                        NDArray lhs_mapping, NDArray rhs_mapping,
                        NDArray lhs_data, NDArray rhs_data,
                        NDArray out_mapping, NDArray out_data) {
  // GData
  GData<DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (!IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // fill out data with zero values
  utils::Fill(stream, gdata.out_data, NElements(out_data), Zero<Reducer>::value);
  return gdata;
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
  csr.row_offsets.data = static_cast<mg_int*>(indptr->data);
  csr.row_offsets.length = indptr->shape[0];
  csr.column_indices.data = static_cast<mg_int*>(indices->data);
  csr.column_indices.length = indices->shape[0];
  const int64_t x_len = ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);

  const DLDataType& dtype = out_data->dtype;
  const bool has_indirect =
    !(IsNoneArray(lhs_mapping) && IsNoneArray(rhs_mapping) && IsNoneArray(out_mapping));
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, kDLGPU, DType, Reducer, {
      GData<DType> gdata = AllocGData<DType, Reducer>(
          rtcfg.stream, x_len, lhs_mapping, rhs_mapping,
          lhs_data, rhs_data, out_mapping, out_data);
      BINARY_OP_SWITCH(op, DType, BinaryOp, {
        TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
          if (has_indirect) {
            typedef IndirectId<kDLGPU, int64_t> IdGetter;
            CallBinaryReduce<DType, IdGetter, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
          } else {
            typedef DirectId<kDLGPU, int64_t> IdGetter;
            CallBinaryReduce<DType, IdGetter, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
          }
        });
      });
    });
    if (reducer == binary_op::kReduceMean) {
      // TODO(minjie): divide
      LOG(FATAL) << "reduce mean is not supported.";
    }
  });
}

template <int NDim, typename DType, typename Reducer>
BcastGData<NDim, DType> AllocBcastGData(
    cudaStream_t stream,
    const BcastInfo& info,
    NDArray lhs_mapping, NDArray rhs_mapping,
    NDArray lhs_data, NDArray rhs_data,
    NDArray out_mapping, NDArray out_data) {
  // GData
  BcastGData<NDim, DType> gdata;
  // dim, shape and stride
  gdata.ndim = info.lhs_shape.size();
  std::copy(info.lhs_shape.begin(), info.lhs_shape.end(), gdata.lhs_shape);
  std::copy(info.lhs_stride.begin(), info.lhs_stride.end(), gdata.lhs_stride);
  std::copy(info.rhs_shape.begin(), info.rhs_shape.end(), gdata.rhs_shape);
  std::copy(info.rhs_stride.begin(), info.rhs_stride.end(), gdata.rhs_stride);
  std::copy(info.out_shape.begin(), info.out_shape.end(), gdata.out_shape);
  std::copy(info.out_stride.begin(), info.out_stride.end(), gdata.out_stride);
  gdata.lhs_len = Prod(info.lhs_shape);
  gdata.rhs_len = Prod(info.rhs_shape);
  gdata.out_len = Prod(info.out_shape);
  // data
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (!IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // fill out data with zero values
  utils::Fill(stream, gdata.out_data, NElements(out_data), Zero<Reducer>::value);
  return gdata;
}

void BinaryReduceBcastImpl(
    const BcastInfo& info,
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
  csr.row_offsets.data = static_cast<mg_int*>(indptr->data);
  csr.row_offsets.length = indptr->shape[0];
  csr.column_indices.data = static_cast<mg_int*>(indices->data);
  csr.column_indices.length = indices->shape[0];
  const int64_t x_len = ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);

  const DLDataType& dtype = out_data->dtype;
  const bool has_indirect =
    !(IsNoneArray(lhs_mapping) && IsNoneArray(rhs_mapping) && IsNoneArray(out_mapping));
  const int bcast_ndim = info.out_shape.size();
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, kDLGPU, DType, Reducer, {
      BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
        BcastGData<NDim, DType> gdata = AllocBcastGData<NDim, DType, Reducer>(
            rtcfg.stream, info, lhs_mapping, rhs_mapping,
            lhs_data, rhs_data, out_mapping, out_data);
        BINARY_OP_SWITCH(op, DType, BinaryOp, {
          TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
            if (has_indirect) {
              typedef IndirectId<kDLGPU, int64_t> IdGetter;
              CallBinaryReduceBcast<NDim, DType, IdGetter, LeftTarget,
                RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
            } else {
              typedef DirectId<kDLGPU, int64_t> IdGetter;
              CallBinaryReduceBcast<NDim, DType, IdGetter, LeftTarget,
                RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
            }
          });
        });
      });
    });
    if (reducer == binary_op::kReduceMean) {
      // TODO(minjie): divide
      LOG(FATAL) << "reduce mean is not supported.";
    }
  });
}

// backward implementations

template <typename DType>
BackwardGData<DType> AllocBackwardGData(
    cudaStream_t stream, int64_t x_len,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs_data, NDArray rhs_data, NDArray out_data, NDArray grad_out_data,
    NDArray grad_lhs_data, NDArray grad_rhs_data) {
  // GData
  BackwardGData<DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  gdata.grad_out_data = static_cast<DType*>(grad_out_data->data);
  if (!IsNoneArray(grad_lhs_data)) {
    gdata.grad_lhs_data = static_cast<DType*>(grad_lhs_data->data);
    // fill out data with zero values
    utils::Fill(stream, gdata.grad_lhs_data, NElements(grad_lhs_data),
                static_cast<DType>(0));
  }
  if (!IsNoneArray(grad_rhs_data)) {
    gdata.grad_rhs_data = static_cast<DType*>(grad_rhs_data->data);
    // fill out data with zero values
    utils::Fill(stream, gdata.grad_rhs_data, NElements(grad_rhs_data),
                static_cast<DType>(0));
  }
  if (!IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  return gdata;
}

void BackwardBinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    NDArray rev_indptr, NDArray rev_indices,
    binary_op::Target lhs, binary_op::Target rhs,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs_data, NDArray rhs_data, NDArray out_data,
    NDArray grad_out_data,
    NDArray grad_lhs_data, NDArray grad_rhs_data) {
  // device
  auto device = runtime::DeviceAPI::Get(out_data->ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // Graph
  Csr csr;
  csr.row_offsets.data = static_cast<mg_int*>(rev_indptr->data);
  csr.row_offsets.length = rev_indptr->shape[0];
  csr.column_indices.data = static_cast<mg_int*>(rev_indices->data);
  csr.column_indices.length = rev_indices->shape[0];
  const int64_t x_len = ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);

  const DLDataType& dtype = out_data->dtype;
  const bool has_indirect =
    !(IsNoneArray(lhs_mapping) && IsNoneArray(rhs_mapping) && IsNoneArray(out_mapping));
  const bool req_lhs = !IsNoneArray(grad_lhs_data);
  const bool req_rhs = !IsNoneArray(grad_rhs_data);
  BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, {
    DGL_DTYPE_SWITCH(dtype, DType, {
      BackwardGData<DType> gdata = AllocBackwardGData<DType>(
          rtcfg.stream, x_len,
          lhs_mapping, rhs_mapping, out_mapping,
          lhs_data, rhs_data, out_data, grad_out_data,
          grad_lhs_data, grad_rhs_data);
      //REDUCER_SWITCH(reducer, kDLGPU, DType, Reducer, {
        typedef ReduceSum<kDLGPU, DType> Reducer;
        BINARY_OP_SWITCH(op, DType, BinaryOp, {
          TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
            if (has_indirect) {
              typedef IndirectId<kDLGPU, int64_t> IdGetter;
              CallBackwardBinaryReduce<Mode, DType, IdGetter, LeftTarget,
                RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
            } else {
              typedef DirectId<kDLGPU, int64_t> IdGetter;
              CallBackwardBinaryReduce<Mode, DType, IdGetter, LeftTarget,
                RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
            }
          });
        });
      //});
      if (reducer == binary_op::kReduceMean) {
        // TODO(minjie): divide
        LOG(FATAL) << "reduce mean is not supported.";
      }
    });
  });
}

template <int NDim, typename DType>
BackwardBcastGData<NDim, DType> AllocBackwardBcastGData(
    cudaStream_t stream,
    const BcastInfo& info,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs, NDArray rhs, NDArray out, NDArray grad_out,
    NDArray grad_lhs, NDArray grad_rhs) {
  // GData
  BackwardBcastGData<NDim, DType> gdata;
  // dim, shape and stride
  gdata.ndim = info.lhs_shape.size();
  gdata.lhs_len = Prod(info.lhs_shape);
  gdata.rhs_len = Prod(info.rhs_shape);
  gdata.out_len = Prod(info.out_shape);
  std::copy(info.lhs_shape.begin(), info.lhs_shape.end(), gdata.lhs_shape);
  std::copy(info.lhs_stride.begin(), info.lhs_stride.end(), gdata.lhs_stride);
  std::copy(info.rhs_shape.begin(), info.rhs_shape.end(), gdata.rhs_shape);
  std::copy(info.rhs_stride.begin(), info.rhs_stride.end(), gdata.rhs_stride);
  std::copy(info.out_shape.begin(), info.out_shape.end(), gdata.out_shape);
  std::copy(info.out_stride.begin(), info.out_stride.end(), gdata.out_stride);
  // mappings
  if (!IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // data
  gdata.lhs_data = static_cast<DType*>(lhs->data);
  gdata.rhs_data = static_cast<DType*>(rhs->data);
  gdata.out_data = static_cast<DType*>(out->data);
  gdata.grad_out_data = static_cast<DType*>(grad_out->data);
  if (!IsNoneArray(grad_lhs)) {
    gdata.grad_lhs_data = static_cast<DType*>(grad_lhs->data);
    // fill out data with zero values
    utils::Fill(stream, gdata.grad_lhs_data, NElements(grad_lhs),
                static_cast<DType>(0));
  }
  if (!IsNoneArray(grad_rhs)) {
    gdata.grad_rhs_data = static_cast<DType*>(grad_rhs->data);
    // fill out data with zero values
    utils::Fill(stream, gdata.grad_rhs_data, NElements(grad_rhs),
                static_cast<DType>(0));
  }
  return gdata;
}

void BackwardBinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    NDArray rev_indptr, NDArray rev_indices,
    binary_op::Target lhs_tgt, binary_op::Target rhs_tgt,
    NDArray lhs_mapping, NDArray rhs_mapping, NDArray out_mapping,
    NDArray lhs, NDArray rhs, NDArray out, NDArray grad_out,
    NDArray grad_lhs, NDArray grad_rhs) {
  // device
  auto device = runtime::DeviceAPI::Get(out->ctx);
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // Graph
  Csr csr;
  csr.row_offsets.data = static_cast<mg_int*>(rev_indptr->data);
  csr.row_offsets.length = rev_indptr->shape[0];
  csr.column_indices.data = static_cast<mg_int*>(rev_indices->data);
  csr.column_indices.length = rev_indices->shape[0];
  const int64_t x_len = ComputeXLength(out);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out->ctx;
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);

  const DLDataType& dtype = out->dtype;
  const bool has_indirect =
    !(IsNoneArray(lhs_mapping) && IsNoneArray(rhs_mapping) && IsNoneArray(out_mapping));
  const int bcast_ndim = info.out_shape.size();
  const bool req_lhs = !IsNoneArray(grad_lhs);
  const bool req_rhs = !IsNoneArray(grad_rhs);
  BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, {
    DGL_DTYPE_SWITCH(dtype, DType, {
      BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
        BackwardBcastGData<NDim, DType> gdata = AllocBackwardBcastGData<NDim, DType>(
            rtcfg.stream, info,
            lhs_mapping, rhs_mapping, out_mapping,
            lhs, rhs, out, grad_out,
            grad_lhs, grad_rhs);
        //REDUCER_SWITCH(reducer, kDLGPU, DType, Reducer, {
        typedef ReduceSum<kDLGPU, DType> Reducer;
          BINARY_OP_SWITCH(op, DType, BinaryOp, {
            TARGET_SWITCH(lhs_tgt, rhs_tgt, LeftTarget, RightTarget, {
              if (has_indirect) {
                typedef IndirectId<kDLGPU, int64_t> IdGetter;
                CallBackwardBinaryReduceBcast<Mode, NDim, DType, IdGetter, LeftTarget,
                  RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
              } else {
                typedef DirectId<kDLGPU, int64_t> IdGetter;
                CallBackwardBinaryReduceBcast<Mode, NDim, DType, IdGetter, LeftTarget,
                  RightTarget, BinaryOp, Reducer>(rtcfg, csr, &gdata);
              }
            });
          });
        //});
      });
      if (reducer == binary_op::kReduceMean) {
        // TODO(minjie): divide
        LOG(FATAL) << "reduce mean is not supported.";
      }
    });
  });
}

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl
