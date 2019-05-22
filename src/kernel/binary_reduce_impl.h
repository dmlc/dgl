/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce_impl.h
 * \brief Implementations of binary reduce operations.
 */
#ifndef DGL_KERNEL_BINARY_REDUCE_IMPL_H_
#define DGL_KERNEL_BINARY_REDUCE_IMPL_H_

#include <minigun/minigun.h>
#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>

#include <algorithm>
#include <string>

#include "../runtime/cuda/cuda_common.h"
#include "./binary_reduce.h"
#include "./binary_reduce_impl_decl.h"
#include "./utils.h"

namespace dgl {
namespace kernel {

/****************************************************
 * BinaryOpReduce
 ****************************************************/

template <int XPU, typename DType, typename Reducer>
GData<DType> AllocGData(
    const DLContext& ctx, int64_t x_len,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_mapping, runtime::NDArray out_data) {
  // GData
  GData<DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // fill out data with zero values
  utils::Fill<XPU>(ctx, gdata.out_data, utils::NElements(out_data), Zero<Reducer>::value);
  return gdata;
}

template <int XPU>
void BinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    runtime::NDArray indptr, runtime::NDArray indices,
    runtime::NDArray rev_indptr, runtime::NDArray rev_indices,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_mapping, runtime::NDArray out_data) {
  using runtime::NDArray;
  using minigun::Csr;
  // device
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  Csr csr = utils::CreateCsr(indptr, indices);
  Csr rev_csr = utils::CreateCsr(rev_indptr, rev_indices);

  const int64_t x_len = utils::ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif
  const DLDataType& dtype = out_data->dtype;
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
      GData<DType> gdata = AllocGData<XPU, DType, Reducer>(
          rtcfg.ctx, x_len, lhs_mapping, rhs_mapping,
          lhs_data, rhs_data, out_mapping, out_data);
      BINARY_OP_SWITCH(op, DType, BinaryOp, {
        TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
          CallBinaryReduce<XPU, DType, LeftTarget,
            RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
        });
      });
    });
  });
}

template <int XPU>
void BinaryReduceImpl_v2(
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping) {
  using runtime::NDArray;
  using minigun::Csr;
  // device
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  const auto& adj = graph->GetAdj(false, "csr");
  const auto& rev_adj = graph->GetAdj(true, "csr");
  Csr csr = utils::CreateCsr(adj[0], adj[1]);
  Csr rev_csr = utils::CreateCsr(rev_adj[0], rev_adj[1]);

  const int64_t x_len = utils::ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif
  const DLDataType& dtype = out_data->dtype;
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
      GData<DType> gdata = AllocGData<XPU, DType, Reducer>(
          rtcfg.ctx, x_len, lhs_mapping, rhs_mapping,
          lhs_data, rhs_data, out_mapping, out_data);
      BINARY_OP_SWITCH(op, DType, BinaryOp, {
        TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
          CallBinaryReduce<XPU, DType, LeftTarget,
            RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
        });
      });
    });
  });
}

/****************************************************
 * BackwardBinaryOpReduce
 ****************************************************/

template <int XPU, typename DType>
BackwardGData<DType> AllocBackwardGData(
    const DLContext& ctx, int64_t x_len,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data) {
  // GData
  BackwardGData<DType> gdata;
  gdata.x_length = x_len;
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  gdata.grad_out_data = static_cast<DType*>(grad_out_data->data);
  if (!utils::IsNoneArray(grad_lhs_data)) {
    gdata.grad_lhs_data = static_cast<DType*>(grad_lhs_data->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_lhs_data, utils::NElements(grad_lhs_data),
                static_cast<DType>(0));
  }
  if (!utils::IsNoneArray(grad_rhs_data)) {
    gdata.grad_rhs_data = static_cast<DType*>(grad_rhs_data->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_rhs_data, utils::NElements(grad_rhs_data),
                static_cast<DType>(0));
  }
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  return gdata;
}

template <int XPU>
void BackwardBinaryReduceImpl(
    const std::string& reducer,
    const std::string& op,
    runtime::NDArray indptr, runtime::NDArray indices,
    runtime::NDArray rev_indptr, runtime::NDArray rev_indices,
    binary_op::Target lhs, binary_op::Target rhs,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data, runtime::NDArray out_data,
    runtime::NDArray grad_out_data,
    runtime::NDArray grad_lhs_data, runtime::NDArray grad_rhs_data) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  // device
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  Csr csr = utils::CreateCsr(indptr, indices);
  Csr rev_csr = utils::CreateCsr(rev_indptr, rev_indices);

  const int64_t x_len = utils::ComputeXLength(out_data);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out_data->dtype;
  const bool req_lhs = !utils::IsNoneArray(grad_lhs_data);
  const bool req_rhs = !utils::IsNoneArray(grad_rhs_data);
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    BackwardGData<DType> gdata = AllocBackwardGData<XPU, DType>(
        rtcfg.ctx, x_len, lhs_mapping, rhs_mapping, out_mapping,
        lhs_data, rhs_data, out_data, grad_out_data,
        grad_lhs_data, grad_rhs_data);
    BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, {
      REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
        BINARY_OP_SWITCH(op, DType, BinaryOp, {
          TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
            CallBackwardBinaryReduce<XPU, Mode, DType, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
          });
        });
      });
    });
  });
}

/****************************************************
 * BinaryOpReduceBcast
 ****************************************************/

template <int XPU, int NDim, typename DType, typename Reducer>
BcastGData<NDim, DType> AllocBcastGData(
    const DLContext& ctx, const BcastInfo& info,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data, runtime::NDArray rhs_data,
    runtime::NDArray out_mapping, runtime::NDArray out_data) {
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
  gdata.lhs_len = utils::Prod(info.lhs_shape);
  gdata.rhs_len = utils::Prod(info.rhs_shape);
  gdata.out_len = utils::Prod(info.out_shape);
  // data
  gdata.lhs_data = static_cast<DType*>(lhs_data->data);
  gdata.rhs_data = static_cast<DType*>(rhs_data->data);
  gdata.out_data = static_cast<DType*>(out_data->data);
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // fill out data with zero values
  utils::Fill<XPU>(ctx, gdata.out_data, utils::NElements(out_data), Zero<Reducer>::value);
  return gdata;
}

template <int XPU>
void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    runtime::NDArray indptr, runtime::NDArray indices,
    runtime::NDArray rev_indptr, runtime::NDArray rev_indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  Csr csr = utils::CreateCsr(indptr, indices);
  Csr rev_csr = utils::CreateCsr(rev_indptr, rev_indices);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int64_t x_len = utils::ComputeXLength(out_data);
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out_data->dtype;
  const int bcast_ndim = info.out_shape.size();
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
      BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
        BcastGData<NDim, DType> gdata = AllocBcastGData<XPU, NDim, DType, Reducer>(
            rtcfg.ctx, info, lhs_mapping, rhs_mapping,
            lhs_data, rhs_data, out_mapping, out_data);
        BINARY_OP_SWITCH(op, DType, BinaryOp, {
          TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
            CallBinaryReduceBcast<XPU, NDim, DType, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
          });
        });
      });
    });
  });
}

template <int XPU>
void BinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    runtime::NDArray indptr, runtime::NDArray indices,
    runtime::NDArray rev_indptr, runtime::NDArray rev_indices,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_mapping,
    runtime::NDArray out_data) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  Csr csr = utils::CreateCsr(indptr, indices);
  Csr rev_csr = utils::CreateCsr(rev_indptr, rev_indices);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int64_t x_len = utils::ComputeXLength(out_data);
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out_data->dtype;
  const int bcast_ndim = info.out_shape.size();
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
      BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
        BcastGData<NDim, DType> gdata = AllocBcastGData<XPU, NDim, DType, Reducer>(
            rtcfg.ctx, info, lhs_mapping, rhs_mapping,
            lhs_data, rhs_data, out_mapping, out_data);
        BINARY_OP_SWITCH(op, DType, BinaryOp, {
          TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
            CallBinaryReduceBcast<XPU, NDim, DType, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
          });
        });
      });
    });
  });
}

template <int XPU>
void BinaryReduceBcastImpl_v2(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    const ImmutableGraph* graph,
    binary_op::Target lhs,
    binary_op::Target rhs,
    runtime::NDArray lhs_data,
    runtime::NDArray rhs_data,
    runtime::NDArray out_data,
    runtime::NDArray lhs_mapping,
    runtime::NDArray rhs_mapping,
    runtime::NDArray out_mapping) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  const auto& adj = graph->GetAdj(false, "csr");
  const auto& rev_adj = graph->GetAdj(true, "csr");
  Csr csr = utils::CreateCsr(adj[0], adj[1]);
  Csr rev_csr = utils::CreateCsr(rev_adj[0], rev_adj[1]);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out_data->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int64_t x_len = utils::ComputeXLength(out_data);
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out_data->dtype;
  const int bcast_ndim = info.out_shape.size();
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
      BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
        BcastGData<NDim, DType> gdata = AllocBcastGData<XPU, NDim, DType, Reducer>(
            rtcfg.ctx, info, lhs_mapping, rhs_mapping,
            lhs_data, rhs_data, out_mapping, out_data);
        BINARY_OP_SWITCH(op, DType, BinaryOp, {
          TARGET_SWITCH(lhs, rhs, LeftTarget, RightTarget, {
            CallBinaryReduceBcast<XPU, NDim, DType, LeftTarget,
              RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
          });
        });
      });
    });
  });
}


/****************************************************
 * BackwardBinaryOpReduceBcast
 ****************************************************/

template <int XPU, int NDim, typename DType>
BackwardBcastGData<NDim, DType> AllocBackwardBcastGData(
    const DLContext& ctx, const BcastInfo& info,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs) {
  // GData
  BackwardBcastGData<NDim, DType> gdata;
  // dim, shape and stride
  gdata.ndim = info.lhs_shape.size();
  gdata.lhs_len = utils::Prod(info.lhs_shape);
  gdata.rhs_len = utils::Prod(info.rhs_shape);
  gdata.out_len = utils::Prod(info.out_shape);
  std::copy(info.lhs_shape.begin(), info.lhs_shape.end(), gdata.lhs_shape);
  std::copy(info.lhs_stride.begin(), info.lhs_stride.end(), gdata.lhs_stride);
  std::copy(info.rhs_shape.begin(), info.rhs_shape.end(), gdata.rhs_shape);
  std::copy(info.rhs_stride.begin(), info.rhs_stride.end(), gdata.rhs_stride);
  std::copy(info.out_shape.begin(), info.out_shape.end(), gdata.out_shape);
  std::copy(info.out_stride.begin(), info.out_stride.end(), gdata.out_stride);
  // mappings
  if (!utils::IsNoneArray(lhs_mapping)) {
    gdata.lhs_mapping = static_cast<int64_t*>(lhs_mapping->data);
  }
  if (!utils::IsNoneArray(rhs_mapping)) {
    gdata.rhs_mapping = static_cast<int64_t*>(rhs_mapping->data);
  }
  if (!utils::IsNoneArray(out_mapping)) {
    gdata.out_mapping = static_cast<int64_t*>(out_mapping->data);
  }
  // data
  gdata.lhs_data = static_cast<DType*>(lhs->data);
  gdata.rhs_data = static_cast<DType*>(rhs->data);
  gdata.out_data = static_cast<DType*>(out->data);
  gdata.grad_out_data = static_cast<DType*>(grad_out->data);
  if (!utils::IsNoneArray(grad_lhs)) {
    gdata.grad_lhs_data = static_cast<DType*>(grad_lhs->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_lhs_data, utils::NElements(grad_lhs),
                static_cast<DType>(0));
  }
  if (!utils::IsNoneArray(grad_rhs)) {
    gdata.grad_rhs_data = static_cast<DType*>(grad_rhs->data);
    // fill out data with zero values
    utils::Fill<XPU>(ctx, gdata.grad_rhs_data, utils::NElements(grad_rhs),
                static_cast<DType>(0));
  }
  return gdata;
}

template <int XPU>
void BackwardBinaryReduceBcastImpl(
    const BcastInfo& info,
    const std::string& reducer,
    const std::string& op,
    runtime::NDArray indptr, runtime::NDArray indices,
    runtime::NDArray rev_indptr, runtime::NDArray rev_indices,
    binary_op::Target lhs_tgt, binary_op::Target rhs_tgt,
    runtime::NDArray lhs_mapping, runtime::NDArray rhs_mapping, runtime::NDArray out_mapping,
    runtime::NDArray lhs, runtime::NDArray rhs, runtime::NDArray out, runtime::NDArray grad_out,
    runtime::NDArray grad_lhs, runtime::NDArray grad_rhs) {
  using runtime::NDArray;
  using minigun::Csr;
#ifdef __CUDACC__
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
#endif
  // Graph
  Csr csr = utils::CreateCsr(indptr, indices);
  Csr rev_csr = utils::CreateCsr(rev_indptr, rev_indices);

  // advance config
  minigun::advance::RuntimeConfig rtcfg;
  rtcfg.ctx = out->ctx;
#ifdef __CUDACC__
  rtcfg.stream = thr_entry->stream;
  const int64_t x_len = utils::ComputeXLength(out);
  const int nt = utils::FindNumThreads(x_len, 64);
  rtcfg.data_num_threads = nt;
  // XXX(minjie): hard-code to let each thread compute two elements to increase
  //              instruction level parallelism
  rtcfg.data_num_blocks = (x_len + (nt * 2) - 1) / (nt * 2);
#endif

  const DLDataType& dtype = out->dtype;
  const int bcast_ndim = info.out_shape.size();
  const bool req_lhs = !utils::IsNoneArray(grad_lhs);
  const bool req_rhs = !utils::IsNoneArray(grad_rhs);
  if (reducer == binary_op::kReduceMean) {
    // TODO(minjie): divide
    LOG(FATAL) << "reduce mean is not supported.";
  }
  DGL_DTYPE_SWITCH(dtype, DType, {
    BCAST_NDIM_SWITCH(bcast_ndim, NDim, {
      BackwardBcastGData<NDim, DType> gdata = AllocBackwardBcastGData<XPU, NDim, DType>(
          rtcfg.ctx, info,
          lhs_mapping, rhs_mapping, out_mapping,
          lhs, rhs, out, grad_out,
          grad_lhs, grad_rhs);
      BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, {
        REDUCER_SWITCH(reducer, XPU, DType, Reducer, {
          BINARY_OP_SWITCH(op, DType, BinaryOp, {
            TARGET_SWITCH(lhs_tgt, rhs_tgt, LeftTarget, RightTarget, {
              CallBackwardBinaryReduceBcast<XPU, Mode, NDim, DType, LeftTarget,
                RightTarget, BinaryOp, Reducer>(rtcfg, csr, rev_csr, &gdata);
            });
          });
        });
      });
    });
  });
}

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_IMPL_H_
