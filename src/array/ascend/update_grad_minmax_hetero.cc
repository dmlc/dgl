/*
 * @file src/array/ascend/update_grad_minmax_hetero.cc
 * @brief Host launcher + template specializations for UpdateGradMinMax_hetero
 *        on Ascend NPU (conditional atomic scatter-add for hetero graph
 *        gradient update).
 *
 * Mirrors scatter_add.cc / segment_reduce.cc:
 *   - aclrtlaunch_*: tiling malloc'd on GM + aclrtMemcpy H2D + launch +
 *     aclrtSynchronizeStream + aclrtFree.
 *   - int64 idx/idx_etype -> int32 conversion via
 *     CopyTo(CPU) + AsNumBits(32) + CopyTo(NPU).
 *   - dtype coverage: int32/int64 x float32 supported; half/double
 *     LOG(FATAL) placeholders.
 *
 * Host logic mirrors CUDA segment_reduce.cuh:184-225 / CPU
 * segment_reduce.h:116-162: the host iterates over edge types, performs the
 * copy_lhs same-(src,dst)-ntype de-duplication, and launches one kernel per
 * relation.
 */
#ifdef DGL_USE_PYTORCH_NPU_STREAM
// Include torch_npu stream headers FIRST: c10 redefines LOG/CHECK macros;
// dmlc's definitions below must win so LOG(FATAL) throws (catchable) instead
// of aborting the process.
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#endif

#include <dgl/array.h>
#include "../kernel_decl.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>
#define ASCEND_CALL(func)                                    \
  {                                                          \
    aclError e = (func);                                     \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

struct HeteroGradTilingData {
  uint32_t N;
  uint32_t M;
  uint32_t featDim;
  uint32_t type;
};

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_kernel_update_grad_minmax_hetero(
    uint32_t blockDim, aclrtStream stream, void* feat, void* idx,
    void* idx_etype, void* out, void* tiling_ptr);

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl

namespace {

// Resolve the stream to launch on. torch_npu executes eager ops on its own
// NPU stream, so DGL must use the same stream for cross-framework ordering
// (review: stream-granularity sync instead of aclrtSynchronizeDevice).
#ifdef DGL_USE_PYTORCH_NPU_STREAM
inline aclrtStream ResolveStream(int64_t device_id) {
  return c10_npu::getCurrentNPUStream(
             static_cast<c10::DeviceIndex>(device_id)).stream();
}
// torch_npu runs on its own ACL context; aclrtSynchronizeStream from DGL's
// default context does not reliably wait on torch streams, so use torch's
// own wrapper (stream-granularity, review: not device-wide sync).
inline void SyncStream(int64_t device_id) {
  c10_npu::stream_synchronize(ResolveStream(device_id));
}
#else
inline aclrtStream ResolveStream(int64_t) {
  return dgl::runtime::getCurrentAscendStream();
}
inline void SyncStream(int64_t) {
  aclrtStream s = dgl::runtime::getCurrentAscendStream();
  aclError e = aclrtSynchronizeStream(s);
  CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e;
}
#endif

}  // namespace
#endif  // DGL_USE_ASCEND

namespace dgl {
namespace aten {

#ifdef DGL_USE_ASCEND
namespace {

inline uint32_t ComputeBlockDim(int64_t n) {
  return static_cast<uint32_t>(n > 0 ? std::min<int64_t>(n, 40) : 1);
}

template <typename DType>
void LaunchUpdateGradMinMaxHeteroKernel(
    int64_t n, int64_t m, int64_t feat_dim, uint32_t type,
    const void* feat_ptr, const void* idx_ptr, const void* idx_etype_ptr,
    void* out_ptr, aclrtStream stream) {
  uint32_t block_dim = ComputeBlockDim(n);

  HeteroGradTilingData tiling = {
      static_cast<uint32_t>(n), static_cast<uint32_t>(m),
      static_cast<uint32_t>(feat_dim), type};

  void* tiling_device = nullptr;
  ASCEND_CALL(aclrtMalloc(
      &tiling_device, sizeof(HeteroGradTilingData),
      ACL_MEM_MALLOC_HUGE_FIRST));
  ASCEND_CALL(aclrtMemcpy(
      tiling_device, sizeof(HeteroGradTilingData), &tiling,
      sizeof(HeteroGradTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

  aclError launch_err = aclrtlaunch_kernel_update_grad_minmax_hetero(
      block_dim, stream, const_cast<void*>(feat_ptr),
      const_cast<void*>(idx_ptr), const_cast<void*>(idx_etype_ptr), out_ptr,
      tiling_device);
  if (launch_err != ACL_SUCCESS) {
    ASCEND_CALL(aclrtFree(tiling_device));
    LOG(FATAL) << "UpdateGradMinMax_hetero kernel launch failed with error "
                  "code: "
               << launch_err;
  }

  ASCEND_CALL(aclrtSynchronizeStream(stream));
  ASCEND_CALL(aclrtFree(tiling_device));
}

template <typename DType>
void UpdateGradMinMaxHeteroAscendImpl(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  if (op != "copy_lhs" && op != "copy_rhs") {
    LOG(FATAL) << "Unsupported binary operator: " << op;
  }

  DGLContext ctx = feat.front()->ctx;
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  // Stream-granularity sync (review: aclrtSynchronizeDevice waits on all
  // streams device-wide, blocking far more than needed).
  aclrtStream stream = ResolveStream(ctx.device_id);
  SyncStream(ctx.device_id);

  std::vector<std::vector<dgl_id_t>> src_dst_ntypes(
      graph->NumVertexTypes(), std::vector<dgl_id_t>());

  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_id_t dst_ntype = pair.first;
    const dgl_id_t src_ntype = pair.second;

    auto& seen = src_dst_ntypes[dst_ntype];
    auto same_src_dst_ntype = std::find(
        std::begin(seen), std::end(seen), src_ntype);
    if (op == "copy_lhs" && same_src_dst_ntype != std::end(seen)) continue;
    seen.push_back(src_ntype);

    const uint32_t type =
        (op == "copy_lhs") ? static_cast<uint32_t>(src_ntype)
                           : static_cast<uint32_t>(etype);

    int64_t n = feat[dst_ntype]->shape[0];
    int64_t m = (*out)[type]->shape[0];
    int64_t dim = 1;
    for (int i = 1; i < (*out)[type]->ndim; ++i) dim *= (*out)[type]->shape[i];

    LaunchUpdateGradMinMaxHeteroKernel<DType>(
        n, m, dim, type, feat[dst_ntype]->data, idx[dst_ntype]->data,
        idx_etype[dst_ntype]->data, (*out)[type]->data, stream);
  }
}

}  // namespace
#endif  // DGL_USE_ASCEND

// ---- Supported: int32/int64 x float32 ----

template <>
void UpdateGradMinMax_hetero<kDGLAscend, int32_t, float>(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
#ifdef DGL_USE_ASCEND
  UpdateGradMinMaxHeteroAscendImpl<float>(graph, op, feat, idx, idx_etype, out);
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                 "-DUSE_ASCEND=ON";
#endif
}

template <>
void UpdateGradMinMax_hetero<kDGLAscend, int64_t, float>(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
#ifdef DGL_USE_ASCEND
  // int64 idx + idx_etype -> int32 conversion.
  // Stream-granularity sync before D2H CopyTo to avoid stream race
  // (review: aclrtSynchronizeDevice waits on all streams device-wide).
  DGLContext ctx = feat.front()->ctx;
  DGLContext cpu_ctx{kDGLCPU, 0};
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));
  SyncStream(ctx.device_id);
  std::vector<NDArray> idx_i32, idx_etype_i32;
  idx_i32.reserve(idx.size());
  idx_etype_i32.reserve(idx_etype.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    NDArray idx_cpu = idx[i].CopyTo(cpu_ctx);
    IdArray idx_i32_cpu = aten::AsNumBits(idx_cpu, 32);
    idx_i32.push_back(idx_i32_cpu.CopyTo(ctx));

    NDArray idx_etype_cpu = idx_etype[i].CopyTo(cpu_ctx);
    IdArray idx_etype_i32_cpu = aten::AsNumBits(idx_etype_cpu, 32);
    idx_etype_i32.push_back(idx_etype_i32_cpu.CopyTo(ctx));
  }
  UpdateGradMinMaxHeteroAscendImpl<float>(
      graph, op, feat, idx_i32, idx_etype_i32, out);
#else
  LOG(FATAL) << "Ascend support is not compiled. Please compile with "
                 "-DUSE_ASCEND=ON";
#endif
}

// ---- Unsupported dtypes ----

template <>
void UpdateGradMinMax_hetero<kDGLAscend, int32_t, uint16_t>(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  LOG(FATAL) << "Current Ascend UpdateGradMinMax_hetero kernel only supports "
                 "float features.";
}

template <>
void UpdateGradMinMax_hetero<kDGLAscend, int64_t, uint16_t>(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  LOG(FATAL) << "Current Ascend UpdateGradMinMax_hetero kernel only supports "
                 "float features.";
}

template <>
void UpdateGradMinMax_hetero<kDGLAscend, int32_t, double>(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  LOG(FATAL) << "Current Ascend UpdateGradMinMax_hetero kernel only supports "
                 "float features.";
}

template <>
void UpdateGradMinMax_hetero<kDGLAscend, int64_t, double>(
    const HeteroGraphPtr& graph, const std::string& op,
    const std::vector<NDArray>& feat, const std::vector<NDArray>& idx,
    const std::vector<NDArray>& idx_etype, std::vector<NDArray>* out) {
  LOG(FATAL) << "Current Ascend UpdateGradMinMax_hetero kernel only supports "
                 "float features.";
}

}  // namespace aten
}  // namespace dgl
