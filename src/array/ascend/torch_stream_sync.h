/**
 * Copyright (c) 2024 by Contributors
 * @file torch_stream_sync.h
 * @brief torch_npu host 队列同步工具(所有 DGL Ascend 算子 launch 前必须调用)。
 *
 * 背景(2026-09-16 定位的跨体系提交竞争):
 *   torch_npu 算子(如 GraphConv 内的 th.matmul、GAT 内的 leaky_relu 等)
 *   不是直接提交到 NPU 设备, 而是先进入 c10_npu 的 host 侧任务队列,
 *   由后台线程异步 flush 到设备; DGL AscendC kernel 则用 ACL 直接 launch,
 *   立即在设备执行。两套提交体系互不感知:
 *     - aclrtSynchronizeDevice/SynchronizeStream 只等待已提交到设备的任务;
 *     - c10_npu::device_synchronize() 同样不 flush host 队列(已实测无效);
 *   因此 DGL kernel 可能在生产者(torch 算子写输入 tensor)完成前执行,
 *   读到全零/过期输入 —— 进程内首次调用必现(实测: GraphConv 首调
 *   gspmm 读到的 ufeat 为 0, 同一块内存二调才有正确值)。
 *
 * 修复: DGL Ascend 算子 launch 前调用 SyncTorchNpuStream(), 即
 *   c10_npu::npuSynchronizeDevice(false) —— torch.npu.synchronize() 的
 *   C++ 底层, 语义 = flush torch_npu host 队列 + 等待设备全部任务完成。
 *
 * 实现说明: 该符号导出自 libtorch_npu.so(无独立 libc10_npu 可供链接),
 *   python 进程中该 so 必然先于 dgl 加载, dlsym(RTLD_DEFAULT) 运行时解析;
 *   函数签名 void(bool) 无 ABI 风险。不做 static null 缓存, 避免
 *   libdgl.so 先于 libtorch_npu.so 加载时首次解析失败后被永久跳过。
 *
 * TODO(ascend): 更优做法是 DGL kernel 直接在 torch 当前 stream 上 launch
 *   (stream 顺序天然保证依赖, 免去同步开销), 需将 getCurrentNPUStream
 *   传给所有 AscendC kernel launch, 涉及面较广, 后续统一改造。
 */
#ifndef DGL_ARRAY_ASCEND_TORCH_STREAM_SYNC_H_
#define DGL_ARRAY_ASCEND_TORCH_STREAM_SYNC_H_

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <dlfcn.h>

namespace dgl {
namespace aten {

/**
 * @brief 防抖版 aclrtSetDevice。
 *
 * 背景: torch_npu 通过自己的设备上下文管理(c10_npu::ExchangeDevice)维护
 * 当前设备; DGL 算子入口的传统 aclrtSetDevice 与之并发时会触发偶发
 * RuntimeError 507033(SetDevice 冲突, 实测 GAT 的 edge_softmax/sddmm/spmm
 * 链路偶发)。进程内设备上下文一旦建立后不会变化, 因此先 aclrtGetDevice
 * 查询、仅在实际不同才设置, 消除与 torch_npu 的并发 SetDevice 竞争。
 */
inline void SetDeviceIfChanged(int device_id) {
  int32_t cur = -1;
  if (aclrtGetDevice(&cur) != ACL_SUCCESS || cur != device_id) {
    aclrtSetDevice(device_id);
  }
}

inline void SyncTorchNpuStream() {
  using SyncFn = void (*)(bool);
  SyncFn npu_sync = reinterpret_cast<SyncFn>(
      dlsym(RTLD_DEFAULT, "_ZN7c10_npu20npuSynchronizeDeviceEb"));
  if (npu_sync == nullptr) {
    // libtorch_npu 未加载或符号不在全局作用域: 显式 dlopen(已加载则仅增引用计数)
    void* h = dlopen("libtorch_npu.so", RTLD_NOW | RTLD_GLOBAL);
    if (h != nullptr) {
      npu_sync = reinterpret_cast<SyncFn>(
          dlsym(h, "_ZN7c10_npu20npuSynchronizeDeviceEb"));
    }
  }
  if (npu_sync != nullptr) {
    npu_sync(false);
  } else {
    // 兜底: 纯 ACL 语义(无 torch 集成场景)
    aclrtSynchronizeDevice();
  }
}

}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND
#endif  // DGL_ARRAY_ASCEND_TORCH_STREAM_SYNC_H_
