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
