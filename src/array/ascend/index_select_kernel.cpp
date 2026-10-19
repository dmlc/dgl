#include "kernel_operator.h"

using namespace AscendC;

template <typename T, typename IdxT>
class KernelIndexSelect {
 public:
  __aicore__ inline KernelIndexSelect() {}

  __aicore__ inline void Init(
      GM_ADDR src, GM_ADDR idx, GM_ADDR dst, uint32_t n) {
    src_gm.SetGlobalBuffer((__gm__ T *)src, n);
    idx_gm.SetGlobalBuffer((__gm__ IdxT *)idx, n);
    dst_gm.SetGlobalBuffer((__gm__ T *)dst, n);
    n_ = n;
  }

  __aicore__ inline void Process() {
    uint32_t block_id = GetBlockIdx();
    uint32_t block_num = GetBlockNum();
    uint32_t chunk = (n_ + block_num - 1) / block_num;
    uint32_t start = block_id * chunk;
    uint32_t end = (start + chunk > n_) ? n_ : start + chunk;
    for (uint32_t i = start; i < end; i++) {
      IdxT p = idx_gm.GetValue(i);
      dst_gm.SetValue(i, src_gm.GetValue(static_cast<uint32_t>(p)));
    }
  }

 private:
  GlobalTensor<T> src_gm;
  GlobalTensor<IdxT> idx_gm;
  GlobalTensor<T> dst_gm;
  uint32_t n_;
};

// int32 data, int64 index
extern "C" __global__ __aicore__ void index_select_i32_i64(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<int32_t, int64_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// int64 data, int64 index
extern "C" __global__ __aicore__ void index_select_i64_i64(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<int64_t, int64_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// float data, int64 index
extern "C" __global__ __aicore__ void index_select_f32_i64(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<float, int64_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// int32 data, int32 index
extern "C" __global__ __aicore__ void index_select_i32_i32(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<int32_t, int32_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// int64 data, int32 index
extern "C" __global__ __aicore__ void index_select_i64_i32(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<int64_t, int32_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// float data, int32 index
extern "C" __global__ __aicore__ void index_select_f32_i32(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<float, int32_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// double data, int64 index
extern "C" __global__ __aicore__ void index_select_f64_i64(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<double, int64_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

// double data, int32 index
extern "C" __global__ __aicore__ void index_select_f64_i32(
    GM_ADDR src, GM_ADDR idx, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  KernelIndexSelect<double, int32_t> op;
  op.Init(src, idx, dst, n);
  op.Process();
}

