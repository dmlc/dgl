#include "kernel_operator.h"

using namespace AscendC;

#define BLOCK_RANGE(n, start_var, end_var) \
  uint32_t block_id = GetBlockIdx();       \
  uint32_t block_num = GetBlockNum();      \
  uint32_t chunk = (n + block_num - 1) / block_num; \
  start_var = block_id * chunk;            \
  end_var = (start_var + chunk > n) ? n : start_var + chunk

// int32 kernels
extern "C" __global__ __aicore__ void binary_l_lt_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar_raw = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                        static_cast<int64_t>(tilingGm.GetValue(2));
  int32_t scalar = static_cast<int32_t>(scalar_raw);
  GlobalTensor<int32_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int32_t*)lhs, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(lhsGm.GetValue(i) < scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_gt_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar_raw = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                        static_cast<int64_t>(tilingGm.GetValue(2));
  int32_t scalar = static_cast<int32_t>(scalar_raw);
  GlobalTensor<int32_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int32_t*)lhs, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(lhsGm.GetValue(i) > scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_le_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar_raw = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                        static_cast<int64_t>(tilingGm.GetValue(2));
  int32_t scalar = static_cast<int32_t>(scalar_raw);
  GlobalTensor<int32_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int32_t*)lhs, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(lhsGm.GetValue(i) <= scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_ge_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar_raw = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                        static_cast<int64_t>(tilingGm.GetValue(2));
  int32_t scalar = static_cast<int32_t>(scalar_raw);
  GlobalTensor<int32_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int32_t*)lhs, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(lhsGm.GetValue(i) >= scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_eq_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar_raw = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                        static_cast<int64_t>(tilingGm.GetValue(2));
  int32_t scalar = static_cast<int32_t>(scalar_raw);
  GlobalTensor<int32_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int32_t*)lhs, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(lhsGm.GetValue(i) == scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_ne_i32(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar_raw = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                        static_cast<int64_t>(tilingGm.GetValue(2));
  int32_t scalar = static_cast<int32_t>(scalar_raw);
  GlobalTensor<int32_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int32_t*)lhs, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(lhsGm.GetValue(i) != scalar));
  }
}

// int64 kernels
extern "C" __global__ __aicore__ void binary_l_lt_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));
  GlobalTensor<int64_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int64_t*)lhs, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(lhsGm.GetValue(i) < scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_gt_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));
  GlobalTensor<int64_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int64_t*)lhs, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(lhsGm.GetValue(i) > scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_le_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));
  GlobalTensor<int64_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int64_t*)lhs, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(lhsGm.GetValue(i) <= scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_ge_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));
  GlobalTensor<int64_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int64_t*)lhs, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(lhsGm.GetValue(i) >= scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_eq_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));
  GlobalTensor<int64_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int64_t*)lhs, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(lhsGm.GetValue(i) == scalar));
  }
}

extern "C" __global__ __aicore__ void binary_l_ne_i64(
    GM_ADDR lhs, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
  uint32_t n = tilingGm.GetValue(0);
  int64_t scalar = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));
  GlobalTensor<int64_t> lhsGm;
  lhsGm.SetGlobalBuffer((__gm__ int64_t*)lhs, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  uint32_t start, end;
  BLOCK_RANGE(n, start, end);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(lhsGm.GetValue(i) != scalar));
  }
}

