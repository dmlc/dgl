#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void as_num_bits_i32_to_i64(
    GM_ADDR src, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  uint32_t block_id = GetBlockIdx();
  uint32_t block_num = GetBlockNum();
  uint32_t chunk = (n + block_num - 1) / block_num;
  uint32_t start = block_id * chunk;
  uint32_t end = (start + chunk > n) ? n : start + chunk;
  GlobalTensor<int32_t> srcGm;
  srcGm.SetGlobalBuffer((__gm__ int32_t*)src, n);
  GlobalTensor<int64_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int64_t>(srcGm.GetValue(i)));
  }
}

extern "C" __global__ __aicore__ void as_num_bits_i64_to_i32(
    GM_ADDR src, GM_ADDR dst, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 1);
  uint32_t n = tilingGm.GetValue(0);
  uint32_t block_id = GetBlockIdx();
  uint32_t block_num = GetBlockNum();
  uint32_t chunk = (n + block_num - 1) / block_num;
  uint32_t start = block_id * chunk;
  uint32_t end = (start + chunk > n) ? n : start + chunk;
  GlobalTensor<int64_t> srcGm;
  srcGm.SetGlobalBuffer((__gm__ int64_t*)src, n);
  GlobalTensor<int32_t> dstGm;
  dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
  for (uint32_t i = start; i < end; i++) {
    dstGm.SetValue(i, static_cast<int32_t>(srcGm.GetValue(i)));
  }
}

