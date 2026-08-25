#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t TILE_SIZE = 1024;

// =============== int32 版本 ===============
class KernelRangeI32 {
public:
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR tiling_ptr) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

        GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
        this->n = tilingGm.GetValue(0);
        this->low = static_cast<int32_t>(tilingGm.GetValue(2));

        uint32_t block_id = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        uint32_t chunk = (n + block_num - 1) / block_num;
        this->start = block_id * chunk;
        this->end = (start + chunk > n) ? n : start + chunk;

        pipe.InitBuffer(tileBuf, TILE_SIZE * sizeof(int32_t));
        dstGm.SetGlobalBuffer((__gm__ int32_t*)dst, n);
    }

    __aicore__ inline void Process() {
        auto tile = tileBuf.Get<int32_t>();
        uint32_t offset = start;
        int32_t base = low + static_cast<int32_t>(start);

        while (offset < end) {
            uint32_t tile_len = end - offset;
            if (tile_len > TILE_SIZE) tile_len = TILE_SIZE;

            for (uint32_t i = 0; i < tile_len; i++) {
                tile.SetValue(i, base + static_cast<int32_t>(i));
            }

            DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(tile_len * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(dstGm[offset], tile, copyParams);

            offset += tile_len;
            base += static_cast<int32_t>(tile_len);
        }
    }

private:
    AscendC::TPipe pipe;
    TBuf<TPosition::VECCALC> tileBuf;
    GlobalTensor<int32_t> dstGm;
    uint32_t n, start, end;
    int32_t low;
};

extern "C" __global__ __aicore__ void range_i32(
    GM_ADDR dst, GM_ADDR tiling_ptr) {
    KernelRangeI32 op;
    op.Init(dst, tiling_ptr);
    op.Process();
}

// =============== int64 版本 ===============
class KernelRangeI64 {
public:
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR tiling_ptr) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

        GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 3);
        this->n = tilingGm.GetValue(0);
        this->low = (static_cast<int64_t>(tilingGm.GetValue(1)) << 32) |
                    static_cast<int64_t>(tilingGm.GetValue(2));

        uint32_t block_id = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        uint32_t chunk = (n + block_num - 1) / block_num;
        this->start = block_id * chunk;
        this->end = (start + chunk > n) ? n : start + chunk;

        pipe.InitBuffer(tileBuf, TILE_SIZE * sizeof(int64_t));
        dstGm.SetGlobalBuffer((__gm__ int64_t*)dst, n);
    }

    __aicore__ inline void Process() {
        auto tile = tileBuf.Get<int64_t>();
        uint32_t offset = start;
        int64_t base = low + static_cast<int64_t>(start);

        while (offset < end) {
            uint32_t tile_len = end - offset;
            if (tile_len > TILE_SIZE) tile_len = TILE_SIZE;

            for (uint32_t i = 0; i < tile_len; i++) {
                tile.SetValue(i, base + static_cast<int64_t>(i));
            }

            DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(tile_len * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(dstGm[offset], tile, copyParams);

            offset += tile_len;
            base += static_cast<int64_t>(tile_len);
        }
    }

private:
    AscendC::TPipe pipe;
    TBuf<TPosition::VECCALC> tileBuf;
    GlobalTensor<int64_t> dstGm;
    uint32_t n, start, end;
    int64_t low;
};

extern "C" __global__ __aicore__ void range_i64(
    GM_ADDR dst, GM_ADDR tiling_ptr) {
    KernelRangeI64 op;
    op.Init(dst, tiling_ptr);
    op.Process();
}
