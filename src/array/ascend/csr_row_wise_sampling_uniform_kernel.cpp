/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_kernel.cpp
 * @brief Multi-core AIV kernel for uniform CSR row-wise sampling on Ascend NPU.
 *
 * Design (v2, 2026-08-21):
 * - 40 vector cores (KERNEL_TYPE_AIV_ONLY). The host computes two GM
 *   tables: row_split[41] (nnz-balanced row-range boundaries, spmm
 *   precedent) and out_starts[41] (output slot where each block starts
 *   writing). Blocks therefore write disjoint output ranges with no
 *   cross-block reduction.
 * - Per row, the CSR window [off, off+deg) of indices (and data) is
 *   bulk-copied GM -> UB with DataCopyPad via VECIN queues, sampled with
 *   scalar reads from the DeQue'd LocalTensor (the DeQue-then-scalar
 *   pattern sddmm_binary_kernel uses reliably), and picked triples are
 *   staged in VECCALC buffers and copied out through the VECOUT queue
 *   one complete Alloc/EnQue/DeQue/Free cycle at a time.
 * - Rows whose degree exceeds the UB window fall back to direct-GM
 *   sampling (v1 path), keeping memory bounded for skewed graphs.
 * - Out-of-range row ids are dropped as empty rows (defense in depth,
 *   matching the CPU path where COOSliceRows discards invalid seeds).
 *
 * History (v1, single-core scalar-GM):
 * - Verified correct (16/16 tests) but used 1 of 40 cores (~0.4x CPU on
 *   100k-row graphs). Replaced by v2.
 */

#include "csr_row_wise_sampling_uniform_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t Xorshift32(uint32_t& x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__aicore__ inline uint32_t RandBelow(uint32_t& state, uint32_t n) {
  if (n == 0) return 0;
  uint32_t r = Xorshift32(state);
  return static_cast<uint32_t>((static_cast<uint64_t>(r) * n) >> 32);
}

}  // namespace

template <typename IdT>
class KernelCsrRowWiseSamplingUniform {
 public:
  __aicore__ inline KernelCsrRowWiseSamplingUniform() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
      GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr, TPipe* pipe) {

    // Struct-pointer tiling access (spmm pattern): field loads through a
    // typed __gm__ pointer are consistently visible across blocks, unlike
    // per-word GetValue on a shared GlobalTensor.
    const __gm__ CsrRowWiseSamplingUniformTiling* tiling =
        (const __gm__ CsrRowWiseSamplingUniformTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    num_samples_ = tiling->num_samples;
    replace_ = tiling->replace;
    has_data_ = tiling->has_data;
    seed_ = tiling->seed;
    select_all_ = tiling->select_all;
    num_total_rows_ = tiling->num_total_rows;
    const uint32_t ub_available = tiling->ub_available;  // runtime-queried

    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_total_rows_ + 1);
    indices_gm_.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, num_rows_);
    out_ptr_gm_.SetGlobalBuffer((__gm__ IdT*)out_ptr, num_rows_ + 1);
    out_rows_gm_.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm_.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm_.SetGlobalBuffer((__gm__ IdT*)out_idxs);

    const uint32_t block_idx = AscendC::GetBlockIdx();
    const uint32_t block_num = AscendC::GetBlockNum();
    block_idx_ = block_idx;
    (void)block_num;  // tables are sized by the same value on the host

    // Row-range and output-slot tables are separate GM parameters, read
    // through typed __gm__ pointers (spmm rowSplit pattern).
    const __gm__ uint32_t* row_split_p = (const __gm__ uint32_t*)row_split;
    const __gm__ uint32_t* out_starts_p = (const __gm__ uint32_t*)out_starts;
    row_begin_ = row_split_p[block_idx];
    row_end_ = row_split_p[block_idx + 1];
    out_start_ = out_starts_p[block_idx];

    // UB layout (total must fit the runtime-queried per-core UB budget):
    // Two double-buffered VECIN windows (indices + data edge ids), one
    // pick-scratch buffer, three VECCALC staging buffers for the output
    // triples, and one double-buffered VECOUT queue. The per-buffer size
    // is the budget divided across ALL buffer instances:
    // 2*2 (VECIN db) + 2*2 (VECOUT db) + 4 (VECCALC) = 12 instances.
    constexpr uint32_t kUbInstances = 2 * kQueueDepth    // win_idx_q_
                                      + 2 * kQueueDepth  // win_data_q_
                                      + 2 * kQueueDepth  // out_q_
                                      + 1                // pick_buf_
                                      + 3;               // out_r/c/e bufs
    window_elems_ = ub_available / kUbInstances / sizeof(IdT);
    pipe->InitBuffer(win_idx_q_, kQueueDepth, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(win_data_q_, kQueueDepth, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(pick_buf_, window_elems_ * sizeof(uint32_t));
    pipe->InitBuffer(out_r_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_c_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_e_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_q_, kQueueDepth, window_elems_ * sizeof(IdT));
  }

  __aicore__ inline void Process() {
    // Idle blocks (when num_rows_ < block_num) exit immediately.
    if (row_begin_ >= row_end_) return;

    uint32_t offset = 0;
    for (uint32_t i = row_begin_; i < row_end_; ++i) {
      out_ptr_gm_.SetValue(i, static_cast<IdT>(out_start_ + offset));
      IdT rid = rows_gm_.GetValue(i);
      if (rid < 0 || rid >= static_cast<IdT>(num_total_rows_)) {
        continue;  // defense in depth: drop invalid seed rows
      }
      IdT off = indptr_gm_.GetValue(static_cast<uint32_t>(rid));
      IdT end = indptr_gm_.GetValue(static_cast<uint32_t>(rid) + 1);
      uint32_t deg = static_cast<uint32_t>(end - off);
      uint32_t num_picks =
          select_all_ ? deg
                      : (replace_ ? (deg == 0 ? 0 : num_samples_)
                                  : (deg < num_samples_ ? deg : num_samples_));

      if (num_picks > 0) {
        uint32_t state = seed_ ^ (i * kGoldenRatioHash + kGoldenRatioOffset);
        if (state == 0) state = kRngFallbackSeed;
        offset +=
            SampleRow(out_start_ + offset, rid, off, deg, num_picks, state);
      }
    }
    if (row_end_ == num_rows_) {
      out_ptr_gm_.SetValue(num_rows_, static_cast<IdT>(out_start_ + offset));
    }
  }

 private:
  // Samples one row. Small-degree rows go through UB (bulk copy in,
  // scalar sampling, bulk copy out); huge-degree rows read GM directly.
  __aicore__ inline uint32_t SampleRow(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks,
      uint32_t& state) {
    if (deg <= window_elems_) {
      return SampleRowThroughUb(out_pos, rid, off, deg, num_picks, state);
    }
    return SampleRowDirectGm(out_pos, rid, off, deg, num_picks, state);
  }

  __aicore__ inline uint32_t SampleRowThroughUb(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks,
      uint32_t& state) {
    const uint32_t copy_bytes = deg * sizeof(IdT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};

    // Load the indices window (and data window if present).
    LocalTensor<IdT> win_idx = win_idx_q_.AllocTensor<IdT>();
    DataCopyPad(win_idx, indices_gm_[off], cp, pad);
    win_idx_q_.EnQue(win_idx);
    win_idx = win_idx_q_.DeQue<IdT>();

    LocalTensor<IdT> win_data;
    if (has_data_) {
      win_data = win_data_q_.AllocTensor<IdT>();
      DataCopyPad(win_data, data_gm_[off], cp, pad);
      win_data_q_.EnQue(win_data);
      win_data = win_data_q_.DeQue<IdT>();
    }

    // Pick local indices into scratch.
    LocalTensor<uint32_t> picks = pick_buf_.Get<uint32_t>();
    if (select_all_ || (!replace_ && num_picks == deg)) {
      for (uint32_t j = 0; j < num_picks; ++j) picks.SetValue(j, j);
    } else if (replace_) {
      for (uint32_t j = 0; j < num_picks; ++j) {
        picks.SetValue(j, RandBelow(state, deg));
      }
    } else {
      // Algorithm R reservoir over local indices.
      for (uint32_t j = 0; j < num_picks; ++j) picks.SetValue(j, j);
      for (uint32_t i2 = num_picks; i2 < deg; ++i2) {
        uint32_t j = RandBelow(state, i2 + 1);
        if (j < num_picks) picks.SetValue(j, i2);
      }
    }

    // Materialize the three output arrays into VECCALC staging buffers.
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    for (uint32_t j = 0; j < num_picks; ++j) {
      uint32_t local = picks.GetValue(j);
      out_r.SetValue(j, rid);
      out_c.SetValue(j, win_idx.GetValue(local));
      out_e.SetValue(
          j,
          has_data_ ? win_data.GetValue(local) : static_cast<IdT>(off + local));
    }

    // Copy each output array out through the VECOUT queue, one complete
    // Alloc -> EnQue -> DeQue -> DataCopyPad -> Free cycle at a time.
    CopyOutStaged(out_r, out_rows_gm_[out_pos], num_picks);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], num_picks);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], num_picks);

    win_idx_q_.FreeTensor(win_idx);
    if (has_data_) win_data_q_.FreeTensor(win_data);
    return num_picks;
  }

  // Fallback for rows whose degree overflows the UB window: v1's direct
  // GM scalar path, unchanged semantics.
  __aicore__ inline uint32_t SampleRowDirectGm(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks,
      uint32_t& state) {
    if (select_all_ || (!replace_ && num_picks == deg)) {
      for (uint32_t j = 0; j < num_picks; ++j) {
        WritePickGm(out_pos + j, rid, off + static_cast<IdT>(j));
      }
    } else if (replace_) {
      for (uint32_t j = 0; j < num_picks; ++j) {
        uint32_t idx = RandBelow(state, deg);
        WritePickGm(out_pos + j, rid, off + static_cast<IdT>(idx));
      }
    } else {
      for (uint32_t j = 0; j < num_picks; ++j)
        out_idxs_gm_.SetValue(out_pos + j, static_cast<IdT>(j));
      for (uint32_t i2 = num_picks; i2 < deg; ++i2) {
        uint32_t j = RandBelow(state, i2 + 1);
        if (j < num_picks)
          out_idxs_gm_.SetValue(out_pos + j, static_cast<IdT>(i2));
      }
      for (uint32_t j = 0; j < num_picks; ++j) {
        IdT local = out_idxs_gm_.GetValue(out_pos + j);
        IdT picked = off + local;
        out_rows_gm_.SetValue(out_pos + j, rid);
        out_cols_gm_.SetValue(
            out_pos + j, indices_gm_.GetValue(static_cast<uint32_t>(picked)));
        out_idxs_gm_.SetValue(
            out_pos + j, has_data_
                             ? data_gm_.GetValue(static_cast<uint32_t>(picked))
                             : picked);
      }
    }
    return num_picks;
  }

  __aicore__ inline void WritePickGm(uint32_t pos, IdT rid, IdT picked) {
    out_rows_gm_.SetValue(pos, rid);
    out_cols_gm_.SetValue(
        pos, indices_gm_.GetValue(static_cast<uint32_t>(picked)));
    out_idxs_gm_.SetValue(
        pos,
        has_data_ ? data_gm_.GetValue(static_cast<uint32_t>(picked)) : picked);
  }

  static constexpr uint32_t kQueueDepth = 2;  // double buffering

  // Copies `count` elements from a VECCALC staging tensor to GM through
  // the VECOUT queue, one complete Alloc/EnQue/DeQue/Free cycle.
  __aicore__ inline void CopyOutStaged(
      LocalTensor<IdT>& staging, GlobalTensor<IdT> dst, uint32_t count) {
    LocalTensor<IdT> out = out_q_.AllocTensor<IdT>();
    for (uint32_t j = 0; j < count; ++j) {
      out.SetValue(j, staging.GetValue(j));
    }
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(IdT)), 0, 0, 0};
    DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
    out_q_.EnQue(out);
    LocalTensor<IdT> ready = out_q_.DeQue<IdT>();
    DataCopyPad(dst, ready, cp);
    out_q_.FreeTensor(ready);
  }

  GlobalTensor<IdT> indptr_gm_, indices_gm_, data_gm_, rows_gm_;
  GlobalTensor<IdT> out_ptr_gm_, out_rows_gm_, out_cols_gm_, out_idxs_gm_;
  TQue<TPosition::VECIN, kQueueDepth> win_idx_q_, win_data_q_;
  TQue<TPosition::VECOUT, kQueueDepth> out_q_;
  TBuf<TPosition::VECCALC> pick_buf_;
  TBuf<TPosition::VECCALC> out_r_buf_, out_c_buf_, out_e_buf_;
  uint32_t num_rows_ = 0, num_samples_ = 0, replace_ = 0, has_data_ = 0;
  uint32_t seed_ = 0, select_all_ = 0, num_total_rows_ = 0, out_start_ = 0;
  uint32_t row_begin_ = 0, row_end_ = 0, window_elems_ = 0;
  uint32_t block_idx_ = 0;
};

extern "C" __global__ __aicore__ void csr_row_wise_sampling_uniform_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingUniform<int32_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, out_ptr, out_rows, out_cols, out_idxs,
      row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_uniform_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingUniform<int64_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, out_ptr, out_rows, out_cols, out_idxs,
      row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}
