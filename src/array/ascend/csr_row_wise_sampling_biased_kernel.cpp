/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_biased_kernel.cpp
 * @brief Multi-core AIV kernels for tag-bucketed biased CSR row-wise
 *        sampling on Ascend NPU.
 *
 * Design (mirrors the uniform v2 multi-core kernel):
 * - The host computes per-row pick counts with a small helper kernel
 *   (csr_row_wise_sampling_biased_num_picks_*), copies them back, and
 *   builds the same nnz-balanced row_split / out_starts GM tables as spmm.
 *   Blocks therefore write disjoint output ranges with no cross-block
 *   reduction.
 * - Per row, tag_offset[row] (T+1 values) delimits T tag buckets over the
 *   row's edge window. Each draw first picks a bucket with probability
 *   proportional to rem[t] * bias[t] (remaining edges x weight), then an
 *   edge uniformly inside the bucket — matching CPU BiasedChoice.
 * - No-replace draws use in-place Fisher-Yates inside the bucket
 *   ([bucket_head, bucket_head + rem[t]) shrinks by one per draw), which
 *   is duplicate-free by construction; the weight update (rem[t] -= 1) is
 *   exactly the CPU TreeSampler's "decrease" semantics.
 * - Rows whose degree exceeds the UB window fall back to direct-GM
 *   sampling with rejection-based dedup (CPU uses a hash set; same
 *   retry-on-hit semantics).
 * - Defense in depth: invalid seed rows are dropped; per-bucket bounds
 *   are clamped to [0, deg] so corrupted tag_offset values cannot read
 *   outside the row or crash the kernel; bias was clamped to >= 0 on the
 *   host, and a row with total weight 0 is skipped.
 */

#include "csr_row_wise_sampling_biased_tiling.h"
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

// Uniform float in [0, 1).
// AscendC forbids uint<->float casts in __aicore__, so the uint32 state
// is converted via a signed int32 intermediate (weighted-kernel
// precedent).
__aicore__ inline float RandFloat01(uint32_t& state) {
  uint32_t r = Xorshift32(state);
  int32_t signed_r = static_cast<int32_t>(r);  // [-2^31, 2^31)
  float f = static_cast<float>(signed_r);
  return (f + 2147483648.0f) / 4294967296.0f;  // [0, 1)
}

}  // namespace

// ---------------------------------------------------------------------------
// Helper kernel: per-row pick counts (biased nnz).
// ---------------------------------------------------------------------------

template <typename IdxT>
class KernelCsrRowWiseSamplingBiasedNumPicks {
 public:
  __aicore__ inline KernelCsrRowWiseSamplingBiasedNumPicks() {}

  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR tag_offset, GM_ADDR bias, GM_ADDR out,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ CsrRowWiseSamplingBiasedTiling* tiling =
        (const __gm__ CsrRowWiseSamplingBiasedTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    num_samples_ = tiling->num_samples;
    replace_ = tiling->replace;
    select_all_ = tiling->select_all;
    num_total_rows_ = tiling->num_total_rows;
    num_tags_ = tiling->num_tags;

    rows_gm_.SetGlobalBuffer((__gm__ IdxT*)rows, num_rows_);
    tag_offset_gm_.SetGlobalBuffer((__gm__ IdxT*)tag_offset);
    bias_gm_.SetGlobalBuffer((__gm__ float*)bias, num_tags_);
    out_gm_.SetGlobalBuffer((__gm__ uint32_t*)out, num_rows_);

    // Results are staged in UB and flushed per chunk through the VECOUT
    // queue: direct GM scalar stores from concurrent blocks are not
    // reliable (measured on 910B3: writes from some blocks never land).
    pipe->InitBuffer(out_q_, kQueueDepth, kChunkElems * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    const uint32_t block_id = AscendC::GetBlockIdx();
    const uint32_t block_num = AscendC::GetBlockNum();
    const uint32_t chunk = (num_rows_ + block_num - 1) / block_num;
    const uint32_t start = block_id * chunk;
    const uint32_t end =
        (start + chunk > num_rows_) ? num_rows_ : start + chunk;
    for (uint32_t base = start; base < end; base += kChunkElems) {
      const uint32_t count =
          (end - base < kChunkElems) ? (end - base) : kChunkElems;
      ProcessChunk(base, count);
    }
  }

 private:
  static constexpr uint32_t kQueueDepth = 2;     // double buffering
  static constexpr uint32_t kChunkElems = 2048;  // rows staged per flush

  __aicore__ inline void ProcessChunk(uint32_t base, uint32_t count) {
    LocalTensor<uint32_t> out = out_q_.AllocTensor<uint32_t>();
    for (uint32_t j = 0; j < count; ++j) {
      const IdxT rid = rows_gm_.GetValue(base + j);
      uint32_t picks = 0;  // invalid row: consistent with the main kernel
      if (rid >= 0 && rid < static_cast<IdxT>(num_total_rows_)) {
        const uint32_t row_base = static_cast<uint32_t>(rid) * (num_tags_ + 1);
        uint32_t nnz = 0;
        for (uint32_t t = 0; t < num_tags_; ++t) {
          const float w = bias_gm_.GetValue(t);
          if (w > 0.0f) {
            // Clamp protects against corrupted offsets; a legit sorted
            // row has 0 <= lo <= hi <= deg.
            uint32_t lo =
                static_cast<uint32_t>(tag_offset_gm_.GetValue(row_base + t));
            uint32_t hi = static_cast<uint32_t>(
                tag_offset_gm_.GetValue(row_base + t + 1));
            if (lo > hi) continue;
            nnz += hi - lo;
          }
        }
        if (select_all_) {
          picks = nnz;
        } else if (replace_) {
          picks = (nnz == 0) ? 0 : num_samples_;
        } else {
          picks = (nnz < num_samples_) ? nnz : num_samples_;
        }
      }
      out.SetValue(j, picks);
    }
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(uint32_t)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    out_q_.EnQue(out);
    LocalTensor<uint32_t> ready = out_q_.DeQue<uint32_t>();
    DataCopyPad(out_gm_[base], ready, cp);
    out_q_.FreeTensor(ready);
  }

  GlobalTensor<IdxT> rows_gm_, tag_offset_gm_;
  GlobalTensor<float> bias_gm_;
  GlobalTensor<uint32_t> out_gm_;
  TQue<TPosition::VECOUT, kQueueDepth> out_q_;
  uint32_t num_rows_ = 0, num_samples_ = 0, replace_ = 0, select_all_ = 0;
  uint32_t num_total_rows_ = 0, num_tags_ = 0;
};

// ---------------------------------------------------------------------------
// Main kernel: biased row-wise sampling.
// ---------------------------------------------------------------------------

template <typename IdT>
class KernelCsrRowWiseSamplingBiased {
 public:
  __aicore__ inline KernelCsrRowWiseSamplingBiased() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR tag_offset, GM_ADDR bias, GM_ADDR out_rows, GM_ADDR out_cols,
      GM_ADDR out_idxs, GM_ADDR row_split, GM_ADDR out_starts,
      GM_ADDR tiling_ptr, TPipe* pipe) {

    const __gm__ CsrRowWiseSamplingBiasedTiling* tiling =
        (const __gm__ CsrRowWiseSamplingBiasedTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    replace_ = tiling->replace;
    has_data_ = tiling->has_data;
    seed_ = tiling->seed;
    select_all_ = tiling->select_all;
    num_total_rows_ = tiling->num_total_rows;
    num_tags_ = tiling->num_tags;
    const uint32_t ub_available = tiling->ub_available;  // runtime-queried

    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_total_rows_ + 1);
    indices_gm_.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, num_rows_);
    tag_offset_gm_.SetGlobalBuffer((__gm__ IdT*)tag_offset);
    bias_gm_.SetGlobalBuffer((__gm__ float*)bias, num_tags_);
    out_rows_gm_.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm_.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm_.SetGlobalBuffer((__gm__ IdT*)out_idxs);

    const uint32_t block_idx = AscendC::GetBlockIdx();
    block_idx_ = block_idx;

    const __gm__ uint32_t* row_split_p = (const __gm__ uint32_t*)row_split;
    out_starts_p_ = (const __gm__ uint32_t*)out_starts;
    row_begin_ = row_split_p[block_idx];
    row_end_ = row_split_p[block_idx + 1];
    // out_starts is a PER-ROW prefix table (num_rows+1 entries): block b's
    // output starts at the prefix of its first row, not at index b.
    out_start_ = out_starts_p_[row_begin_];

    // UB layout (total must fit the runtime-queried per-core UB budget):
    // Two double-buffered VECIN windows (indices + data edge ids), one
    // pick-scratch buffer, three VECCALC staging buffers for the output
    // triples, one double-buffered VECOUT queue, plus the biased-sampling
    // per-row state (rem[T] uint32 + bias[T] float) and the GM-fallback
    // dedup table. rem/bias are sized for the worst case (kMaxTagCount)
    // and charged to the budget first; the remainder is split across the
    // window instances:
    // 2*2 (VECIN db) + 2*2 (VECOUT db) + 1 (pick) + 3 (out r/c/e) = 12.
    constexpr uint32_t kUbInstances = 2 * kQueueDepth    // win_idx_q_
                                      + 2 * kQueueDepth  // win_data_q_
                                      + 2 * kQueueDepth  // out_q_
                                      + 1                // pick_buf_
                                      + 3;               // out_r/c/e bufs
    const uint32_t state_bytes =
        kMaxTagCount * sizeof(uint32_t) +      // rem_buf_
        kMaxTagCount * sizeof(float) +         // bias_buf_
        2 * (kMaxTagCount + 1) * sizeof(IdT);  // tag_q_ (db)
    uint32_t window_budget =
        (ub_available > state_bytes) ? (ub_available - state_bytes) : 0;
    window_elems_ = window_budget / kUbInstances / sizeof(IdT);

    pipe->InitBuffer(win_idx_q_, kQueueDepth, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(win_data_q_, kQueueDepth, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(pick_buf_, window_elems_ * sizeof(uint32_t));
    pipe->InitBuffer(out_r_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_c_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_e_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_q_, kQueueDepth, window_elems_ * sizeof(IdT));

    // rem/bias live in a single VECCALC TBuf sized for kMaxTagCount so the
    // buffer layout never depends on the launch's T.
    pipe->InitBuffer(
        state_buf_,
        kMaxTagCount * sizeof(uint32_t) + kMaxTagCount * sizeof(float));

    // Tag-row staging: one row of tag_offset (T+1 values) is bulk-copied
    // into UB per row via DataCopyPad instead of T+1 scalar GM reads
    // (measured 26ns per scalar read on 910B3 — the dominant cost at
    // large T). Double-buffered so the copy of row i+1 overlaps the
    // sampling of row i.
    pipe->InitBuffer(tag_q_, kQueueDepth, (kMaxTagCount + 1) * sizeof(IdT));
  }

  __aicore__ inline void Process() {
    // Idle blocks (when num_rows_ < block_num) exit immediately.
    if (row_begin_ >= row_end_) return;

    // Per-block constant: copy bias (already clamped to >= 0 by the host)
    // into UB once.
    LocalTensor<uint8_t> state_raw = state_buf_.Get<uint8_t>();
    rem_buf_ = state_raw.ReinterpretCast<uint32_t>();
    bias_buf_ = state_raw.ReinterpretCast<float>()[kMaxTagCount];
    for (uint32_t t = 0; t < num_tags_; ++t) {
      bias_buf_.SetValue(t, bias_gm_.GetValue(t));
    }

    uint32_t offset = 0;
    for (uint32_t i = row_begin_; i < row_end_; ++i) {
      const IdT rid = rows_gm_.GetValue(i);
      if (rid < 0 || rid >= static_cast<IdT>(num_total_rows_)) {
        continue;  // defense in depth: drop invalid seed rows
      }
      const IdT off = indptr_gm_.GetValue(static_cast<uint32_t>(rid));
      const IdT end = indptr_gm_.GetValue(static_cast<uint32_t>(rid) + 1);
      const uint32_t deg = static_cast<uint32_t>(end - off);
      // Pick count was computed by the helper kernel and baked into the
      // per-row prefix table by the host; read this row's slice back.
      const uint32_t out_pos = out_start_ + offset;
      const uint32_t row_picks =
          out_starts_p_[i + 1] - out_starts_p_[i];  // host-baked row picks
      if (row_picks == 0) continue;

      uint32_t state = seed_ ^ (i * kGoldenRatioHash + kGoldenRatioOffset);
      if (state == 0) state = kRngFallbackSeed;
      const uint32_t written =
          SampleRow(out_pos, rid, off, deg, row_picks, state);
      offset += written;
    }
  }

 private:
  // Stages one row's tag_offset (T+1 values) into UB via DataCopyPad.
  // Must be followed by a StageTagRowWait before tag_row_ is read.
  __aicore__ inline void StageTagRow(IdT rid) {
    const uint32_t row_base = static_cast<uint32_t>(rid) * (num_tags_ + 1);
    const uint32_t copy_bytes = (num_tags_ + 1) * sizeof(IdT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
    LocalTensor<IdT> t = tag_q_.AllocTensor<IdT>();
    DataCopyPad(t, tag_offset_gm_[row_base], cp, pad);
    tag_q_.EnQue(t);
  }

  __aicore__ inline void StageTagRowWait() { tag_row_ = tag_q_.DeQue<IdT>(); }

  __aicore__ inline void StageTagRowRelease() { tag_q_.FreeTensor(tag_row_); }

  // Loads tag bounds for the staged row into rem_[], clamped into [0, deg].
  // Returns the total positive-weight bucket population (biased nnz).
  // Call StageTagRow + StageTagRowWait first.
  // Clamps a staged tag value into [0, deg] without truncation: negative
  // or beyond-deg int64 values (corrupted input) degrade to clamped
  // bounds instead of wrapping in a uint32 cast (review R3).
  __aicore__ inline uint32_t ClampTag(IdT v, uint32_t deg) {
    if (v < 0) return 0;
    const IdT deg_id = static_cast<IdT>(deg);
    return (v > deg_id) ? deg : static_cast<uint32_t>(v);
  }

  __aicore__ inline uint32_t LoadTagBounds(uint32_t deg) {
    uint32_t nnz = 0;
    for (uint32_t t = 0; t < num_tags_; ++t) {
      uint32_t lo = ClampTag(tag_row_.GetValue(t), deg);
      uint32_t hi = ClampTag(tag_row_.GetValue(t + 1), deg);
      if (hi < lo) hi = lo;  // corrupted non-monotonic row: empty bucket
      rem_buf_.SetValue(t, hi - lo);
      nnz += hi - lo;
    }
    return nnz;
  }

  // Draws one bucket id with probability proportional to rem[t] * bias[t].
  // Returns kNoBucket when the total weight is zero.
  // Converts a uint32 count to float via a signed intermediate
  // (uint<->float casts are forbidden in __aicore__).
  __aicore__ inline float CountToFloat(uint32_t n) {
    int32_t s = static_cast<int32_t>(n);
    return static_cast<float>(s);
  }

  __aicore__ inline int32_t DrawBucket(uint32_t& state) {
    float total = 0.0f;
    for (uint32_t t = 0; t < num_tags_; ++t) {
      const uint32_t r = rem_buf_.GetValue(t);
      if (r > 0) total += CountToFloat(r) * bias_buf_.GetValue(t);
    }
    if (total <= 0.0f) return kNoBucket;
    float pick = RandFloat01(state) * total;
    for (uint32_t t = 0; t < num_tags_; ++t) {
      const uint32_t r = rem_buf_.GetValue(t);
      if (r == 0) continue;
      pick -= CountToFloat(r) * bias_buf_.GetValue(t);
      if (pick < 0.0f) return static_cast<int32_t>(t);
    }
    return num_tags_ - 1;  // float rounding fell off the tail
  }

  // Samples one row. Small-degree rows go through UB (bulk copy in,
  // bucket sampling, bulk copy out); huge-degree rows read GM directly.
  __aicore__ inline uint32_t SampleRow(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks,
      uint32_t& state) {
    if (deg <= window_elems_ && num_picks <= window_elems_) {
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

    StageTagRow(rid);
    StageTagRowWait();
    const uint32_t nnz = LoadTagBounds(deg);
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();

    const bool take_all = select_all_ || (!replace_ && num_picks >= nnz);
    if (take_all) {
      // Deterministic shortcut: every positive-weight edge, bucket order.
      uint32_t j = 0;
      for (uint32_t t = 0; t < num_tags_; ++t) {
        if (bias_buf_.GetValue(t) <= 0.0f) continue;
        uint32_t lo = TagBucketHead(t, deg);
        for (uint32_t k2 = 0; k2 < rem_buf_.GetValue(t); ++k2) {
          WritePickLocal(
              out_r, out_c, out_e, j++, rid, win_idx, win_data, lo + k2, off);
        }
      }
    } else {
      for (uint32_t d = 0; d < num_picks; ++d) {
        const int32_t tb = DrawBucket(state);
        if (tb < 0) break;  // total weight exhausted
        const uint32_t t = static_cast<uint32_t>(tb);
        const uint32_t r = rem_buf_.GetValue(t);
        uint32_t lo = TagBucketHead(t, deg);
        const uint32_t j = RandBelow(state, r);
        WritePickLocal(
            out_r, out_c, out_e, d, rid, win_idx, win_data, lo + j, off);
        if (!replace_) {
          // Fisher-Yates inside the bucket: swap the drawn slot with the
          // bucket tail and shrink; duplicates impossible by construction.
          const uint32_t tail = lo + r - 1;
          if (j != r - 1) {
            const IdT tmp = win_idx.GetValue(lo + j);
            win_idx.SetValue(lo + j, win_idx.GetValue(tail));
            win_idx.SetValue(tail, tmp);
          }
          rem_buf_.SetValue(t, r - 1);
        }
      }
    }

    CopyOutStaged(out_r, out_rows_gm_[out_pos], num_picks);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], num_picks);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], num_picks);

    StageTagRowRelease();
    win_idx_q_.FreeTensor(win_idx);
    if (has_data_) win_data_q_.FreeTensor(win_data);
    return num_picks;
  }

  // Fallback for rows whose degree (or pick count) overflows the UB
  // window: v1-style direct GM scalar path with rejection dedup.
  __aicore__ inline uint32_t SampleRowDirectGm(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks,
      uint32_t& state) {
    StageTagRow(rid);
    StageTagRowWait();
    const uint32_t nnz = LoadTagBounds(deg);

    const bool take_all = select_all_ || (!replace_ && num_picks >= nnz);
    if (take_all) {
      uint32_t j = 0;
      for (uint32_t t = 0; t < num_tags_; ++t) {
        if (bias_buf_.GetValue(t) <= 0.0f) continue;
        const uint32_t lo = TagBucketHead(t, deg);
        for (uint32_t k2 = 0; k2 < rem_buf_.GetValue(t); ++k2) {
          WritePickGm(out_pos + j, rid, off + static_cast<IdT>(lo + k2));
          ++j;
        }
      }
      StageTagRowRelease();
      return j;
    }

    for (uint32_t d = 0; d < num_picks; ++d) {
      const int32_t tb = DrawBucket(state);
      if (tb < 0) break;
      const uint32_t t = static_cast<uint32_t>(tb);
      const uint32_t r = rem_buf_.GetValue(t);
      const uint32_t lo = TagBucketHead(t, deg);
      uint32_t j = RandBelow(state, r);
      if (!replace_) {
        // Rejection sampling: redraw while the drawn edge was already
        // picked this row (CPU uses a hash set with the same semantics).
        // Dedup reads back the eid column already written to GM for this
        // row — no capacity-capped staging, so arbitrarily large pick
        // counts stay in bounds (review R1: picked_local could overflow
        // window_elems_).
        uint32_t tries = 0;
        bool dup = true;
        while (dup && tries < r) {
          dup = false;
          for (uint32_t k2 = 0; k2 < d; ++k2) {
            if (out_idxs_gm_.GetValue(out_pos + k2) ==
                off + static_cast<IdT>(lo + j)) {
              dup = true;
              break;
            }
          }
          if (dup) {
            j = RandBelow(state, r);
            ++tries;
          }
        }
        if (dup) {
          // Fall back to sequential scan from the bucket head (the
          // random search failed; correctness first).
          for (uint32_t k2 = 0; k2 < r; ++k2) {
            bool seen = false;
            for (uint32_t k3 = 0; k3 < d; ++k3) {
              if (out_idxs_gm_.GetValue(out_pos + k3) ==
                  off + static_cast<IdT>(lo + k2)) {
                seen = true;
                break;
              }
            }
            if (!seen) {
              j = k2;
              break;
            }
          }
        }
      }
      WritePickGm(out_pos + d, rid, off + static_cast<IdT>(lo + j));
      if (!replace_) rem_buf_.SetValue(t, r - 1);
    }
    StageTagRowRelease();
    return num_picks;
  }

  // Clamped bucket head (local offset within the row) for tag t.
  // Clamped bucket head (local offset within the staged row) for tag t.
  __aicore__ inline uint32_t TagBucketHead(uint32_t t, uint32_t deg) {
    return ClampTag(tag_row_.GetValue(t), deg);
  }

  __aicore__ inline void WritePickLocal(
      LocalTensor<IdT>& out_r, LocalTensor<IdT>& out_c, LocalTensor<IdT>& out_e,
      uint32_t j, IdT rid, LocalTensor<IdT>& win_idx,
      LocalTensor<IdT>& win_data, uint32_t local, IdT off) {
    out_r.SetValue(j, rid);
    out_c.SetValue(j, win_idx.GetValue(local));
    out_e.SetValue(
        j,
        has_data_ ? win_data.GetValue(local) : static_cast<IdT>(off + local));
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
  static constexpr int32_t kNoBucket = -1;

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
  GlobalTensor<IdT> tag_offset_gm_;
  GlobalTensor<float> bias_gm_;
  GlobalTensor<IdT> out_rows_gm_, out_cols_gm_, out_idxs_gm_;
  TQue<TPosition::VECIN, kQueueDepth> win_idx_q_, win_data_q_;
  TQue<TPosition::VECOUT, kQueueDepth> out_q_;
  TBuf<TPosition::VECCALC> pick_buf_;
  TBuf<TPosition::VECCALC> out_r_buf_, out_c_buf_, out_e_buf_;
  TBuf<TPosition::VECCALC> state_buf_;
  TQue<TPosition::VECIN, kQueueDepth> tag_q_;
  LocalTensor<uint32_t> rem_buf_;
  LocalTensor<float> bias_buf_;
  // Staged copy of the current row's tag_offset (T+1 values, in UB).
  LocalTensor<IdT> tag_row_;
  uint32_t num_rows_ = 0, replace_ = 0, has_data_ = 0;
  uint32_t seed_ = 0, select_all_ = 0, num_total_rows_ = 0, num_tags_ = 0;
  uint32_t out_start_ = 0;
  uint32_t row_begin_ = 0, row_end_ = 0, window_elems_ = 0;
  uint32_t block_idx_ = 0;
  const __gm__ uint32_t* out_starts_p_ = nullptr;
};

extern "C" __global__ __aicore__ void csr_row_wise_sampling_biased_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR tag_offset, GM_ADDR bias, GM_ADDR out_rows, GM_ADDR out_cols,
    GM_ADDR out_idxs, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingBiased<int32_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, tag_offset, bias, out_rows, out_cols,
      out_idxs, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_biased_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR tag_offset, GM_ADDR bias, GM_ADDR out_rows, GM_ADDR out_cols,
    GM_ADDR out_idxs, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingBiased<int64_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, tag_offset, bias, out_rows, out_cols,
      out_idxs, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
csr_row_wise_sampling_biased_num_picks_int32(
    GM_ADDR rows, GM_ADDR tag_offset, GM_ADDR bias, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingBiasedNumPicks<int32_t> op;
  TPipe pipe;
  op.Init(rows, tag_offset, bias, out, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
csr_row_wise_sampling_biased_num_picks_int64(
    GM_ADDR rows, GM_ADDR tag_offset, GM_ADDR bias, GM_ADDR out,
    GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingBiasedNumPicks<int64_t> op;
  TPipe pipe;
  op.Init(rows, tag_offset, bias, out, tiling_ptr, &pipe);
  op.Process();
}
