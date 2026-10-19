/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_fused_kernel.cpp
 * @brief Multi-core AIV kernel for uniform CSR row-wise fused sampling
 *        (sampling + seed-node mapping) on Ascend NPU.
 *
 * Design (2026-08-22, twin of csr_row_wise_sampling_uniform_kernel.cpp per
 * ADR-0011 — cloned v2 skeleton, fused from birth with batch writes):
 * - Vector cores (KERNEL_TYPE_AIV_ONLY), count queried at runtime; the
 *   host computes nnz-balanced row_split / out_starts GM tables (spmm
 *   precedent). Blocks write disjoint output ranges with no cross-block
 *   reduction.
 * - Fused extras vs the plain uniform kernel:
 *   * out_ptr carries the block-local CSR indptr (same as plain).
 *   * out_rows carries the ROW POSITION i (index into rows), not the raw
 *     rid — the CPU reference (CSRRowWisePickFused) semantics.
 *   * seed_pairs carries interleaved (rid, row_position) pairs, one per
 *     valid row, when map_seed_nodes=1. The host scatters them into
 *     seed_mapping (ADR-0014: AscendC has no GM scatter primitive, so the
 *     kernel only produces a compact contiguous array — every kernel
 *     output is a contiguous batch write, honoring the SetValue ban of
 *     ADR-0013).
 * - Batch writes everywhere (ADR-0013): GlobalTensor::SetValue/GetValue
 *   are banned in production kernels. All outputs are staged in UB
 *   (LocalTensor scalar ops are fine) and flushed with DataCopyPad.
 *   - Per row, the CSR window [off, off+deg) of indices (and data) is
 *     bulk-copied GM -> UB via VECIN queues; sampled picks are staged in
 *     VECCALC buffers and copied out through the VECOUT queue one
 *     complete Alloc/EnQue/DeQue/Free cycle at a time.
 *   - out_ptr (a running prefix sum) and the seed pairs accumulate in UB
 *     across the block's row loop and flush in meta-sized chunks.
 *   - The rows array and indptr segments are prefetched into UB in
 *     meta-sized blocks and read from LocalTensor.
 *   - Huge-degree rows (deg > window) are processed in window-sized
 *     segments. No-replace sampling uses a prefix-window reservoir: every
 *     element k (0-based over the whole row) is offered to the reservoir
 *     with algorithm-R odds n/(k+1) against the RUNNING state, so the
 *     outcome is exactly algorithm R over the full row regardless of the
 *     window segmentation. Emission then revisits windows in order, so
 *     edge order within a huge row follows window order rather than
 *     reservoir slot order (set-equal to CPU; degenerate compare uses
 *     sorted edge sets).
 * - Out-of-range row ids are dropped as empty rows (defense in depth,
 *   matching the CPU path where COOSliceRows discards invalid seeds).
 */

#include "csr_row_wise_sampling_uniform_fused_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t Xorshift32Fused(uint32_t& x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__aicore__ inline uint32_t RandBelowFused(uint32_t& state, uint32_t n) {
  if (n == 0) return 0;
  uint32_t r = Xorshift32Fused(state);
  return static_cast<uint32_t>((static_cast<uint64_t>(r) * n) >> 32);
}

}  // namespace

template <typename IdT>
class KernelCsrRowWiseSamplingUniformFused {
 public:
  __aicore__ inline KernelCsrRowWiseSamplingUniformFused() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
      GM_ADDR seed_pairs, GM_ADDR row_split, GM_ADDR out_starts,
      GM_ADDR tiling_ptr, TPipe* pipe) {

    // Struct-pointer tiling access (spmm pattern): field loads through a
    // typed __gm__ pointer are consistently visible across blocks.
    const __gm__ CsrRowWiseSamplingUniformFusedTiling* tiling =
        (const __gm__ CsrRowWiseSamplingUniformFusedTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    num_samples_ = tiling->num_samples;
    replace_ = tiling->replace;
    has_data_ = tiling->has_data;
    seed_ = tiling->seed;
    select_all_ = tiling->select_all;
    num_total_rows_ = tiling->num_total_rows;
    map_seed_nodes_ = tiling->map_seed_nodes;
    const uint32_t ub_available = tiling->ub_available;

    indptr_p_ = (const __gm__ IdT*)indptr;
    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_total_rows_ + 1);
    indices_gm_.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    rows_p_ = (const __gm__ IdT*)rows;
    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, num_rows_);
    out_ptr_gm_.SetGlobalBuffer((__gm__ IdT*)out_ptr, num_rows_ + 1);
    out_rows_gm_.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm_.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm_.SetGlobalBuffer((__gm__ IdT*)out_idxs);
    // seed_pairs is our own contiguous output (ADR-0014); the host always
    // provides a real buffer (a 1-element dummy when map_seed_nodes=0),
    // so the binding is unconditionally safe.
    seed_pairs_gm_.SetGlobalBuffer((__gm__ IdT*)seed_pairs);

    const uint32_t block_idx = AscendC::GetBlockIdx();
    block_idx_ = block_idx;
    (void)AscendC::GetBlockNum();  // tables sized by the same value on host

    const __gm__ uint32_t* row_split_p = (const __gm__ uint32_t*)row_split;
    const __gm__ uint32_t* out_starts_p = (const __gm__ uint32_t*)out_starts;
    row_begin_ = row_split_p[block_idx];
    row_end_ = row_split_p[block_idx + 1];
    out_start_ = out_starts_p[block_idx];

    // UB budget split: see the shared window equation in the tiling
    // header (single source of truth for kernel and host).
    window_elems_ = (ub_available - kMetaUbReserveFused) /
                    kWindowInstancesFused / sizeof(IdT);
    meta_rows_ = kMetaRowsFused;

    pipe->InitBuffer(win_idx_q_, kQueueDepthFused, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(
        win_data_q_, kQueueDepthFused, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(pick_buf_, window_elems_ * sizeof(uint32_t));
    pipe->InitBuffer(out_r_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_c_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_e_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_q_, kQueueDepthFused, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_ptr_buf_, (meta_rows_ + 1) * sizeof(IdT));
    pipe->InitBuffer(seed_pairs_buf_, 2 * meta_rows_ * sizeof(IdT));
  }

  __aicore__ inline void Process() {
    if (row_begin_ >= row_end_) return;  // idle block: owns zero rows

    uint32_t offset = 0;
    uint32_t out_ptr_staged = 0;
    uint32_t pairs_staged = 0;
    LocalTensor<IdT> out_ptr_local = out_ptr_buf_.Get<IdT>();
    LocalTensor<IdT> pairs_local = seed_pairs_buf_.Get<IdT>();

    for (uint32_t i = row_begin_; i < row_end_; ++i) {
      out_ptr_local.SetValue(
          out_ptr_staged, static_cast<IdT>(out_start_ + offset));
      ++out_ptr_staged;

      const IdT rid = ReadRows(i);
      const bool rid_valid =
          rid >= 0 && rid < static_cast<IdT>(num_total_rows_);
      IdT off = 0;
      uint32_t deg = 0;
      if (rid_valid) ReadIndptr(rid, &off, &deg);
      if (map_seed_nodes_) {
        // One pair per row (invalid rids become sentinel pairs the host
        // skips): pairs are a strict function of the row index, so each
        // block writes its own contiguous slice [row_begin_, row_end_)
        // with no cross-block offset bookkeeping.
        pairs_local.SetValue(2 * pairs_staged, rid);
        pairs_local.SetValue(2 * pairs_staged + 1, static_cast<IdT>(i));
        ++pairs_staged;
      }

      uint32_t num_picks = 0;
      if (rid_valid && deg > 0) {
        num_picks = select_all_ ? deg
                    : replace_  ? num_samples_
                                : (deg < num_samples_ ? deg : num_samples_);
      }
      if (num_picks > 0) {
        uint32_t state =
            seed_ ^ (i * kGoldenRatioHashFused + kGoldenRatioOffsetFused);
        if (state == 0) state = kRngFallbackSeedFused;
        offset +=
            SampleRow(out_start_ + offset, i, rid, off, deg, num_picks, state);
      }
      // Flush meta staging before overflow (one spare out_ptr slot for
      // the final prefix entry appended after the loop).
      if (out_ptr_staged >= meta_rows_) {
        FlushOutPtr(out_ptr_local, out_ptr_staged);
        out_ptr_staged = 0;
      }
      if (pairs_staged >= meta_rows_) {
        FlushSeedPairs(pairs_local, pairs_staged);
        pairs_staged = 0;
      }
    }

    // Final prefix entry: total picked by this block.
    out_ptr_local.SetValue(
        out_ptr_staged, static_cast<IdT>(out_start_ + offset));
    ++out_ptr_staged;
    FlushOutPtr(out_ptr_local, out_ptr_staged);
    if (pairs_staged > 0) FlushSeedPairs(pairs_local, pairs_staged);
  }

 private:
  // Processes one row: stages its out_ptr prefix entry and seed pair,
  // samples its edges, returns the updated output offset.
  __aicore__ inline uint32_t ProcessRow(
      uint32_t i, uint32_t offset, LocalTensor<IdT>& out_ptr_local,
      uint32_t* out_ptr_staged, LocalTensor<IdT>& pairs_local,
      uint32_t* pairs_staged) {
    out_ptr_local.SetValue(
        *out_ptr_staged, static_cast<IdT>(out_start_ + offset));
    ++*out_ptr_staged;

    const IdT rid = ReadRows(i);
    const bool rid_valid = rid >= 0 && rid < static_cast<IdT>(num_total_rows_);
    IdT off = 0;
    uint32_t deg = 0;
    if (rid_valid) ReadIndptr(rid, &off, &deg);
    if (map_seed_nodes_) {
      // One pair per row (invalid rids become sentinel pairs the host
      // skips): pairs are a strict function of the row index, so each
      // block writes its own contiguous slice [row_begin_, row_end_)
      // with no cross-block offset bookkeeping.
      pairs_local.SetValue(2 * *pairs_staged, rid);
      pairs_local.SetValue(2 * *pairs_staged + 1, static_cast<IdT>(i));
      ++*pairs_staged;
    }

    uint32_t num_picks = 0;
    if (rid_valid && deg > 0) {
      num_picks = select_all_ ? deg
                  : replace_  ? num_samples_
                              : (deg < num_samples_ ? deg : num_samples_);
    }
    if (num_picks > 0) {
      uint32_t state =
          seed_ ^ (i * kGoldenRatioHashFused + kGoldenRatioOffsetFused);
      if (state == 0) state = kRngFallbackSeedFused;
      offset +=
          SampleRow(out_start_ + offset, i, rid, off, deg, num_picks, state);
    }
    return offset;
  }

  // Reads rows[i] through the typed __gm__ pointer captured in Init
  // (the rowSplit table pattern from the plain uniform kernel): the
  // launch was preceded by a host-side stream sync, so direct scalar
  // loads are consistently visible. A UB prefetch here would need queue
  // discipline for the async copy to land before the read (the async
  // prefetch in the first draft raced the read — stale UB garbage).
  __aicore__ inline IdT ReadRows(uint32_t i) { return rows_p_[i]; }

  // Reads indptr[rid] and indptr[rid+1] the same way.
  __aicore__ inline void ReadIndptr(IdT rid, IdT* off, uint32_t* deg) {
    const IdT start = indptr_p_[rid];
    const IdT end = indptr_p_[rid + 1];
    *off = start;
    *deg = static_cast<uint32_t>(end - start);
  }

  // Samples one row. Rows within the window go through UB; huge rows are
  // processed in segments.
  __aicore__ inline uint32_t SampleRow(
      uint32_t out_pos, uint32_t row_pos, IdT rid, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state) {
    if (deg <= window_elems_) {
      return SampleRowThroughUb(
          out_pos, row_pos, rid, off, deg, num_picks, state);
    }
    return SampleRowSegmented(
        out_pos, row_pos, rid, off, deg, num_picks, state);
  }

  __aicore__ inline uint32_t SampleRowThroughUb(
      uint32_t out_pos, uint32_t row_pos, IdT rid, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state) {
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

    // Pick local indices into scratch (LocalTensor scalar ops only).
    LocalTensor<uint32_t> picks = pick_buf_.Get<uint32_t>();
    if (select_all_ || (!replace_ && num_picks == deg)) {
      for (uint32_t j = 0; j < num_picks; ++j) picks.SetValue(j, j);
    } else if (replace_) {
      for (uint32_t j = 0; j < num_picks; ++j) {
        picks.SetValue(j, RandBelowFused(state, deg));
      }
    } else {
      // Algorithm R reservoir over local indices.
      for (uint32_t j = 0; j < num_picks; ++j) picks.SetValue(j, j);
      for (uint32_t i2 = num_picks; i2 < deg; ++i2) {
        uint32_t j = RandBelowFused(state, i2 + 1);
        if (j < num_picks) picks.SetValue(j, i2);
      }
    }

    // Materialize the three output arrays into VECCALC staging buffers.
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    for (uint32_t j = 0; j < num_picks; ++j) {
      uint32_t local = picks.GetValue(j);
      out_r.SetValue(j, static_cast<IdT>(row_pos));  // row POSITION (fused)
      out_c.SetValue(j, win_idx.GetValue(local));
      out_e.SetValue(
          j,
          has_data_ ? win_data.GetValue(local) : static_cast<IdT>(off + local));
    }

    CopyOutStaged(out_r, out_rows_gm_[out_pos], num_picks);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], num_picks);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], num_picks);

    win_idx_q_.FreeTensor(win_idx);
    if (has_data_) win_data_q_.FreeTensor(win_data);
    return num_picks;
  }

  // Huge-degree rows: stream the row through window-sized segments.
  //  - select-all / full-degree: emit every edge in window order.
  //  - replace: draw all pick positions first (pure RNG, no data needed),
  //    then each window emits the picks that land inside it — one pass
  //    over picks per window keeps this linear in picks*windows/deg.
  //  - no-replace: prefix-window reservoir. Window 0 seeds the reservoir
  //    with [0, num_picks) and offers its remainder with algorithm-R
  //    odds; each later window offers its elements with odds n/(k+1)
  //    against the RUNNING count k — identical to algorithm R over the
  //    full row, independent of segmentation. Emission then revisits
  //    windows in order.
  __aicore__ inline uint32_t SampleRowSegmented(
      uint32_t out_pos, uint32_t row_pos, IdT rid, IdT off, uint32_t deg,
      uint32_t num_picks, uint32_t& state) {
    if (select_all_ || (!replace_ && num_picks == deg)) {
      uint32_t emitted = 0;
      while (emitted < deg) {
        const uint32_t chunk =
            (deg - emitted) < window_elems_ ? (deg - emitted) : window_elems_;
        emitted += EmitWindow(
            out_pos, row_pos, off, emitted, chunk, nullptr, 0, false);
        out_pos += chunk;
      }
      return deg;
    }
    LocalTensor<uint32_t> picks = pick_buf_.Get<uint32_t>();
    // The pick cursor is per-row state: picks slots hold UNORDERED
    // indices (reservoir slots get replaced by arbitrary later indices),
    // so a monotone cursor skips picks that live in earlier windows —
    // scan the full picks array per window instead, and reset for each
    // row so the previous row's tail cannot leak in.
    if (replace_) {
      for (uint32_t j = 0; j < num_picks; ++j) {
        picks.SetValue(j, RandBelowFused(state, deg));
      }
      // Sort-free partitioned emission: one pass per window over the
      // picks array (each pick lands in exactly one window).
      uint32_t emitted = 0;
      for (uint32_t base = 0; base < deg && emitted < num_picks;
           base += window_elems_) {
        const uint32_t chunk =
            (deg - base) < window_elems_ ? (deg - base) : window_elems_;
        const uint32_t got = EmitWindow(
            out_pos, row_pos, off, base, chunk, &picks, num_picks, true);
        emitted += got;
        out_pos += got;
      }
      return num_picks;
    }
    // No replacement: prefix-window reservoir.
    for (uint32_t j = 0; j < num_picks; ++j) picks.SetValue(j, j);
    uint32_t consumed = num_picks;
    for (uint32_t base = 0; base < deg; base += window_elems_) {
      const uint32_t chunk =
          (deg - base) < window_elems_ ? (deg - base) : window_elems_;
      if (consumed < base + chunk) {
        // Elements [consumed, base+chunk) have not been offered yet.
        for (uint32_t k = consumed; k < base + chunk; ++k) {
          const uint32_t j = RandBelowFused(state, k + 1);
          if (j < num_picks) picks.SetValue(j, k);
        }
        consumed = base + chunk;
      }
    }
    // Materialize winners window by window (window-order emission).
    uint32_t emitted = 0;
    for (uint32_t base = 0; base < deg && emitted < num_picks;
         base += window_elems_) {
      const uint32_t chunk =
          (deg - base) < window_elems_ ? (deg - base) : window_elems_;
      const uint32_t got = EmitWindow(
          out_pos, row_pos, off, base, chunk, &picks, num_picks, false);
      emitted += got;
      out_pos += got;
    }
    return num_picks;
  }

  // Loads window [base, base+chunk) of the row into UB and emits edges:
  //  - picks == nullptr: emit every element of the window (select-all /
  //    full-degree path); returns chunk.
  //  - picks != nullptr && replace_mode: picks[p..num_picks) hold absolute
  //    local indices; emit those inside this window (a pick lands in
  //    exactly one window; duplicates fine). Returns the emitted count.
  //  - picks != nullptr && !replace_mode: same emission rule over the
  //    reservoir winners (all num_picks indices are distinct).
  // For pick emission the window keeps a cursor so consecutive windows
  // resume the scan where the previous left off (linear overall).
  __aicore__ inline uint32_t EmitWindow(
      uint32_t out_pos, uint32_t row_pos, IdT off, uint32_t base,
      uint32_t chunk, LocalTensor<uint32_t>* picks, uint32_t num_picks,
      bool replace_mode) {
    LocalTensor<IdT> win_idx = win_idx_q_.AllocTensor<IdT>();
    LocalTensor<IdT> win_data;
    const uint32_t copy_bytes = chunk * sizeof(IdT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
    DataCopyPad(win_idx, indices_gm_[off + base], cp, pad);
    win_idx_q_.EnQue(win_idx);
    win_idx = win_idx_q_.DeQue<IdT>();
    if (has_data_) {
      win_data = win_data_q_.AllocTensor<IdT>();
      DataCopyPad(win_data, data_gm_[off + base], cp, pad);
      win_data_q_.EnQue(win_data);
      win_data = win_data_q_.DeQue<IdT>();
    }
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    uint32_t emitted = 0;
    if (picks == nullptr) {
      for (uint32_t j = 0; j < chunk; ++j) {
        out_r.SetValue(j, static_cast<IdT>(row_pos));
        out_c.SetValue(j, win_idx.GetValue(j));
        out_e.SetValue(
            j, has_data_ ? win_data.GetValue(j)
                         : static_cast<IdT>(off + base + j));
      }
      emitted = chunk;
    } else {
      // Full scan per window: picks slots are unordered (reservoir
      // replacement puts later indices into early slots), so a cursor
      // monotone in p is not monotone in window — cursor-based resumes
      // DROP picks (verified counterexample in the review). The extra
      // scan cost is bounded by picks*windows, same order as emission.
      for (uint32_t p = 0; p < num_picks; ++p) {
        const uint32_t idx = (*picks).GetValue(p);
        if (idx >= base && idx < base + chunk) {
          out_r.SetValue(emitted, static_cast<IdT>(row_pos));
          out_c.SetValue(emitted, win_idx.GetValue(idx - base));
          out_e.SetValue(
              emitted, has_data_ ? win_data.GetValue(idx - base)
                                 : static_cast<IdT>(off + idx));
          ++emitted;
        }
      }
    }
    if (emitted > 0) {
      CopyOutStaged(out_r, out_rows_gm_[out_pos], emitted);
      CopyOutStaged(out_c, out_cols_gm_[out_pos], emitted);
      CopyOutStaged(out_e, out_idxs_gm_[out_pos], emitted);
    }
    win_idx_q_.FreeTensor(win_idx);
    if (has_data_) win_data_q_.FreeTensor(win_data);
    return emitted;
  }

  // Writes staged out_ptr entries to GM (contiguous run starting at the
  // block's row_begin_ + already-flushed count).
  __aicore__ inline void FlushOutPtr(
      LocalTensor<IdT>& staging, uint32_t count) {
    if (count == 0) return;
    const uint32_t copy_bytes = count * sizeof(IdT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPad(out_ptr_gm_[row_begin_ + out_ptr_flushed_], staging, cp);
    // DataCopyPad is asynchronous: drain MTE3 before the next row
    // overwrites the staging buffer it is still reading from.
    AscendC::PipeBarrier<PIPE_MTE3>();
    out_ptr_flushed_ += count;
  }

  // Writes staged interleaved (rid, pos) pairs to the compact output
  // array. The destination is the block's own row range — pairs are a
  // function of the row index, so no cross-block offset is needed.
  __aicore__ inline void FlushSeedPairs(
      LocalTensor<IdT>& staging, uint32_t count) {
    if (count == 0) return;
    const uint32_t copy_bytes = 2 * count * sizeof(IdT);
    DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
    DataCopyPad(seed_pairs_gm_[2 * (row_begin_ + pairs_flushed_)], staging, cp);
    // Same drain requirement as FlushOutPtr above.
    AscendC::PipeBarrier<PIPE_MTE3>();
    pairs_flushed_ += count;
  }

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

  const __gm__ IdT* indptr_p_ = nullptr;
  const __gm__ IdT* rows_p_ = nullptr;
  GlobalTensor<IdT> indptr_gm_, indices_gm_, data_gm_, rows_gm_;
  GlobalTensor<IdT> out_ptr_gm_, out_rows_gm_, out_cols_gm_, out_idxs_gm_;
  GlobalTensor<IdT> seed_pairs_gm_;
  TQue<TPosition::VECIN, kQueueDepthFused> win_idx_q_, win_data_q_;
  TQue<TPosition::VECOUT, kQueueDepthFused> out_q_;
  TBuf<TPosition::VECCALC> pick_buf_;
  TBuf<TPosition::VECCALC> out_r_buf_, out_c_buf_, out_e_buf_;
  TBuf<TPosition::VECCALC> out_ptr_buf_, seed_pairs_buf_;
  uint32_t num_rows_ = 0, num_samples_ = 0, replace_ = 0, has_data_ = 0;
  uint32_t seed_ = 0, select_all_ = 0, num_total_rows_ = 0, out_start_ = 0;
  uint32_t row_begin_ = 0, row_end_ = 0, window_elems_ = 0;
  uint32_t block_idx_ = 0, map_seed_nodes_ = 0, meta_rows_ = 0;
  uint32_t out_ptr_flushed_ = 0, pairs_flushed_ = 0;
};

extern "C" __global__ __aicore__ void csr_row_wise_sampling_uniform_fused_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR seed_pairs, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingUniformFused<int32_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, out_ptr, out_rows, out_cols, out_idxs,
      seed_pairs, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_uniform_fused_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR seed_pairs, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelCsrRowWiseSamplingUniformFused<int64_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, out_ptr, out_rows, out_cols, out_idxs,
      seed_pairs, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}
