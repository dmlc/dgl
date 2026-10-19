/**
 * Copyright (c) 2024 by Contributors
 * @file coo2csr_kernel.cpp
 * @brief Multi-core AIV kernels for COO -> CSR conversion (counting sort).
 *
 * Two independent kernel launches with no inter-core synchronization
 * (ADR-0005); the host runs the indptr exclusive scan between them:
 *
 *   Pass 1 (coo_to_csr_count_*): each block owns a disjoint row range
 *     [row_split[b], row_split[b+1]). It streams the row[] edge array
 *     through UB in chunks (DataCopyPad via a VECIN queue, full
 *     Alloc->EnQue->DeQue->Free cycles), counts edges whose row falls in
 *     its range into a UB-resident histogram, and reduces the min/max row
 *     id seen. The histogram is written back with one DMA copy into the
 *     block's private slice of the counts workspace; min/max are DMA'd
 *     into a per-block slab of a separate reductions buffer (never two
 *     blocks scalar-writing one cache line).
 *
 *   Pass 2 (coo_to_csr_scatter_*): same row-range partition. Blocks
 *     stream row/col/data chunk-wise through UB in global edge order and,
 *     for each edge whose row belongs to the block, write
 *       indices[pos] = col[i];  data[pos] = has_data ? data[i] : i
 *     at pos = indptr[row] + cursor[row]++ with cursors initialized from
 *     the scanned indptr. Picked outputs are staged in UB and flushed
 *     with DMA when a chunk completes (a chunk's picks are contiguous in
 *     the output only within a row, so the flush is per-row-run; runs are
 *     emitted with one DataCopyPad each). Processing edges in global
 *     order makes the scatter stable: same-row edges keep their original
 *     relative order. Blocks only write output slots of their own rows,
 *     so writes are disjoint across blocks.
 *
 * All bulk GM traffic is DMA (DataCopyPad); scalar SetValue/GetValue on
 * GlobalTensor is a production blacklist API and carries per-core DCache
 * write-back coherence risk (ADR-0007). Control tables (tiling,
 * row_split) are read through typed __gm__ pointers, matching the spmm /
 * csr_row_wise_sampling_uniform precedents.
 *
 * Large row counts: when a block's row range cannot fit its histogram in
 * UB, the host launches the pass pair multiple times in row bands (each
 * band re-streams the edge array; see harness DESIGN docs/coo2csr §2.4).
 */

#include "coo2csr_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

// ---------------------------------------------------------------------------
// Pass 1: per-block row histogram + min/max reduction.
// ---------------------------------------------------------------------------
template <typename IdT>
class KernelCooToCsrCount {
 public:
  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR counts, GM_ADDR reduces, GM_ADDR row_split,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ CooToCsrTiling* t = (const __gm__ CooToCsrTiling*)tiling_ptr;
    nnz_ = t->nnz;
    num_rows_ = t->num_rows;
    const uint32_t ub_available = t->ub_available;

    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, nnz_);
    counts_gm_.SetGlobalBuffer((__gm__ uint32_t*)counts, num_rows_);
    reduces_gm_.SetGlobalBuffer(
        (__gm__ uint32_t*)reduces, GetBlockNum() * kReduceWordsPerBlock);

    const uint32_t block_idx = GetBlockIdx();
    const __gm__ uint32_t* split = (const __gm__ uint32_t*)row_split;
    row_begin_ = split[block_idx];
    row_end_ = split[block_idx + 1];
    if (row_begin_ >= row_end_) {
      // Idle block: skip all buffer setup; Process only writes sentinel
      // reductions through reduce_buf_, so allocate just that one line.
      pipe->InitBuffer(reduce_buf_, kCacheLineBytes);
      initialized_ = false;
      return;
    }

    // UB budget: double-buffered stream chunk (2 instances) + histogram
    // (1 instance). The histogram needs the whole range; the chunk gets
    // what remains. Host-side banding guarantees
    // (row_end_ - row_begin_) * 4 + 2 * sizeof(IdT) <= ub_available.
    const uint32_t hist_bytes = (row_end_ - row_begin_) * sizeof(uint32_t);
    uint32_t chunk_bytes = (ub_available - hist_bytes) / 2;
    chunk_bytes &= ~static_cast<uint32_t>(sizeof(IdT) - 1);
    chunk_elems_ = chunk_bytes / sizeof(IdT);
    if (chunk_elems_ == 0) chunk_elems_ = 1;

    pipe->InitBuffer(stream_q_, 2, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(hist_buf_, hist_bytes);
    pipe->InitBuffer(
        reduce_buf_, kCacheLineBytes);  // one aligned line for min/max
    hist_ = hist_buf_.Get<uint32_t>();
    for (uint32_t r = 0; r < row_end_ - row_begin_; ++r) hist_.SetValue(r, 0);
    min_ = row_end_;
    max_ = row_begin_;
    initialized_ = true;
  }

  __aicore__ inline void Process() {
    if (row_begin_ >= row_end_) {
      // Empty range: emit sentinel reductions (min > max signals "no
      // edges in range"); the host skips such blocks.
      WriteReduce(kReduceEmpty, kReduceEmpty);
      return;
    }
    const IdT begin = static_cast<IdT>(row_begin_);
    const IdT end = static_cast<IdT>(row_end_);
    // Stream row[] through UB in chunks (full Alloc->EnQue->DeQue->Free
    // cycle per chunk; queue discipline) and count into the histogram.
    DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
    for (int64_t base = 0; base < nnz_; base += chunk_elems_) {
      const uint32_t cnt = static_cast<uint32_t>(
          nnz_ - base < chunk_elems_ ? nnz_ - base : chunk_elems_);
      LocalTensor<IdT> buf = stream_q_.AllocTensor<IdT>();
      DataCopyExtParams cp{
          1, static_cast<uint32_t>(cnt * sizeof(IdT)), 0, 0, 0};
      DataCopyPad(buf, rows_gm_[base], cp, pad);
      stream_q_.EnQue(buf);
      buf = stream_q_.DeQue<IdT>();
      for (uint32_t i = 0; i < cnt; ++i) {
        const IdT r = buf.GetValue(i);
        if (r >= begin && r < end) {
          const uint32_t slot = static_cast<uint32_t>(r - begin);
          hist_.SetValue(slot, hist_.GetValue(slot) + 1);
          if (r < min_) min_ = r;
          if (r > max_) max_ = r;
        }
      }
      stream_q_.FreeTensor(buf);
    }

    // DMA the histogram into this block's private counts slice.
    DataCopyExtParams cp{
        1, static_cast<uint32_t>((row_end_ - row_begin_) * sizeof(uint32_t)), 0,
        0, 0};
    DataCopyPad(counts_gm_[row_begin_], hist_, cp);
    // A range with no edges keeps the sentinel pair (min > max).
    WriteReduce(
        min_ <= max_ ? static_cast<uint32_t>(min_) : kReduceEmpty,
        min_ <= max_ ? static_cast<uint32_t>(max_) : kReduceEmpty);
  }

 private:
  // Min/max go out through a small staging buffer with a DMA copy: the
  // per-block slab is cache-line aligned so no two blocks touch the same
  // line through the scalar path (staged here as a vector write).
  __aicore__ inline void WriteReduce(uint32_t lo, uint32_t hi) {
    LocalTensor<uint32_t> red = reduce_buf_.Get<uint32_t>();
    red.SetValue(0, lo);
    red.SetValue(1, hi);
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(kReduceWordsPerBlock * sizeof(uint32_t)), 0, 0,
        0};
    DataCopyPad(reduces_gm_[GetBlockIdx() * kReduceWordsPerBlock], red, cp);
  }

  GlobalTensor<IdT> rows_gm_;
  GlobalTensor<uint32_t> counts_gm_, reduces_gm_;
  TQue<TPosition::VECIN, 2> stream_q_;
  TBuf<TPosition::VECCALC> hist_buf_;
  TBuf<TPosition::VECCALC> reduce_buf_;
  LocalTensor<uint32_t> hist_;
  int64_t nnz_ = 0;
  uint32_t num_rows_ = 0;
  uint32_t row_begin_ = 0, row_end_ = 0, chunk_elems_ = 0;
  IdT min_ = 0, max_ = 0;
  bool initialized_ = false;
};

// ---------------------------------------------------------------------------
// Pass 2: stable scatter of col/data into CSR order.
// ---------------------------------------------------------------------------
template <typename IdT>
class KernelCooToCsrScatter {
 public:
  __aicore__ inline void Init(
      GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR indptr,
      GM_ADDR out_indices, GM_ADDR out_data, GM_ADDR row_split,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ CooToCsrTiling* t = (const __gm__ CooToCsrTiling*)tiling_ptr;
    nnz_ = t->nnz;
    num_rows_ = t->num_rows;
    has_data_ = t->has_data;
    const uint32_t ub_available = t->ub_available;

    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, nnz_);
    cols_gm_.SetGlobalBuffer((__gm__ IdT*)cols, nnz_);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data, nnz_);
    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_rows_ + 1);
    out_indices_gm_.SetGlobalBuffer((__gm__ IdT*)out_indices, nnz_);
    out_data_gm_.SetGlobalBuffer((__gm__ IdT*)out_data, nnz_);

    const uint32_t block_idx = GetBlockIdx();
    const __gm__ uint32_t* split = (const __gm__ uint32_t*)row_split;
    row_begin_ = split[block_idx];
    row_end_ = split[block_idx + 1];
    if (row_begin_ >= row_end_) return;  // idle block: no buffers needed

    // UB budget: three double-buffered stream chunks (rows, cols, data
    // = 6 instances), one histogram-sized cursor buffer, one staging
    // buffer for output runs. The cursor needs the whole row range
    // (same size as the Pass 1 histogram, guaranteed by banding).
    const uint32_t range = row_end_ - row_begin_;
    const uint32_t cursor_bytes = range * sizeof(IdT);
    // Guard: host banding keeps the cursor comfortably below the UB
    // budget; clamp defensively against unsigned underflow anyway.
    if (cursor_bytes + sizeof(IdT) > ub_available) return;
    uint32_t rest = ub_available - cursor_bytes;
    // Staging (col + eid = two buffers) gets a quarter of what is left,
    // capped at one chunk's worth of picks; streams share the rest.
    const uint32_t stage_bytes = rest / 4 & ~static_cast<uint32_t>(3);
    rest -= stage_bytes * 2;
    uint32_t chunk_bytes = rest / 6 & ~static_cast<uint32_t>(sizeof(IdT) - 1);
    chunk_elems_ = chunk_bytes / sizeof(IdT);
    if (chunk_elems_ == 0) chunk_elems_ = 1;
    stage_elems_ = stage_bytes / sizeof(IdT);
    if (stage_elems_ == 0) stage_elems_ = 1;

    pipe->InitBuffer(rows_q_, 2, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(cols_q_, 2, chunk_elems_ * sizeof(IdT));
    if (has_data_) pipe->InitBuffer(data_q_, 2, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(cursor_buf_, cursor_bytes);
    pipe->InitBuffer(stage_col_buf_, stage_bytes);
    pipe->InitBuffer(stage_eid_buf_, stage_bytes);

    // Cursors start at each row's indptr base: stream the block's
    // indptr slice in and reinterpret it as the cursor array.
    {
      LocalTensor<IdT> cur = cursor_buf_.Get<IdT>();
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyExtParams cp{
          1, static_cast<uint32_t>(range * sizeof(IdT)), 0, 0, 0};
      DataCopyPad(cur, indptr_gm_[row_begin_], cp, pad);
    }
    initialized_ = true;
  }

  __aicore__ inline void Process() {
    if (!initialized_) return;
    const IdT begin = static_cast<IdT>(row_begin_);
    const IdT end = static_cast<IdT>(row_end_);
    cur_ = cursor_buf_.Get<IdT>();
    stage_col_ = stage_col_buf_.Get<IdT>();
    stage_eid_ = stage_eid_buf_.Get<IdT>();
    staged_ = 0;

    for (int64_t base = 0; base < nnz_; base += chunk_elems_) {
      const uint32_t cnt = static_cast<uint32_t>(
          nnz_ - base < chunk_elems_ ? nnz_ - base : chunk_elems_);
      // One full queue cycle per array per chunk (queue discipline).
      LocalTensor<IdT> rb = rows_q_.AllocTensor<IdT>();
      DataCopyExtParams cp{
          1, static_cast<uint32_t>(cnt * sizeof(IdT)), 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(rb, rows_gm_[base], cp, pad);
      rows_q_.EnQue(rb);
      rb = rows_q_.DeQue<IdT>();

      LocalTensor<IdT> cb = cols_q_.AllocTensor<IdT>();
      DataCopyPad(cb, cols_gm_[base], cp, pad);
      cols_q_.EnQue(cb);
      cb = cols_q_.DeQue<IdT>();

      LocalTensor<IdT> db;
      if (has_data_) {
        db = data_q_.AllocTensor<IdT>();
        DataCopyPad(db, data_gm_[base], cp, pad);
        data_q_.EnQue(db);
        db = data_q_.DeQue<IdT>();
      }

      for (uint32_t i = 0; i < cnt; ++i) {
        const IdT r = rb.GetValue(i);
        if (r < begin || r >= end) continue;
        const uint32_t slot = static_cast<uint32_t>(r - begin);
        const IdT pos = cur_.GetValue(slot);
        cur_.SetValue(slot, pos + 1);
        if (staged_ == 0) {
          run_base_ = pos;  // first slot of the current contiguous run
        } else if (pos != run_base_ + static_cast<IdT>(staged_)) {
          FlushRun();  // gap: previous run is complete, start a new one
          run_base_ = pos;
        }
        stage_col_.SetValue(staged_, cb.GetValue(i));
        stage_eid_.SetValue(
            staged_, has_data_ ? db.GetValue(i) : static_cast<IdT>(base + i));
        ++staged_;
        if (staged_ == stage_elems_) FlushRun();
      }

      rows_q_.FreeTensor(rb);
      cols_q_.FreeTensor(cb);
      if (has_data_) data_q_.FreeTensor(db);
    }
    FlushRun();
  }

 private:
  // Emits the staged contiguous run (a single row's output slice) with
  // one DMA pair and resets the staging counter. GM-destination
  // DataCopyPad takes no pad params (three-arg form, v2 sampling kernel
  // precedent).
  __aicore__ inline void FlushRun() {
    if (staged_ == 0) return;
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(staged_ * sizeof(IdT)), 0, 0, 0};
    DataCopyPad(out_indices_gm_[run_base_], stage_col_, cp);
    DataCopyPad(out_data_gm_[run_base_], stage_eid_, cp);
    staged_ = 0;
  }

  GlobalTensor<IdT> rows_gm_, cols_gm_, data_gm_, indptr_gm_;
  GlobalTensor<IdT> out_indices_gm_, out_data_gm_;
  TQue<TPosition::VECIN, 2> rows_q_, cols_q_, data_q_;
  TBuf<TPosition::VECCALC> cursor_buf_, stage_col_buf_, stage_eid_buf_;
  LocalTensor<IdT> cur_, stage_col_, stage_eid_;
  int64_t nnz_ = 0;
  uint32_t num_rows_ = 0, has_data_ = 0, staged_ = 0;
  uint32_t row_begin_ = 0, row_end_ = 0, chunk_elems_ = 0, stage_elems_ = 0;
  IdT run_base_ = 0;
  bool initialized_ = false;
};

}  // namespace

extern "C" __global__ __aicore__ void coo_to_csr_count_int32(
    GM_ADDR rows, GM_ADDR counts, GM_ADDR reduces, GM_ADDR row_split,
    GM_ADDR tiling) {
  KernelCooToCsrCount<int32_t> op;
  TPipe pipe;
  op.Init(rows, counts, reduces, row_split, tiling, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_to_csr_count_int64(
    GM_ADDR rows, GM_ADDR counts, GM_ADDR reduces, GM_ADDR row_split,
    GM_ADDR tiling) {
  KernelCooToCsrCount<int64_t> op;
  TPipe pipe;
  op.Init(rows, counts, reduces, row_split, tiling, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_to_csr_scatter_int32(
    GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR indptr,
    GM_ADDR out_indices, GM_ADDR out_data, GM_ADDR row_split, GM_ADDR tiling) {
  KernelCooToCsrScatter<int32_t> op;
  TPipe pipe;
  op.Init(
      rows, cols, data, indptr, out_indices, out_data, row_split, tiling,
      &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void coo_to_csr_scatter_int64(
    GM_ADDR rows, GM_ADDR cols, GM_ADDR data, GM_ADDR indptr,
    GM_ADDR out_indices, GM_ADDR out_data, GM_ADDR row_split, GM_ADDR tiling) {
  KernelCooToCsrScatter<int64_t> op;
  TPipe pipe;
  op.Init(
      rows, cols, data, indptr, out_indices, out_data, row_split, tiling,
      &pipe);
  op.Process();
}
