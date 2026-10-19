/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_topk_kernel.cpp
 * @brief Multi-core AIV kernel for CSR row-wise topk on Ascend NPU.
 *
 * Per row, the CSR window [off, off+deg) of weights (gathered through the
 * eid mapping when the matrix carries a data array) is bulk-copied GM -> UB,
 * sorted with the hardware sort network (Sort<float, true> = Sort32 +
 * in-UB multi-way merge on 910B), and the head k (descending) or tail k
 * (ascending) entries are gathered back to (row, col, eid) triples and
 * copied out through the VECOUT queue.
 *
 * Rows whose degree exceeds the UB window fall back to a direct-GM scalar
 * heap (v2 sampling precedent), keeping memory bounded for skewed graphs.
 *
 * Weight ordering semantics: the sort network is float-only (P0 finding);
 * non-f32 weights are normalized to f32 on the host before launch.
 */

#include "csr_row_wise_topk_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {}  // namespace

template <typename IdT>
class KernelCsrRowWiseTopk {
 public:
  __aicore__ inline KernelCsrRowWiseTopk() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR weight, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
      GM_ADDR row_split, GM_ADDR out_starts, GM_ADDR tiling_ptr, TPipe* pipe) {

    const __gm__ CsrRowWiseTopkTiling* tiling =
        (const __gm__ CsrRowWiseTopkTiling*)tiling_ptr;
    num_rows_ = tiling->num_rows;
    k_ = tiling->k;
    select_all_ = tiling->select_all;
    ascending_ = tiling->ascending;
    has_data_ = tiling->has_data;
    num_total_rows_ = tiling->num_total_rows;
    const uint32_t ub_available = tiling->ub_available;

    indptr_gm_.SetGlobalBuffer((__gm__ IdT*)indptr, num_total_rows_ + 1);
    indices_gm_.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data_) data_gm_.SetGlobalBuffer((__gm__ IdT*)data);
    rows_gm_.SetGlobalBuffer((__gm__ IdT*)rows, num_rows_);
    weight_gm_.SetGlobalBuffer((__gm__ float*)weight);
    out_rows_gm_.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm_.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm_.SetGlobalBuffer((__gm__ IdT*)out_idxs);

    const uint32_t block_idx = AscendC::GetBlockIdx();
    block_idx_ = block_idx;
    const __gm__ uint32_t* row_split_p = (const __gm__ uint32_t*)row_split;
    const __gm__ uint32_t* out_starts_p = (const __gm__ uint32_t*)out_starts;
    row_begin_ = row_split_p[block_idx];
    row_end_ = row_split_p[block_idx + 1];
    out_start_ = out_starts_p[block_idx];

    // UB layout (multi-core rule 1 — total bytes must fit the queried
    // per-core budget). Per window element, worst case (IdT = int32):
    //   12 x 4B : double-buffered VECIN weight/eid windows + VECOUT queue
    //    1 x 4B : u32 index array       (index_buf_)
    //    1 x 8B : proposal buffer       (sorted_buf_)
    //    1 x 8B : heap scratch          (sort_tmp_buf_, GM fallback)
    //    3 x 4B : output triple staging (out_r/c/e)
    //  = 68 bytes per window element (queue-free emission dropped the
    //  double-buffered VECOUT queue).
    // Exact per-element layout cost (see the launcher: 48 + 7*IdT
    // bytes; the historical 68B constant overcommitted the int64
    // TPipe past the physical UB tail on this runtime).
    const uint32_t bytes_per_elem = 48 + 7 * static_cast<uint32_t>(sizeof(IdT));
    window_elems_ = ub_available / bytes_per_elem;
    // Device-verified envelope (see the launcher): the UB sort path
    // is exact up to 992-element windows on this runtime.
    constexpr uint32_t kVerifiedWindowElems = 992;
    if (window_elems_ > kVerifiedWindowElems) {
      window_elems_ = kVerifiedWindowElems;
    }
    // The staged sort pads to 32; rows up to the window take the UB path.
    window_elems_ = window_elems_ / kSortElemsPerRepeat * kSortElemsPerRepeat;
    // The sort network processes 32 elements per repeat and repeatTimes is
    // capped at 255 -> one Sort call covers at most 8160 elements; the
    // window is comfortably below that on every known SoC budget.
    pipe->InitBuffer(win_w_q_, kQueueDepth, window_elems_ * sizeof(float));
    pipe->InitBuffer(win_w_buf_, window_elems_ * sizeof(float));
    pipe->InitBuffer(win_e_q_, kQueueDepth, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(sort_tmp_buf_, window_elems_ * sizeof(float) * 2);
    pipe->InitBuffer(sorted_buf_, window_elems_ * sizeof(float) * 2);
    pipe->InitBuffer(index_buf_, window_elems_ * sizeof(uint32_t));
    pipe->InitBuffer(out_r_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_c_buf_, window_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_e_buf_, window_elems_ * sizeof(IdT));
  }

  __aicore__ inline void Process() {
    if (row_begin_ >= row_end_) return;
    uint32_t offset = 0;
    for (uint32_t i = row_begin_; i < row_end_; ++i) {
      IdT rid = rows_gm_.GetValue(i);
      if (rid < 0 || rid >= static_cast<IdT>(num_total_rows_)) {
        continue;  // defense in depth: drop invalid seed rows
      }
      IdT off = indptr_gm_.GetValue(static_cast<uint32_t>(rid));
      IdT end = indptr_gm_.GetValue(static_cast<uint32_t>(rid) + 1);
      const uint32_t deg = static_cast<uint32_t>(end - off);
      const uint32_t num_picks = select_all_ ? deg : (deg < k_ ? deg : k_);
      if (num_picks == 0) continue;
      // The UB path sorts the whole padded window with the high-level
      // Sort (Sort32 runs + in-UB hardware merging); rows wider than the
      // window take the scalar GM heap.
      // select-all fast path: every edge is picked, so the weight order
      // carries no information — emit the window as-is and skip the sort
      // entirely (k == -1 semantics; ties with the CPU's weight-ordered
      // output are not required, the edge SET is what matters).
      if (select_all_ && num_picks == deg) {
        offset += EmitRowDirect(out_start_ + offset, rid, off, deg);
        continue;
      }
      // Rows wider than the window are rejected by the launcher
      // (the GM scalar-heap fallback is unreliable on this runtime);
      // this branch only survives tiling corruption, so the row is
      // dropped defensively.
      if (deg <= window_elems_) {
        offset +=
            TopkRowThroughUb(out_start_ + offset, rid, off, deg, num_picks);
      }
    }
  }

 private:
  __aicore__ inline uint32_t TopkRowThroughUb(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg, uint32_t num_picks) {
    // The DMA pads the weight window to whole 32-element runs in one
    // copy (right padding carries -inf, which sorts to the run tail).
    const uint32_t padded = (deg + kSortElemsPerRepeat - 1) /
                            kSortElemsPerRepeat * kSortElemsPerRepeat;
    const uint32_t pad_tail = padded - deg;
    LocalTensor<float> win_w;
    if (has_data_) {
      // Weight indices go through the eid mapping: copy the data window
      // first, then gather weights by it. The weight window is filled by
      // scalar stores (no DMA), so it lives in a plain VECCALC buffer —
      // a queue cycle here would be fake (no MTE2 producer) and corrupts
      // the queue's event state across calls. The pad tail is laid down
      // with one vectorized Duplicate and the head overwritten by the
      // deg scalar gathers.
      win_w = win_w_buf_.Get<float>();
      Duplicate(win_w, kPadValue, padded);
      LocalTensor<IdT> win_e = win_e_q_.AllocTensor<IdT>();
      DataCopyPad(
          win_e, data_gm_[off],
          DataCopyExtParams{1, deg * sizeof(IdT), 0, 0, 0},
          DataCopyPadExtParams<IdT>{false, 0, 0, 0});
      win_e_q_.EnQue(win_e);
      win_e = win_e_q_.DeQue<IdT>();
      for (uint32_t j = 0; j < deg; ++j) {
        win_w.SetValue(
            j, weight_gm_.GetValue(static_cast<uint32_t>(win_e.GetValue(j))));
      }
      win_e_q_.FreeTensor(win_e);
    } else {
      win_w = win_w_q_.AllocTensor<float>();
      DataCopyPad(
          win_w, weight_gm_[off],
          DataCopyExtParams{
              1, deg * static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
          DataCopyPadExtParams<float>{
              true, 0, static_cast<uint8_t>(pad_tail), kPadValue});
      win_w_q_.EnQue(win_w);
      win_w = win_w_q_.DeQue<float>();
    }

    // Sort the whole padded window with the hardware sort network in
    // proposal format (8B per element: value u32 + index u32, descending
    // by value): Sort32 runs merged in-UB by the high-level Sort. The
    // merge scratch must hold repeatTimes * 64 floats (the in-UB
    // full-sort contract, sized by formula since the tiling-side helper
    // needs a PlatformAscendC the direct-invoke mode does not have).
    LocalTensor<int32_t> index = index_buf_.Get<int32_t>();
    ArithProgression(index, 0, 1, padded);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sorted_a = sorted_buf_.Get<float>();
    LocalTensor<float> sort_tmp = sort_tmp_buf_.Get<float>();
    const uint32_t repeat_times = padded / kSortElemsPerRepeat;
    Sort<float, true>(
        sorted_a, win_w, index.ReinterpretCast<uint32_t>(), sort_tmp,
        static_cast<int32_t>(repeat_times));

    // The network yields a DESCENDING list (device-verified: raw float
    // bits, no sign folding) with the -inf pads at its tail. The k
    // largest are the head window; the k smallest are the tail of the
    // real-value region [deg - num_picks, deg) — pads sit beyond deg and
    // never enter it.
    const uint32_t src = ascending_ ? deg - num_picks : 0;
    LocalTensor<uint32_t> sorted_u32 = sorted_a.ReinterpretCast<uint32_t>();
    EmitPicks(sorted_u32, src, num_picks, out_pos, rid, off);
    if (!has_data_) win_w_q_.FreeTensor(win_w);
    return num_picks;
  }

  __aicore__ inline void EmitPicks(
      LocalTensor<uint32_t>& sorted_u32, uint32_t src, uint32_t num_picks,
      uint32_t out_pos, IdT rid, IdT off) {
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    for (uint32_t j = 0; j < num_picks; ++j) {
      // Proposal layout (device-verified): (key, index) pairs interleaved,
      // 8B per element, ascending by key. The key is the raw float bits
      // with the sign bit folded (radix-order); only the index is consumed.
      const uint32_t local = sorted_u32.GetValue(2 * (src + j) + 1);
      const IdT picked = static_cast<IdT>(off) + static_cast<IdT>(local);
      out_r.SetValue(j, rid);
      out_c.SetValue(j, indices_gm_.GetValue(picked));
      out_e.SetValue(j, has_data_ ? data_gm_.GetValue(picked) : picked);
    }
    CopyOutStaged(out_r, out_rows_gm_[out_pos], num_picks);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], num_picks);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], num_picks);
  }

  // select-all fast path: emit the row's edges in CSR order without
  // sorting (all of them are picked).
  __aicore__ inline uint32_t EmitRowDirect(
      uint32_t out_pos, IdT rid, IdT off, uint32_t deg) {
    LocalTensor<IdT> out_r = out_r_buf_.Get<IdT>();
    LocalTensor<IdT> out_c = out_c_buf_.Get<IdT>();
    LocalTensor<IdT> out_e = out_e_buf_.Get<IdT>();
    for (uint32_t j = 0; j < deg; ++j) {
      const IdT picked = off + static_cast<IdT>(j);
      out_r.SetValue(j, rid);
      out_c.SetValue(j, indices_gm_.GetValue(picked));
      out_e.SetValue(j, has_data_ ? data_gm_.GetValue(picked) : picked);
    }
    CopyOutStaged(out_r, out_rows_gm_[out_pos], deg);
    CopyOutStaged(out_c, out_cols_gm_[out_pos], deg);
    CopyOutStaged(out_e, out_idxs_gm_[out_pos], deg);
    return deg;
  }

  static constexpr uint32_t kQueueDepth = 2;           // double buffering
  static constexpr uint32_t kSortElemsPerRepeat = 32;  // Sort32 granularity
  static constexpr uint32_t kMergeWays = 4;            // MrgSort lane count
  static constexpr uint32_t kProposalFloats = 2;       // proposal = 2 u32 words
  static constexpr float kPadValue = -3.402823466e38F;  // -FLT_MAX

  __aicore__ inline void CopyOutStaged(
      LocalTensor<IdT>& staging, GlobalTensor<IdT> dst, uint32_t count) {
    // Queue-free emission (profiler-driven, scalar bound at 88%): the
    // staging tensor is a plain VECCALC buffer, so the queue cycle here
    // was pure overhead — AllocTensor + count scalar copies + EnQue/DeQue
    // events + FreeTensor per array, three arrays per row. The V->MTE3
    // visibility the queue provided becomes one explicit event pair.
    event_t eid =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eid);
    WaitFlag<HardEvent::V_MTE3>(eid);
    DataCopyExtParams cp{
        1, static_cast<uint32_t>(count * sizeof(IdT)), 0, 0, 0};
    DataCopyPad(dst, staging, cp);
  }

  GlobalTensor<IdT> indptr_gm_, indices_gm_, data_gm_, rows_gm_;
  GlobalTensor<float> weight_gm_;
  GlobalTensor<IdT> out_rows_gm_, out_cols_gm_, out_idxs_gm_;
  TQue<TPosition::VECIN, kQueueDepth> win_w_q_, win_e_q_;
  TBuf<TPosition::VECCALC> win_w_buf_;
  TBuf<TPosition::VECCALC> sort_tmp_buf_, sorted_buf_;
  TBuf<TPosition::VECCALC> index_buf_;
  TBuf<TPosition::VECCALC> out_r_buf_, out_c_buf_, out_e_buf_;
  uint32_t num_rows_ = 0, k_ = 0, select_all_ = 0, ascending_ = 0;
  uint32_t has_data_ = 0, num_total_rows_ = 0;
  uint32_t out_start_ = 0, row_begin_ = 0, row_end_ = 0, window_elems_ = 0;
  uint32_t block_idx_ = 0;
};

extern "C" __global__ __aicore__ void csr_row_wise_topk_int32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR weight,
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs, GM_ADDR row_split,
    GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KernelCsrRowWiseTopk<int32_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, weight, out_rows, out_cols, out_idxs,
      row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_topk_int64(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR weight,
    GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs, GM_ADDR row_split,
    GM_ADDR out_starts, GM_ADDR tiling_ptr) {
  KernelCsrRowWiseTopk<int64_t> op;
  TPipe pipe;
  op.Init(
      indptr, indices, data, rows, weight, out_rows, out_cols, out_idxs,
      row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}
