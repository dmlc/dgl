/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_uniform_fused_prep_kernel.cpp
 * @brief Single-core preparation kernel for uniform CSR fused sampling.
 *
 * Replaces the host-side pick/partition chain (degree D2H → CPU pick
 * computation → balanced partition → block-offset prefix sums → packed
 * table upload) with one device pass over the degree array that
 * CSRGetRowNNZ already produced. Single core on purpose: the workload is
 * a linear 100k-element scan (~1ms), far below the multi-block
 * coordination cost, and a single writer keeps row_split/out_starts
 * trivially consistent.
 *
 * Outputs (all device-resident, consumed directly by the sampling
 * kernel through typed __gm__ pointers):
 *   picks[num_rows]      per-row pick counts (uint32)
 *   row_split[block_dim + 1]  nnz-balanced row-range boundaries
 *   out_starts[block_dim + 1] per-block output offsets; the final entry
 *                             doubles as max_output for the host's
 *                             output allocation (one scalar D2H)
 *
 * The balancing mirrors BuildBalancedPartitionsFused exactly: prefix
 * targets at total*part/block_dim, first index whose prefix >= target,
 * ties resolved to the same boundary (lower_bound semantics).
 */

#include "csr_row_wise_sampling_uniform_fused_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

// Degrees staged per GM read. Sized so that, together with the
// double-buffered pick queue below, the kernel stays within the AIV UB
// budget for the int64 degree instantiation (2 deg windows + 2 pick
// windows + the two block tables).
constexpr uint32_t kPrepWindow = 4096;

}  // namespace

template <typename DegT>
class KernelFusedPrep {
 public:
  __aicore__ inline KernelFusedPrep() {}

  __aicore__ inline void Init(
      GM_ADDR deg, GM_ADDR picks, GM_ADDR row_split, GM_ADDR out_starts,
      GM_ADDR tiling_ptr, TPipe* pipe) {
    const __gm__ uint32_t* tiling = (const __gm__ uint32_t*)tiling_ptr;
    num_rows_ = tiling[0];
    fanout_ = tiling[1];
    replace_ = tiling[2];
    select_all_ = tiling[3];
    block_dim_ = tiling[4] <= kMaxBlocks ? tiling[4] : kMaxBlocks;

    // Degrees arrive with the graph's idtype (int32/int64) — read them
    // through the matching type or every other word is garbage.
    deg_gm_.SetGlobalBuffer((__gm__ DegT*)deg, num_rows_);
    picks_gm_.SetGlobalBuffer((__gm__ uint32_t*)picks, num_rows_);
    row_split_gm_.SetGlobalBuffer((__gm__ uint32_t*)row_split, block_dim_ + 1);
    out_starts_gm_.SetGlobalBuffer(
        (__gm__ uint32_t*)out_starts, block_dim_ + 1);

    pipe->InitBuffer(win_buf_, 2, kPrepWindow * sizeof(DegT));
    pipe->InitBuffer(pick_q_, 2, kPrepWindow * sizeof(uint32_t));
    pipe->InitBuffer(split_buf_, (kMaxBlocks + 1) * sizeof(uint32_t));
    pipe->InitBuffer(starts_buf_, (kMaxBlocks + 1) * sizeof(uint32_t));
  }

  __aicore__ inline void Process() {
    // Pass over the degrees in windows: compute picks, stage them for a
    // batched write, and track the running prefix. row_split/out_starts
    // are small (< 42 entries) and written once at the end through the
    // MTE path (multi-word GM scalar stores are banned family-wide).
    LocalTensor<uint32_t> split_local = split_buf_.Get<uint32_t>();

    uint64_t prefix = 0;     // running pick total
    uint32_t next_part = 1;  // next boundary index to resolve
    uint32_t split_idx = 0;  // rows consumed by resolved boundaries
    split_local.SetValue(0, 0);

    uint32_t row = 0;
    while (row < num_rows_) {
      const uint32_t avail = num_rows_ - row;
      const uint32_t count = avail < kPrepWindow ? avail : kPrepWindow;
      const uint32_t copy_bytes = count * sizeof(DegT);
      DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
      DataCopyPadExtParams<DegT> pad{false, 0, 0, 0};
      LocalTensor<DegT> deg_win = win_buf_.AllocTensor<DegT>();
      DataCopyPad(deg_win, deg_gm_[row], cp, pad);
      win_buf_.EnQue(deg_win);
      deg_win = win_buf_.DeQue<DegT>();

      LocalTensor<uint32_t> pick_win = pick_q_.AllocTensor<uint32_t>();
      for (uint32_t j = 0; j < count; ++j) {
        pick_win.SetValue(j, PickOf(deg_win.GetValue(j)));
      }

      // Boundaries resolve in the second pass below (targets need the
      // final total); this loop is a pure scan.

      // Consume pick_win on the vector side while the buffer is still
      // V-owned: the running prefix must be accumulated before the
      // buffer is handed to MTE3.
      for (uint32_t j = 0; j < count; ++j) {
        prefix += pick_win.GetValue(j);
      }
      // Batched picks write (MTE path for cross-block consistency).
      // Staged through the VECOUT queue (same idiom as CopyOutStaged in
      // the sampling kernel): the EnQue/DeQue pair orders the V-side
      // SetValue stores before the MTE-side read, and the depth-2 pool
      // hands out a different buffer each window, so a still-in-flight
      // copy from the previous window is never overwritten (DataCopyPad
      // is asynchronous).
      pick_q_.EnQue(pick_win);
      LocalTensor<uint32_t> ready = pick_q_.DeQue<uint32_t>();
      {
        const uint32_t write_bytes = count * sizeof(uint32_t);
        DataCopyExtParams wcp{1, write_bytes, 0, 0, 0};
        DataCopyPad(picks_gm_[row], ready, wcp);
      }
      pick_q_.FreeTensor(ready);
      row += count;
      win_buf_.FreeTensor(deg_win);
    }

    // Boundary resolution needs targets = total*part/block_dim, which
    // needs the full prefix. The pick count is a pure function of the
    // degree, so this pass recomputes it from deg_gm_ instead of reading
    // back the picks_gm_ image the MTE3 copies above just started
    // writing: a scalar GM read of an in-flight MTE3 target has no
    // inter-pipe ordering guarantee (PipeBarrier<PIPE_MTE3> orders the
    // MTE3 pipe against itself, not the scalar unit against it), and the
    // read-back was observed returning stale words — zero tails that
    // collapsed out_starts, wrong boundaries that overlapped row ranges.
    const uint32_t total = static_cast<uint32_t>(prefix);
    uint32_t boundary = 0;  // candidate row index
    uint64_t run2 = 0;      // second-pass prefix
    uint32_t part = 1;
    uint32_t resolved = 1;  // entries filled in split_local (incl. [0])
    split_local.SetValue(0, 0);
    while (boundary < num_rows_ && part < block_dim_) {
      const uint32_t target = static_cast<uint32_t>(
          (static_cast<uint64_t>(total) * part) / block_dim_);
      if (total == 0) {
        // Degenerate: equal row splits (matches the host's zero-weight
        // fallback).
        for (; part < block_dim_; ++part, ++resolved) {
          split_local.SetValue(
              resolved,
              static_cast<uint32_t>(
                  (static_cast<uint64_t>(part) * num_rows_) / block_dim_));
        }
        break;
      }
      if (run2 < target) {
        // advance one row at a time until the prefix reaches the target
        run2 += PickOf(static_cast<uint32_t>(deg_gm_.GetValue(boundary)));
        ++boundary;
      } else {
        // boundary is the first index with prefix >= target
        split_local.SetValue(resolved, boundary);
        ++resolved;
        ++part;
      }
    }
    for (; resolved <= block_dim_; ++resolved) {
      split_local.SetValue(resolved, num_rows_);
    }
    // Guard monotonicity against ties (mirrors the host path).
    for (uint32_t i = 1; i <= block_dim_; ++i) {
      const uint32_t prev = split_local.GetValue(i - 1);
      const uint32_t cur = split_local.GetValue(i);
      if (cur < prev) split_local.SetValue(i, prev);
    }

    // out_starts[b] = prefix at row_split[b]; one more scan building
    // both tables' final GM images through the MTE path.
    LocalTensor<uint32_t> starts_local = starts_buf_.Get<uint32_t>();
    uint64_t run3 = 0;
    uint32_t sb = 0;  // index into split_local
    uint32_t next_split = split_local.GetValue(0);
    for (uint32_t i = 0; i <= num_rows_; ++i) {
      while (sb < block_dim_ && i == next_split) {
        starts_local.SetValue(sb, static_cast<uint32_t>(run3));
        ++sb;
        next_split = split_local.GetValue(sb);
      }
      if (i < num_rows_) {
        run3 += PickOf(static_cast<uint32_t>(deg_gm_.GetValue(i)));
      }
    }
    starts_local.SetValue(block_dim_, static_cast<uint32_t>(run3));

    {
      const uint32_t bytes = (block_dim_ + 1) * sizeof(uint32_t);
      DataCopyExtParams cp1{1, bytes, 0, 0, 0};
      (void)cp1;
      DataCopyPad(row_split_gm_[0], split_local, cp1);
      DataCopyExtParams cp2{1, bytes, 0, 0, 0};
      DataCopyPad(out_starts_gm_[0], starts_local, cp2);
    }
  }

 private:
  // Pick count for one row, a pure function of its degree. The staging
  // pass and both scans below recompute it from the same degree word,
  // so every consumer sees bit-identical values without ever reading
  // back the picks_gm_ image while its MTE3 copy is in flight.
  __aicore__ inline uint32_t PickOf(uint32_t d) const {
    if (select_all_) return d;
    if (replace_) return d == 0 ? 0u : fanout_;
    return d < fanout_ ? d : fanout_;
  }

  // row_split capacity guard, shared with the host launcher (which
  // clamps the launched block count to it). The kernel-side clamp below
  // is belt-and-braces: the UB tables hold kMaxSamplingBlocksFused + 1
  // entries, so a hostile tiling could otherwise overflow them silently.
  static constexpr uint32_t kMaxBlocks = kMaxSamplingBlocksFused;

  GlobalTensor<DegT> deg_gm_;
  GlobalTensor<uint32_t> picks_gm_, row_split_gm_, out_starts_gm_;
  uint32_t num_rows_ = 0, fanout_ = 0, replace_ = 0, select_all_ = 0,
           block_dim_ = 0;
  TQue<TPosition::VECIN, 2> win_buf_;
  TQue<TPosition::VECOUT, 2> pick_q_;
  TBuf<TPosition::VECCALC> split_buf_, starts_buf_;
};

extern "C" __global__ __aicore__ void
csr_row_wise_sampling_uniform_fused_prep_int32(
    GM_ADDR deg, GM_ADDR picks, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelFusedPrep<int32_t> op;
  TPipe pipe;
  op.Init(deg, picks, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}

extern "C" __global__ __aicore__ void
csr_row_wise_sampling_uniform_fused_prep_int64(
    GM_ADDR deg, GM_ADDR picks, GM_ADDR row_split, GM_ADDR out_starts,
    GM_ADDR tiling_ptr) {
  KernelFusedPrep<int64_t> op;
  TPipe pipe;
  op.Init(deg, picks, row_split, out_starts, tiling_ptr, &pipe);
  op.Process();
}
