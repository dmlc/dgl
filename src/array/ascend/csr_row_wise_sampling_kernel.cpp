/**
 * Copyright (c) 2024 by Contributors
 * @file csr_row_wise_sampling_kernel.cpp
 * @brief AscendC kernel for weighted (probability/mask) CSR row-wise sampling.
 *
 * Single-core (blockDim=1) scalar-GM kernel. For each requested seed row it
 * gathers the per-row edge probabilities into a workspace, computes the pick
 * count, performs weighted sampling, and writes the picked (row, col, edge_id)
 * triples into the over-allocated output buffers.
 *
 * Sampling methods:
 *  - replace        : CDF inverse-transform with binary search.
 *  - no-replace     : CDF linear scan with per-pick renormalization + marking.
 *  - select_all     : copy every neighbor with prob > 0.
 *
 * RNG: xorshift32 per-row seeded; uniform float via state / 2^32.
 *
 * NOTE: AscendC forbids double-precision operations inside __aicore__
 * functions, so all accumulation is done in float. Only float32 probability
 * input is supported (the host launcher is instantiated for FloatType=float).
 *
 * Workspace (passed as GM_ADDRs, each sized max_deg by the host):
 *  - prob_ws : per-row selected probabilities (float), prob[edge_id].
 *  - cdf_ws  : prefix-sum CDF (float), used for replace.
 *  - used_ws : per-row "used" generation markers (uint32), used for no-replace.
 *    A row i marks picked offset j by writing (i+1); 0 means unused, so no
 *    per-row clearing is required (the host memsets used_ws to 0 once).
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

__aicore__ inline uint32_t xorshift32(uint32_t& x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

// Uniform float in [0, 1).
// AscendC forbids double and uint<->float casts in __aicore__, so the uint32
// state is converted via a signed int32 intermediate.
__aicore__ inline float rand_unit(uint32_t& state) {
  uint32_t r = xorshift32(state);
  int32_t signed_r = static_cast<int32_t>(r);     // [-2^31, 2^31)
  float f = static_cast<float>(signed_r);          // [-2^31, 2^31)
  return (f + 2147483648.0f) / 4294967296.0f;      // [0, 1)
}

}  // namespace

template <typename IdT>
class KernelCsrRowWiseSampling {
 public:
  __aicore__ inline KernelCsrRowWiseSampling() {}

  __aicore__ inline void Init(
      GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows,
      GM_ADDR prob, GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols,
      GM_ADDR out_idxs, GM_ADDR prob_ws, GM_ADDR cdf_ws, GM_ADDR used_ws,
      uint32_t num_rows, uint32_t num_samples, uint32_t replace,
      uint32_t has_data, uint32_t seed, uint32_t select_all) {
    indptr_gm.SetGlobalBuffer((__gm__ IdT*)indptr);
    indices_gm.SetGlobalBuffer((__gm__ IdT*)indices);
    if (has_data) data_gm.SetGlobalBuffer((__gm__ IdT*)data);
    rows_gm.SetGlobalBuffer((__gm__ IdT*)rows, num_rows);
    prob_gm.SetGlobalBuffer((__gm__ float*)prob);
    out_ptr_gm.SetGlobalBuffer((__gm__ IdT*)out_ptr, num_rows + 1);
    out_rows_gm.SetGlobalBuffer((__gm__ IdT*)out_rows);
    out_cols_gm.SetGlobalBuffer((__gm__ IdT*)out_cols);
    out_idxs_gm.SetGlobalBuffer((__gm__ IdT*)out_idxs);
    prob_ws_gm.SetGlobalBuffer((__gm__ float*)prob_ws);
    cdf_ws_gm.SetGlobalBuffer((__gm__ float*)cdf_ws);
    used_ws_gm.SetGlobalBuffer((__gm__ uint32_t*)used_ws);
    num_rows_ = num_rows;
    num_samples_ = num_samples;
    replace_ = replace;
    has_data_ = has_data;
    seed_ = seed;
    select_all_ = select_all;
  }

  __aicore__ inline void Process() {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < num_rows_; ++i) {
      out_ptr_gm.SetValue(i, static_cast<IdT>(offset));

      IdT rid = rows_gm.GetValue(i);
      IdT off = indptr_gm.GetValue(static_cast<uint32_t>(rid));
      IdT end = indptr_gm.GetValue(static_cast<uint32_t>(rid) + 1);
      uint32_t deg = static_cast<uint32_t>(end - off);

      if (deg == 0) {
        continue;
      }

      // Gather per-row selected probabilities: prob_sel[j] = prob[eid(off+j)].
      // eid = has_data ? data[off+j] : (off+j).
      for (uint32_t j = 0; j < deg; ++j) {
        IdT eid = has_data_ ? data_gm.GetValue(static_cast<uint32_t>(off) + j)
                            : static_cast<IdT>(static_cast<uint32_t>(off) + j);
        float p = prob_gm.GetValue(static_cast<uint32_t>(eid));
        prob_ws_gm.SetValue(j, p);
      }

      // Count positive-probability neighbors.
      uint32_t nnz_pos = 0;
      for (uint32_t j = 0; j < deg; ++j) {
        if (prob_ws_gm.GetValue(j) > 0.0f) ++nnz_pos;
      }

      uint32_t num_picks = 0;
      if (select_all_) {
        num_picks = nnz_pos;
      } else if (replace_) {
        num_picks = (nnz_pos == 0) ? 0 : num_samples_;
      } else {
        num_picks = (nnz_pos < num_samples_) ? nnz_pos : num_samples_;
      }

      if (num_picks > 0) {
        uint32_t state = seed_ ^ (i * 2654435761u + 0x9e3779b9u);
        if (state == 0) state = 0x12345678u;

        if (select_all_) {
          // Copy every neighbor with prob > 0.
          uint32_t written = 0;
          for (uint32_t j = 0; j < deg && written < num_picks; ++j) {
            if (prob_ws_gm.GetValue(j) > 0.0f) {
              WritePick(offset + written, rid, off + static_cast<IdT>(j));
              ++written;
            }
          }
        } else if (replace_) {
          // CDF inverse-transform with binary search.
          float total = 0.0f;
          for (uint32_t j = 0; j < deg; ++j) {
            total += prob_ws_gm.GetValue(j);
            cdf_ws_gm.SetValue(j, total);
          }
          for (uint32_t p = 0; p < num_picks; ++p) {
            float u = rand_unit(state);
            float target = u * total;
            // Binary search: smallest j with cdf[j] > target.
            uint32_t lo = 0, hi = deg;
            while (lo < hi) {
              uint32_t mid = lo + (hi - lo) / 2;
              if (cdf_ws_gm.GetValue(mid) <= target) {
                lo = mid + 1;
              } else {
                hi = mid;
              }
            }
            uint32_t j = (lo < deg) ? lo : (deg - 1);
            WritePick(offset + p, rid, off + static_cast<IdT>(j));
          }
        } else {
          // No replacement: linear scan with renormalization + marking.
          // used_ws uses generation id (i+1); 0 means unused.
          float remaining = 0.0f;
          for (uint32_t j = 0; j < deg; ++j) {
            remaining += prob_ws_gm.GetValue(j);
          }
          for (uint32_t p = 0; p < num_picks; ++p) {
            if (remaining <= 0.0f) break;
            float u = rand_unit(state);
            float target = u * remaining;
            float running = 0.0f;
            uint32_t picked_j = deg;  // sentinel
            for (uint32_t j = 0; j < deg; ++j) {
              if (used_ws_gm.GetValue(j) == i + 1) continue;
              float pv = prob_ws_gm.GetValue(j);
              if (pv <= 0.0f) continue;
              running += pv;
              if (running > target) {
                picked_j = j;
                break;
              }
            }
            if (picked_j == deg) {
              // Floating-point fallback: first usable offset.
              for (uint32_t j = 0; j < deg; ++j) {
                if (used_ws_gm.GetValue(j) == i + 1) continue;
                if (prob_ws_gm.GetValue(j) > 0.0f) {
                  picked_j = j;
                  break;
                }
              }
            }
            if (picked_j == deg) break;  // no usable neighbor left
            used_ws_gm.SetValue(picked_j, i + 1);
            remaining -= prob_ws_gm.GetValue(picked_j);
            WritePick(offset + p, rid, off + static_cast<IdT>(picked_j));
          }
        }
      }
      offset += num_picks;
    }
    out_ptr_gm.SetValue(num_rows_, static_cast<IdT>(offset));
  }

 private:
  __aicore__ inline void WritePick(uint32_t pos, IdT rid, IdT picked) {
    out_rows_gm.SetValue(pos, rid);
    out_cols_gm.SetValue(pos, indices_gm.GetValue(static_cast<uint32_t>(picked)));
    out_idxs_gm.SetValue(
        pos, has_data_ ? data_gm.GetValue(static_cast<uint32_t>(picked))
                       : picked);
  }

  GlobalTensor<IdT> indptr_gm;
  GlobalTensor<IdT> indices_gm;
  GlobalTensor<IdT> data_gm;
  GlobalTensor<IdT> rows_gm;
  GlobalTensor<float> prob_gm;
  GlobalTensor<IdT> out_ptr_gm;
  GlobalTensor<IdT> out_rows_gm;
  GlobalTensor<IdT> out_cols_gm;
  GlobalTensor<IdT> out_idxs_gm;
  GlobalTensor<float> prob_ws_gm;
  GlobalTensor<float> cdf_ws_gm;
  GlobalTensor<uint32_t> used_ws_gm;
  uint32_t num_rows_ = 0;
  uint32_t num_samples_ = 0;
  uint32_t replace_ = 0;
  uint32_t has_data_ = 0;
  uint32_t seed_ = 0;
  uint32_t select_all_ = 0;
};

extern "C" __global__ __aicore__ void csr_row_wise_sampling_int32_f32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR prob,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR prob_ws, GM_ADDR cdf_ws, GM_ADDR used_ws, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 6);
  KernelCsrRowWiseSampling<int32_t> op;
  op.Init(indptr, indices, data, rows, prob, out_ptr, out_rows, out_cols,
          out_idxs, prob_ws, cdf_ws, used_ws, tilingGm.GetValue(0),
          tilingGm.GetValue(1), tilingGm.GetValue(2), tilingGm.GetValue(3),
          tilingGm.GetValue(4), tilingGm.GetValue(5));
  op.Process();
}

extern "C" __global__ __aicore__ void csr_row_wise_sampling_int64_f32(
    GM_ADDR indptr, GM_ADDR indices, GM_ADDR data, GM_ADDR rows, GM_ADDR prob,
    GM_ADDR out_ptr, GM_ADDR out_rows, GM_ADDR out_cols, GM_ADDR out_idxs,
    GM_ADDR prob_ws, GM_ADDR cdf_ws, GM_ADDR used_ws, GM_ADDR tiling_ptr) {
  GlobalTensor<uint32_t> tilingGm;
  tilingGm.SetGlobalBuffer((__gm__ uint32_t*)tiling_ptr, 6);
  KernelCsrRowWiseSampling<int64_t> op;
  op.Init(indptr, indices, data, rows, prob, out_ptr, out_rows, out_cols,
          out_idxs, prob_ws, cdf_ws, used_ws, tilingGm.GetValue(0),
          tilingGm.GetValue(1), tilingGm.GetValue(2), tilingGm.GetValue(3),
          tilingGm.GetValue(4), tilingGm.GetValue(5));
  op.Process();
}
