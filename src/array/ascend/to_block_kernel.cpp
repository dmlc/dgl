#include "to_block_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

// ---------------------------------------------------------------------------
// K1: dedup_hash_build (single-core, block_dim=1)
// ---------------------------------------------------------------------------
template <typename IdT>
class KernelDedupHashBuild {
 public:
  __aicore__ inline void Init(
      GM_ADDR src_nodes_in, GM_ADDR keys_out, GM_ADDR vals_out,
      GM_ADDR unique_out, GM_ADDR count_out, GM_ADDR tiling_ptr,
      TPipe* pipe) {
    const __gm__ ToBlockTiling* tiling =
        (const __gm__ ToBlockTiling*)tiling_ptr;
    num_ids_ = tiling->num_ids;
    num_seeds_ = tiling->num_seeds;
    hash_capacity_ = static_cast<uint32_t>(tiling->hash_capacity);
    hash_mask_ = static_cast<IdT>(tiling->hash_mask);
    const uint32_t ub_avail = tiling->ub_available;

    src_gm_.SetGlobalBuffer((__gm__ IdT*)src_nodes_in, num_ids_);
    keys_out_gm_.SetGlobalBuffer((__gm__ IdT*)keys_out, hash_capacity_);
    vals_out_gm_.SetGlobalBuffer((__gm__ IdT*)vals_out, hash_capacity_);
    unique_gm_.SetGlobalBuffer((__gm__ IdT*)unique_out, num_ids_);
    count_gm_.SetGlobalBuffer((__gm__ int64_t*)count_out, 1);

    // UB budget: hash keys + hash values + stream buffer + staging buffer
    const uint32_t hash_bytes = 2 * hash_capacity_ * sizeof(IdT);
    const uint32_t avail = (ub_avail > hash_bytes) ? (ub_avail - hash_bytes) : 0;
    chunk_elems_ = avail / (4 * sizeof(IdT));
    if (chunk_elems_ == 0) chunk_elems_ = 64;
    if (chunk_elems_ > 65536) chunk_elems_ = 65536;

    pipe->InitBuffer(stream_q_, 2, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(keys_buf_, hash_capacity_ * sizeof(IdT));
    pipe->InitBuffer(vals_buf_, hash_capacity_ * sizeof(IdT));
    pipe->InitBuffer(staging_buf_, chunk_elems_ * sizeof(IdT));
    // 32B-aligned staging for the int64 count (DataCopyPad UB source must be
    // 32-byte aligned; a small 8B TBuf would not satisfy that requirement).
    pipe->InitBuffer(count_buf_, 32);

    keys_ = keys_buf_.Get<IdT>();
    vals_ = vals_buf_.Get<IdT>();
    staging_ = staging_buf_.Get<IdT>();
    count_local_ = count_buf_.Get<int64_t>();
  }

  __aicore__ inline void Process() {
    // Initialize hash table to empty (-1 sentinel)
    for (uint32_t i = 0; i < hash_capacity_; ++i) {
      keys_.SetValue(i, static_cast<IdT>(kToBlockEmptyKey));
      vals_.SetValue(i, static_cast<IdT>(kToBlockEmptyKey));
    }
    // Scalar writes of the hash table must be visible before any DMA read.
    AscendC::PipeBarrier<PIPE_ALL>();

    IdT next_local = 0;
    uint32_t staged = 0;
    bool saturated = false;

    for (int64_t base = 0; base < num_ids_; base += chunk_elems_) {
      uint32_t cnt = static_cast<uint32_t>(
          (num_ids_ - base < chunk_elems_) ? (num_ids_ - base) : chunk_elems_);
      uint32_t copy_bytes = cnt * static_cast<uint32_t>(sizeof(IdT));

      // Load chunk via VECIN queue
      LocalTensor<IdT> buf = stream_q_.AllocTensor<IdT>();
      DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(buf, src_gm_[base], cp, pad);
      stream_q_.EnQue(buf);
      buf = stream_q_.DeQue<IdT>();

      for (uint32_t i = 0; i < cnt; ++i) {
        IdT id = buf.GetValue(i);
        // Quadratic probe in open-addressing hash table
        uint32_t pos = static_cast<uint32_t>(id & hash_mask_);
        uint32_t delta = 1;
        bool found = false;
        while (true) {
          IdT existing = keys_.GetValue(pos);
          if (existing == static_cast<IdT>(kToBlockEmptyKey)) {
            // Empty slot: insert
            keys_.SetValue(pos, id);
            vals_.SetValue(pos, next_local);
            staging_.SetValue(staged, id);
            ++staged;
            ++next_local;
            found = true;
            break;
          } else if (existing == id) {
            // Already exists: skip
            found = true;
            break;
          }
          // Quadratic probing: pos = (pos + delta^2) & mask
          pos = static_cast<uint32_t>(
              (static_cast<uint64_t>(pos) +
               static_cast<uint64_t>(delta) * static_cast<uint64_t>(delta)) &
              static_cast<uint64_t>(hash_mask_));
          ++delta;
          // Saturation guard: if we probed every slot without finding an
          // empty one or the key, the table is full. Stop inserting so the
          // kernel does not hang (R2); the host-side capacity check should
          // have rejected this case, this is a belt-and-suspenders guard.
          if (delta > kToBlockMaxProbes) {
            saturated = true;
            break;
          }
        }
        if (saturated) break;

        // Flush staging if full
        if (staged >= chunk_elems_) {
          // Scalar staging writes must be visible before MTE3 reads them.
          AscendC::PipeBarrier<PIPE_ALL>();
          DataCopyExtParams ocp{1, staged * static_cast<uint32_t>(sizeof(IdT)), 0, 0, 0};
          DataCopyPadExtParams<IdT> opad{false, 0, 0, 0};
          DataCopyPad(
              unique_gm_[static_cast<int64_t>(next_local) - static_cast<int64_t>(staged)],
              staging_, ocp);
          staged = 0;
        }
      }
      stream_q_.FreeTensor(buf);
      if (saturated) break;
    }

    // Final flush of remaining staged unique ids
    if (staged > 0) {
      // Scalar staging writes must be visible before MTE3 reads them.
      AscendC::PipeBarrier<PIPE_ALL>();
      DataCopyExtParams cp{1, staged * static_cast<uint32_t>(sizeof(IdT)), 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(
          unique_gm_[static_cast<int64_t>(next_local) - static_cast<int64_t>(staged)],
          staging_, cp);
    }

    if (saturated) {
      // Table saturated: cannot guarantee correctness. Write -1 count so
      // the host can detect and report an error.
      count_local_.SetValue(0, static_cast<int64_t>(-1));
      AscendC::PipeBarrier<PIPE_ALL>();
      DataCopyExtParams ccp{1, sizeof(int64_t), 0, 0, 0};
      DataCopyPad(count_gm_[0], count_local_, ccp);
      return;
    }

    // Write hash table keys and values to GM.
    // Scalar hash-table writes must be visible before MTE3 reads them.
    AscendC::PipeBarrier<PIPE_ALL>();
    {
      uint32_t ht_bytes = hash_capacity_ * static_cast<uint32_t>(sizeof(IdT));
      DataCopyExtParams cp{1, ht_bytes, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(keys_out_gm_[0], keys_, cp);
      DataCopyPad(vals_out_gm_[0], vals_, cp);
    }

    // Write total unique count through a UB staging tensor (no
    // GlobalTensor::SetValue — R4).
    count_local_.SetValue(0, static_cast<int64_t>(next_local));
    AscendC::PipeBarrier<PIPE_ALL>();
    DataCopyExtParams ccp{1, sizeof(int64_t), 0, 0, 0};
    DataCopyPad(count_gm_[0], count_local_, ccp);
  }

 private:
  GlobalTensor<IdT> src_gm_, keys_out_gm_, vals_out_gm_, unique_gm_;
  GlobalTensor<int64_t> count_gm_;
  TQue<TPosition::VECIN, 2> stream_q_;
  TBuf<TPosition::VECCALC> keys_buf_, vals_buf_, staging_buf_, count_buf_;
  LocalTensor<IdT> keys_, vals_, staging_;
  LocalTensor<int64_t> count_local_;
  int64_t num_ids_ = 0, num_seeds_ = 0;
  uint32_t hash_capacity_ = 0, chunk_elems_ = 0;
  IdT hash_mask_ = 0;
};

// ---------------------------------------------------------------------------
// K3: map_edges (multi-core, block_dim=N)
// Loads hash tables into UB, streams edges in chunks, maps via quad probe.
// ---------------------------------------------------------------------------
template <typename IdT>
class KernelMapEdges {
 public:
  __aicore__ inline void Init(
      GM_ADDR edge_src_in, GM_ADDR edge_dst_in,
      GM_ADDR hash_keys_src, GM_ADDR hash_vals_src,
      GM_ADDR hash_keys_dst, GM_ADDR hash_vals_dst,
      GM_ADDR new_src_out, GM_ADDR new_dst_out,
      GM_ADDR tiling_ptr, TPipe* pipe) {

    const __gm__ ToBlockTiling* tiling =
        (const __gm__ ToBlockTiling*)tiling_ptr;
    num_edges_ = tiling->num_edges;
    hash_capacity_src_ = static_cast<uint32_t>(tiling->hash_capacity_src);
    hash_mask_src_ = static_cast<IdT>(tiling->hash_mask_src);
    hash_capacity_dst_ = static_cast<uint32_t>(tiling->hash_capacity_dst);
    hash_mask_dst_ = static_cast<IdT>(tiling->hash_mask_dst);
    const uint32_t block_dim = tiling->block_dim;
    const uint32_t block_idx = AscendC::GetBlockIdx();

    edge_src_gm_.SetGlobalBuffer((__gm__ IdT*)edge_src_in, num_edges_);
    edge_dst_gm_.SetGlobalBuffer((__gm__ IdT*)edge_dst_in, num_edges_);
    hash_keys_src_gm_.SetGlobalBuffer((__gm__ IdT*)hash_keys_src, hash_capacity_src_);
    hash_vals_src_gm_.SetGlobalBuffer((__gm__ IdT*)hash_vals_src, hash_capacity_src_);
    hash_keys_dst_gm_.SetGlobalBuffer((__gm__ IdT*)hash_keys_dst, hash_capacity_dst_);
    hash_vals_dst_gm_.SetGlobalBuffer((__gm__ IdT*)hash_vals_dst, hash_capacity_dst_);
    new_src_gm_.SetGlobalBuffer((__gm__ IdT*)new_src_out, num_edges_);
    new_dst_gm_.SetGlobalBuffer((__gm__ IdT*)new_dst_out, num_edges_);

    // Compute edge range for this block (contiguous split)
    const uint32_t epb = static_cast<uint32_t>(
        (num_edges_ + static_cast<int64_t>(block_dim) - 1) /
        static_cast<int64_t>(block_dim));
    edge_begin_ = block_idx * epb;
    edge_end_ = (block_idx + 1) * epb;
    if (edge_end_ > static_cast<uint32_t>(num_edges_)) edge_end_ = static_cast<uint32_t>(num_edges_);

    if (edge_begin_ >= edge_end_) {
      initialized_ = false;
      return;
    }

    const uint32_t ub_avail = tiling->ub_available;
    const uint32_t hash_src_bytes = hash_capacity_src_ * sizeof(IdT);
    const uint32_t hash_dst_bytes = hash_capacity_dst_ * sizeof(IdT);
    // R3 fix: K3 allocates FOUR hash arrays (src_keys + src_vals +
    // dst_keys + dst_vals), so the hash total must count both keys and
    // values arrays for each side.
    const uint32_t hash_total = 2 * (hash_src_bytes + hash_dst_bytes);
    // Buffer instances: 2 (src stream db) + 2 (dst stream db) + 1 (out_src)
    // + 1 (out_dst) = 6 chunk-sized instances.
    const uint32_t avail = (ub_avail > hash_total) ? (ub_avail - hash_total) : 0;
    chunk_elems_ = avail / (6 * sizeof(IdT));
    if (chunk_elems_ == 0) chunk_elems_ = 64;
    if (chunk_elems_ > 65536) chunk_elems_ = 65536;

    pipe->InitBuffer(stream_src_q_, 2, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(stream_dst_q_, 2, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(src_ht_keys_buf_, hash_src_bytes);
    pipe->InitBuffer(src_ht_vals_buf_, hash_src_bytes);
    pipe->InitBuffer(dst_ht_keys_buf_, hash_dst_bytes);
    pipe->InitBuffer(dst_ht_vals_buf_, hash_dst_bytes);
    pipe->InitBuffer(out_src_buf_, chunk_elems_ * sizeof(IdT));
    pipe->InitBuffer(out_dst_buf_, chunk_elems_ * sizeof(IdT));

    src_ht_keys_ = src_ht_keys_buf_.Get<IdT>();
    src_ht_vals_ = src_ht_vals_buf_.Get<IdT>();
    dst_ht_keys_ = dst_ht_keys_buf_.Get<IdT>();
    dst_ht_vals_ = dst_ht_vals_buf_.Get<IdT>();
    out_src_ = out_src_buf_.Get<IdT>();
    out_dst_ = out_dst_buf_.Get<IdT>();

    // Preload hash tables into UB
    {
      uint32_t sb = hash_src_bytes;
      DataCopyExtParams cp{1, sb, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(src_ht_keys_, hash_keys_src_gm_[0], cp, pad);
    }
    {
      uint32_t sb = hash_src_bytes;
      DataCopyExtParams cp{1, sb, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(src_ht_vals_, hash_vals_src_gm_[0], cp, pad);
    }
    {
      uint32_t sb = hash_dst_bytes;
      DataCopyExtParams cp{1, sb, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(dst_ht_keys_, hash_keys_dst_gm_[0], cp, pad);
    }
    {
      uint32_t sb = hash_dst_bytes;
      DataCopyExtParams cp{1, sb, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(dst_ht_vals_, hash_vals_dst_gm_[0], cp, pad);
    }

    // MTE2/MTE3 hash-table loads must be visible before the Scalar pipe
    // reads them in Process() (R1).
    AscendC::PipeBarrier<PIPE_ALL>();

    initialized_ = true;
  }

  __aicore__ inline void Process() {
    if (!initialized_) return;

    uint32_t processed = 0;
    const uint32_t total = edge_end_ - edge_begin_;

    while (processed < total) {
      uint32_t cnt = (total - processed < chunk_elems_) ? (total - processed) : chunk_elems_;
      uint32_t copy_bytes = cnt * static_cast<uint32_t>(sizeof(IdT));

      // Stream edge src into UB
      LocalTensor<IdT> sb = stream_src_q_.AllocTensor<IdT>();
      DataCopyExtParams cp{1, copy_bytes, 0, 0, 0};
      DataCopyPadExtParams<IdT> pad{false, 0, 0, 0};
      DataCopyPad(sb, edge_src_gm_[static_cast<int64_t>(edge_begin_) + static_cast<int64_t>(processed)], cp, pad);
      stream_src_q_.EnQue(sb);
      sb = stream_src_q_.DeQue<IdT>();

      // Stream edge dst into UB
      LocalTensor<IdT> db = stream_dst_q_.AllocTensor<IdT>();
      DataCopyPad(db, edge_dst_gm_[static_cast<int64_t>(edge_begin_) + static_cast<int64_t>(processed)], cp, pad);
      stream_dst_q_.EnQue(db);
      db = stream_dst_q_.DeQue<IdT>();

      // Map edges via hash lookup, write to staging
      for (uint32_t i = 0; i < cnt; ++i) {
        IdT src_id = sb.GetValue(i);
        IdT dst_id = db.GetValue(i);

        IdT new_s = Lookup(src_id, src_ht_keys_, src_ht_vals_,
                           hash_mask_src_);
        IdT new_d = Lookup(dst_id, dst_ht_keys_, dst_ht_vals_,
                           hash_mask_dst_);

        out_src_.SetValue(i, new_s);
        out_dst_.SetValue(i, new_d);
      }

      // Scalar out-buffer writes must be visible before MTE3 reads them (R1).
      AscendC::PipeBarrier<PIPE_ALL>();

      // Write staging to GM
      {
        DataCopyExtParams ocp{1, copy_bytes, 0, 0, 0};
        DataCopyPad(new_src_gm_[static_cast<int64_t>(edge_begin_) + static_cast<int64_t>(processed)], out_src_, ocp);
      }
      {
        DataCopyExtParams ocp{1, copy_bytes, 0, 0, 0};
        DataCopyPad(new_dst_gm_[static_cast<int64_t>(edge_begin_) + static_cast<int64_t>(processed)], out_dst_, ocp);
      }

      stream_src_q_.FreeTensor(sb);
      stream_dst_q_.FreeTensor(db);
      processed += cnt;
    }
  }

 private:
  // Quadratic probe: search for key in hash table. Returns (IdT)-1 if not found.
  __aicore__ inline IdT Lookup(
      IdT key, LocalTensor<IdT>& ht_keys, LocalTensor<IdT>& ht_vals,
      IdT mask) {
    uint32_t pos = static_cast<uint32_t>(key & mask);
    uint32_t delta = 1;
    while (true) {
      IdT existing = ht_keys.GetValue(pos);
      if (existing == static_cast<IdT>(kToBlockEmptyKey)) {
        return static_cast<IdT>(-1);
      } else if (existing == key) {
        return ht_vals.GetValue(pos);
      }
      pos = static_cast<uint32_t>(
          (static_cast<uint64_t>(pos) +
           static_cast<uint64_t>(delta) * static_cast<uint64_t>(delta)) &
          static_cast<uint64_t>(mask));
      ++delta;
    }
  }

  GlobalTensor<IdT> edge_src_gm_, edge_dst_gm_;
  GlobalTensor<IdT> hash_keys_src_gm_, hash_vals_src_gm_;
  GlobalTensor<IdT> hash_keys_dst_gm_, hash_vals_dst_gm_;
  GlobalTensor<IdT> new_src_gm_, new_dst_gm_;

  TQue<TPosition::VECIN, 2> stream_src_q_, stream_dst_q_;
  TBuf<TPosition::VECCALC> src_ht_keys_buf_, src_ht_vals_buf_;
  TBuf<TPosition::VECCALC> dst_ht_keys_buf_, dst_ht_vals_buf_;
  TBuf<TPosition::VECCALC> out_src_buf_, out_dst_buf_;
  LocalTensor<IdT> src_ht_keys_, src_ht_vals_;
  LocalTensor<IdT> dst_ht_keys_, dst_ht_vals_;
  LocalTensor<IdT> out_src_, out_dst_;

  int64_t num_edges_ = 0;
  uint32_t hash_capacity_src_ = 0, hash_capacity_dst_ = 0;
  uint32_t chunk_elems_ = 0;
  uint32_t edge_begin_ = 0, edge_end_ = 0;
  IdT hash_mask_src_ = 0, hash_mask_dst_ = 0;
  bool initialized_ = false;
};

}  // namespace

// Single kernel entry: branch on op_type and dtype_bytes
extern "C" __global__ __aicore__ void to_block_kernel(
    GM_ADDR input1,      // K1: src_nodes; K3: edge_src
    GM_ADDR input2,      // K1: unused; K3: edge_dst
    GM_ADDR hash_arg1,   // K1: keys_out; K3: hash_keys_src
    GM_ADDR hash_arg2,   // K1: vals_out; K3: hash_vals_src
    GM_ADDR hash_arg3,   // K1: unique_out; K3: hash_keys_dst
    GM_ADDR hash_arg4,   // K1: count_out; K3: hash_vals_dst
    GM_ADDR output1,     // K1: unused; K3: new_src_out
    GM_ADDR output2,     // K1: unused; K3: new_dst_out
    GM_ADDR tiling_ptr) {

  const __gm__ ToBlockTiling* tiling =
      (const __gm__ ToBlockTiling*)tiling_ptr;

  if (tiling->op_type == 0) {
    // dedup_hash_build
    TPipe pipe;
    if (tiling->dtype_bytes == 4) {
      KernelDedupHashBuild<int32_t> op;
      op.Init(input1, hash_arg1, hash_arg2, hash_arg3, hash_arg4,
              tiling_ptr, &pipe);
      op.Process();
    } else {
      KernelDedupHashBuild<int64_t> op;
      op.Init(input1, hash_arg1, hash_arg2, hash_arg3, hash_arg4,
              tiling_ptr, &pipe);
      op.Process();
    }
  } else if (tiling->op_type == 1) {
    // map_edges
    TPipe pipe;
    if (tiling->dtype_bytes == 4) {
      KernelMapEdges<int32_t> op;
      op.Init(input1, input2, hash_arg1, hash_arg2, hash_arg3, hash_arg4,
              output1, output2, tiling_ptr, &pipe);
      op.Process();
    } else {
      KernelMapEdges<int64_t> op;
      op.Init(input1, input2, hash_arg1, hash_arg2, hash_arg3, hash_arg4,
              output1, output2, tiling_ptr, &pipe);
      op.Process();
    }
  }
}