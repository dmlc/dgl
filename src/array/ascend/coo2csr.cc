/**
 * Copyright (c) 2024 by Contributors
 * @file coo2csr.cc
 * @brief Ascend NPU implementation of COO -> CSR conversion.
 *
 * Counting sort on the NPU (ADR-0006): two kernel launches per row band
 * with a host-side exclusive scan in between (ADR-0005) —
 *
 *   count kernel:   per-block row histograms + min/max reductions
 *   host:           exclusive scan of counts -> indptr
 *   scatter kernel: stable scatter of col/data in global edge order
 *
 * A preprocess cache (ADR-0004) keyed on the input tensors' identity
 * returns the previously computed CSR for repeated conversions of the
 * same graph — the common case in training loops. Cache entries hold
 * references to the input arrays, so a freed tensor's address can never
 * be mistaken for a new tensor's (allocator address reuse is safe), and
 * an LRU byte budget bounds memory held by dead graphs.
 */

#include <dgl/array.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../array_op.h"

// The large-row fallback calls the CPU reference implementation. Its
// template is only declared in array_op.h; the int32/int64 kDGLCPU
// instantiations live in src/array/cpu/spmat_op_impl_coo.cc, so declare
// them extern to link against those symbols instead of instantiating
// the (undefined) template body here.
namespace dgl {
namespace aten {
namespace impl {
extern template CSRMatrix COOToCSR<kDGLCPU, int32_t>(COOMatrix coo);
extern template CSRMatrix COOToCSR<kDGLCPU, int64_t>(COOMatrix coo);
}  // namespace impl
}  // namespace aten
}  // namespace dgl

#ifdef DGL_USE_ASCEND
#include <acl/acl.h>
#include <acl/acl_rt.h>

#include "coo2csr_tiling.h"

#define ASCEND_CALL(func)                                   \
  {                                                         \
    aclError e = (func);                                    \
    CHECK(e == ACL_SUCCESS) << "Ascend Error, code: " << e; \
  }

namespace dgl {
namespace runtime {
aclrtStream getCurrentAscendStream();
}  // namespace runtime
}  // namespace dgl

extern "C" uint32_t aclrtlaunch_coo_to_csr_count_int32(
    uint32_t blockDim, aclrtStream stream, void* rows, void* counts,
    void* reduces, void* row_split, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_to_csr_count_int64(
    uint32_t blockDim, aclrtStream stream, void* rows, void* counts,
    void* reduces, void* row_split, void* tiling);

extern "C" uint32_t aclrtlaunch_coo_to_csr_scatter_int32(
    uint32_t blockDim, aclrtStream stream, void* rows, void* cols, void* data,
    void* indptr, void* out_indices, void* out_data, void* row_split,
    void* tiling);

extern "C" uint32_t aclrtlaunch_coo_to_csr_scatter_int64(
    uint32_t blockDim, aclrtStream stream, void* rows, void* cols, void* data,
    void* indptr, void* out_indices, void* out_data, void* row_split,
    void* tiling);

namespace dgl {
namespace aten {
namespace impl {
namespace {

// ---------------------------------------------------------------------------
// Preprocess cache (ADR-0004).
// ---------------------------------------------------------------------------

struct CooToCsrCacheKey {
  int device_id;
  const void* row_ptr;
  const void* col_ptr;
  const void* data_ptr;
  int64_t num_rows;
  int64_t num_cols;
  int64_t nnz;
  uint8_t id_bits;
  bool has_data;
  bool row_sorted;
  bool col_sorted;

  bool operator==(const CooToCsrCacheKey& other) const {
    return device_id == other.device_id && row_ptr == other.row_ptr &&
           col_ptr == other.col_ptr && data_ptr == other.data_ptr &&
           num_rows == other.num_rows && num_cols == other.num_cols &&
           nnz == other.nnz && id_bits == other.id_bits &&
           has_data == other.has_data && row_sorted == other.row_sorted &&
           col_sorted == other.col_sorted;
  }
};

struct CooToCsrCacheKeyHash {
  size_t operator()(const CooToCsrCacheKey& k) const {
    size_t h = std::hash<int>{}(k.device_id);
    auto combine = [&h](size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    combine(std::hash<const void*>{}(k.row_ptr));
    combine(std::hash<const void*>{}(k.col_ptr));
    combine(std::hash<const void*>{}(k.data_ptr));
    combine(std::hash<int64_t>{}(k.num_rows));
    combine(std::hash<int64_t>{}(k.num_cols));
    combine(std::hash<int64_t>{}(k.nnz));
    combine(std::hash<uint8_t>{}(k.id_bits));
    combine(k.has_data ? 1 : 0);
    combine(k.row_sorted ? 2 : 0);
    combine(k.col_sorted ? 4 : 0);
    return h;
  }
};

struct CooToCsrCacheValue {
  CSRMatrix csr;           // holds indptr/indices/data output arrays
  NDArray row, col, data;  // input retention: pins the key's identity
  uint64_t last_used = 0;
  uint64_t bytes = 0;
};

// LRU byte budget for entries whose inputs are otherwise dead. The
// budget is deliberately modest relative to HBM (2GB on 64GB parts).
constexpr uint64_t kCacheBudgetBytes = 2ULL << 30;

// Row-count threshold above which the conversion routes to the CPU
// reference implementation. The counting-sort kernels rescan the whole
// edge array once per row band (~47k rows/band on a 192KB-UB part);
// past ~1M rows the band multiplication outweighs the NPU advantage
// (multi-scale matrix, 910B3: NPU wins through 1M rows, loses by 10M).
constexpr int64_t kNpuRowCountLimit = 1'000'000;

uint64_t NextCacheClock() {
  static std::atomic<uint64_t> clock{1};
  return clock.fetch_add(1, std::memory_order_relaxed);
}

// The cache is process-lifetime state. Its entries own NDArrays whose
// destructors call into the ACL runtime; at process exit, static
// destruction order is not guaranteed relative to the runtime's own
// singletons, which produced "pure virtual method called" aborts in
// spawned workers. Deliberately leak the containers (never destroyed);
// the OS reclaims everything at exit.
auto& CacheMap() {
  static auto* map = new std::unordered_map<
      CooToCsrCacheKey, std::shared_ptr<CooToCsrCacheValue>,
      CooToCsrCacheKeyHash>();
  return *map;
}
auto& CacheMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

// Called with the mutex held. Device pointers and allocator state are
// process-specific: a fork()ed child (distributed-training workers fork
// after the parent has already converted graphs) inherits cache entries
// whose addresses are invalid in its address space. Detect the PID
// change on entry and drop the whole table.
void ResetCacheIfForked() {
  static std::atomic<pid_t> owner{0};
  const pid_t current = getpid();
  if (owner.load(std::memory_order_relaxed) != current) {
    CacheMap().clear();
    owner.store(current, std::memory_order_relaxed);
  }
}

uint64_t EstimateEntryBytes(const COOMatrix& coo, const CSRMatrix& csr) {
  const auto arr_bytes = [](const NDArray& a) {
    return a.defined() ? static_cast<uint64_t>(a.GetSize()) : 0;
  };
  // Inputs shared with a live graph are ~free; charging the full size
  // keeps the budget conservative for orphaned entries.
  return arr_bytes(coo.row) + arr_bytes(coo.col) + arr_bytes(coo.data) +
         arr_bytes(csr.indptr) + arr_bytes(csr.indices) + arr_bytes(csr.data);
}

// Evicts LRU entries until the budget is met. Called with the mutex held.
void EvictCacheLocked(uint64_t budget) {
  uint64_t total = std::accumulate(
      CacheMap().begin(), CacheMap().end(), uint64_t{0},
      [](uint64_t sum, const auto& kv) { return sum + kv.second->bytes; });
  while (total > budget && !CacheMap().empty()) {
    const auto victim = std::min_element(
        CacheMap().begin(), CacheMap().end(), [](const auto& a, const auto& b) {
          return a.second->last_used < b.second->last_used;
        });
    total -= victim->second->bytes;
    LOG(INFO) << "[Ascend][COOToCSR][Cache] evicting entry ("
              << victim->second->bytes << " bytes, LRU)";
    CacheMap().erase(victim);
  }
}

bool CacheDisabled() {
  static const bool disabled = []() {
    const char* env = std::getenv("DGL_COO2CSR_CACHE_DISABLE");
    return env != nullptr && env[0] == '1';
  }();
  return disabled;
}

// ---------------------------------------------------------------------------
// Counting-sort conversion (no cache).
// ---------------------------------------------------------------------------

uint32_t QueryVectorCoreCount(int device_id) {
  int64_t core_num = 0;
  aclError err =
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &core_num);
  if (err != ACL_SUCCESS || core_num <= 0 || core_num > 4096) {
    return kDefaultVectorCoreCount;
  }
  return static_cast<uint32_t>(core_num);
}

uint32_t QueryUbAvailableBytes(int device_id) {
  int64_t ub_bytes = 0;
  aclError err = aclrtGetDeviceInfo(
      device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, &ub_bytes);
  if (err != ACL_SUCCESS ||
      ub_bytes <= static_cast<int64_t>(kUbReservedBytes) ||
      ub_bytes > (1 << 30)) {
    return kDefaultUbBytes - kUbReservedBytes;
  }
  return static_cast<uint32_t>(ub_bytes - kUbReservedBytes);
}

// Device workspaces shared across bands of one conversion.
struct BandWorkspaces {
  uint32_t* counts_dev = nullptr;
  uint32_t* reduces_dev = nullptr;
  uint32_t* row_split_dev = nullptr;
  CooToCsrTiling* tiling_dev = nullptr;

  void Allocate(int64_t num_rows, uint32_t block_dim) {
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&counts_dev), num_rows * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&reduces_dev),
        block_dim * kReduceWordsPerBlock * sizeof(uint32_t),
        ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&row_split_dev),
        (block_dim + 1) * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ASCEND_CALL(aclrtMalloc(
        reinterpret_cast<void**>(&tiling_dev), sizeof(CooToCsrTiling),
        ACL_MEM_MALLOC_HUGE_FIRST));
  }
  void Free() {
    if (counts_dev) ASCEND_CALL(aclrtFree(counts_dev));
    if (reduces_dev) ASCEND_CALL(aclrtFree(reduces_dev));
    if (row_split_dev) ASCEND_CALL(aclrtFree(row_split_dev));
    if (tiling_dev) ASCEND_CALL(aclrtFree(tiling_dev));
    counts_dev = reduces_dev = row_split_dev = nullptr;
    tiling_dev = nullptr;
  }
  // RAII backstop: ~10 throwing sites (ASCEND_CALL / CHECK) sit between
  // Allocate and the explicit Free on the success path; dmlc CHECK throws
  // in default DGL builds, so any of them would otherwise leak all four
  // device buffers. A partial-Allocate failure is covered too: already
  // allocated members are non-null here.
  ~BandWorkspaces() { Free(); }
  BandWorkspaces() = default;
  BandWorkspaces(const BandWorkspaces&) = delete;
  BandWorkspaces& operator=(const BandWorkspaces&) = delete;
};

// Even row-range split of one band across active blocks. row_split has
// active_blocks+1 entries; histogram weights are unknown before counting,
// so the split is by row count.
std::vector<uint32_t> BuildBandRowSplit(
    int64_t band_begin, int64_t band_end, uint32_t active_blocks) {
  const int64_t band_rows = band_end - band_begin;
  std::vector<uint32_t> split(active_blocks + 1, 0);
  for (uint32_t b = 0; b <= active_blocks; ++b) {
    const int64_t pos =
        band_begin + (band_rows * b + active_blocks - 1) / active_blocks;
    split[b] = static_cast<uint32_t>(std::min<int64_t>(pos, band_end));
  }
  return split;
}

template <typename IdType>
void LaunchCountKernel(
    uint32_t block_dim, aclrtStream stream, const void* rows,
    uint32_t* counts_dev, uint32_t* reduces_dev, uint32_t* row_split_dev,
    CooToCsrTiling* tiling_dev) {
  uint32_t rc = 0;
  if (std::is_same<IdType, int32_t>::value) {
    rc = aclrtlaunch_coo_to_csr_count_int32(
        block_dim, stream, const_cast<void*>(rows), counts_dev, reduces_dev,
        row_split_dev, tiling_dev);
  } else {
    rc = aclrtlaunch_coo_to_csr_count_int64(
        block_dim, stream, const_cast<void*>(rows), counts_dev, reduces_dev,
        row_split_dev, tiling_dev);
  }
  CHECK(rc == ACL_SUCCESS) << "coo_to_csr_count launch failed: " << rc;
}

template <typename IdType>
void LaunchScatterKernel(
    uint32_t block_dim, aclrtStream stream, const void* rows, const void* cols,
    const void* data, void* indptr, void* out_indices, void* out_data,
    uint32_t* row_split_dev, CooToCsrTiling* tiling_dev) {
  uint32_t rc = 0;
  if (std::is_same<IdType, int32_t>::value) {
    rc = aclrtlaunch_coo_to_csr_scatter_int32(
        block_dim, stream, const_cast<void*>(rows), const_cast<void*>(cols),
        const_cast<void*>(data), indptr, out_indices, out_data, row_split_dev,
        tiling_dev);
  } else {
    rc = aclrtlaunch_coo_to_csr_scatter_int64(
        block_dim, stream, const_cast<void*>(rows), const_cast<void*>(cols),
        const_cast<void*>(data), indptr, out_indices, out_data, row_split_dev,
        tiling_dev);
  }
  CHECK(rc == ACL_SUCCESS) << "coo_to_csr_scatter launch failed: " << rc;
}

// Runs the count kernel for every band and accumulates the exclusive
// scan into indptr_vec (which receives num_rows+1 entries total).
// Verifies the per-block min/max tripwires per band; the authoritative
// out-of-range check (total == nnz) is the caller's.
template <typename IdType>
void CountAllBands(
    const COOMatrix& coo, BandWorkspaces& ws, uint32_t block_dim,
    aclrtStream stream, uint32_t rows_per_band, uint64_t num_bands,
    int64_t num_rows, std::vector<IdType>* indptr_vec) {
  for (uint64_t band = 0; band < num_bands; ++band) {
    const int64_t band_begin = band * rows_per_band;
    const int64_t band_end =
        std::min<int64_t>(band_begin + rows_per_band, num_rows);
    const uint32_t band_rows = static_cast<uint32_t>(band_end - band_begin);
    const uint32_t active_blocks = std::min<uint32_t>(block_dim, band_rows);
    // Upload the FULL table (block_dim+1 entries): idle blocks beyond
    // active_blocks read their own pair to see an empty range and exit.
    std::vector<uint32_t> row_split_host =
        BuildBandRowSplit(band_begin, band_end, active_blocks);
    row_split_host.resize(block_dim + 1, row_split_host.back());
    ASCEND_CALL(aclrtMemcpy(
        ws.row_split_dev, (block_dim + 1) * sizeof(uint32_t),
        row_split_host.data(), (block_dim + 1) * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchCountKernel<IdType>(
        block_dim, stream, coo.row->data, ws.counts_dev, ws.reduces_dev,
        ws.row_split_dev, ws.tiling_dev);
    ASCEND_CALL(aclrtSynchronizeStream(stream));

    std::vector<uint32_t> counts_host(band_rows);
    ASCEND_CALL(aclrtMemcpy(
        counts_host.data(), band_rows * sizeof(uint32_t),
        ws.counts_dev + band_begin, band_rows * sizeof(uint32_t),
        ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<uint32_t> reduces_host(
        static_cast<size_t>(block_dim) * kReduceWordsPerBlock);
    ASCEND_CALL(aclrtMemcpy(
        reduces_host.data(),
        static_cast<size_t>(block_dim) * kReduceWordsPerBlock *
            sizeof(uint32_t),
        ws.reduces_dev,
        static_cast<size_t>(block_dim) * kReduceWordsPerBlock *
            sizeof(uint32_t),
        ACL_MEMCPY_DEVICE_TO_HOST));

    // A block whose observed [min, max] escapes its assigned range means
    // the kernel's filter logic broke.
    for (uint32_t b = 0; b < active_blocks; ++b) {
      const uint32_t lo = reduces_host[b * kReduceWordsPerBlock + 0];
      const uint32_t hi = reduces_host[b * kReduceWordsPerBlock + 1];
      if (lo == kReduceEmpty) continue;
      CHECK(lo >= row_split_host[b] && hi < row_split_host[b + 1])
          << "COOToCSR: count kernel range invariant violated";
    }

    if (band == 0) indptr_vec->push_back(0);
    for (uint32_t r = 0; r < band_rows; ++r) {
      indptr_vec->push_back(indptr_vec->back() + counts_host[r]);
    }
  }
}

// CPU fallback for graphs with row counts beyond the counting-sort
// kernels' sweet spot (see kNpuRowCountLimit): copy the COO to host,
// run the reference CPU implementation, copy the CSR back.
template <typename IdType>
CSRMatrix COOToCSRViaCpu(const COOMatrix& coo) {
  const DGLContext ctx = coo.row->ctx;
  const DGLContext cpu_ctx = DGLContext{kDGLCPU, 0};
  COOMatrix coo_cpu = coo;
  coo_cpu.row = coo.row.CopyTo(cpu_ctx);
  coo_cpu.col = coo.col.CopyTo(cpu_ctx);
  if (aten::COOHasData(coo)) coo_cpu.data = coo.data.CopyTo(cpu_ctx);
  coo_cpu.is_pinned = false;
  const CSRMatrix csr_cpu = impl::COOToCSR<kDGLCPU, IdType>(coo_cpu);
  CSRMatrix csr = csr_cpu;
  csr.indptr = csr_cpu.indptr.CopyTo(ctx);
  csr.indices = csr_cpu.indices.CopyTo(ctx);
  csr.data = csr_cpu.data.CopyTo(ctx);
  return csr;
}

template <typename IdType>
CSRMatrix COOToCSRCountingSort(const COOMatrix& coo) {
  const DGLContext ctx = coo.row->ctx;
  const int64_t nnz = coo.row->shape[0];
  const int64_t num_rows = coo.num_rows;
  CHECK_NO_OVERFLOW(coo.row->dtype, nnz);

  // Large-row fallback. The counting-sort kernels iterate once per row
  // band over the whole edge array; with num_rows beyond ~1M the band
  // count (num_rows / ~47k) multiplies the scan work until the scalar
  // histogram pass is slower than the CPU reference (measured: 10M rows /
  // 50M edges took 310s on NPU vs 133s on the old CPU path — a true-cold
  // regression hidden until the multi-scale matrix ran with fresh tensors
  // per rep). A2 has no vector-scatter-atomic API to vectorize the
  // histogram (SetAtomicAdd only sums contiguous DataCopy segments), so
  // the honest fix is routing giant graphs to the CPU implementation.
  if (num_rows > kNpuRowCountLimit) {
    return COOToCSRViaCpu<IdType>(coo);
  }

  const uint32_t block_dim = QueryVectorCoreCount(ctx.device_id);
  const uint32_t ub_available = QueryUbAvailableBytes(ctx.device_id);
  aclrtStream stream = dgl::runtime::getCurrentAscendStream();
  ASCEND_CALL(aclrtSetDevice(ctx.device_id));

  NDArray indptr = NDArray::Empty({num_rows + 1}, coo.row->dtype, ctx);
  NDArray indices = NDArray::Empty({nnz}, coo.col->dtype, ctx);
  // The CSR data array is always freshly allocated: the scatter kernel
  // streams the input col/data arrays chunk-by-chunk while writing the
  // reordered output, so aliasing the input data buffer (as the input
  // row-major order differs from the output CSR order for unsorted COO)
  // would let already-written regions overwrite input chunks not yet
  // read. The dtype matches the CPU path's ret_data (id dtype; the CPU
  // reference reads/writes data through IdType pointers as well).
  const bool has_data = aten::COOHasData(coo);
  NDArray data = NDArray::Empty({nnz}, coo.row->dtype, ctx);

  if (nnz == 0) {
    ASCEND_CALL(aclrtMemset(
        indptr->data, (num_rows + 1) * sizeof(IdType), 0,
        (num_rows + 1) * sizeof(IdType)));
    return CSRMatrix(
        num_rows, coo.num_cols, indptr, indices, data, coo.col_sorted);
  }

  // Row-band sizing: one band must fit (band_rows histogram words + the
  // stream chunk minimum) in a block's UB, and the cursor array of the
  // scatter pass (band_rows * sizeof(IdType)) likewise.
  CHECK(num_rows <= 0xFFFFFFFFLL)
      << "COOToCSR: num_rows " << num_rows
      << " exceeds uint32 addressing used by the counting-sort kernels";
  const uint32_t min_chunk_elems = 16;  // keep DMA efficiency sane
  const uint32_t hist_guard =
      ub_available - 2 * min_chunk_elems * sizeof(IdType);
  const uint32_t max_rows_per_band =
      std::max<uint32_t>(hist_guard / sizeof(uint32_t), 1);
  const uint32_t rows_per_band =
      std::min<uint32_t>(static_cast<uint32_t>(num_rows), max_rows_per_band);
  const uint64_t num_bands =
      (static_cast<uint64_t>(num_rows) + rows_per_band - 1) / rows_per_band;

  BandWorkspaces ws;
  ws.Allocate(num_rows, block_dim);

  CooToCsrTiling tiling_host;
  tiling_host.nnz = nnz;
  tiling_host.num_rows = static_cast<uint32_t>(num_rows);
  tiling_host.ub_available = ub_available;
  tiling_host.has_data = has_data ? 1 : 0;
  ASCEND_CALL(aclrtMemcpy(
      ws.tiling_dev, sizeof(CooToCsrTiling), &tiling_host,
      sizeof(CooToCsrTiling), ACL_MEMCPY_HOST_TO_DEVICE));

  std::vector<IdType> indptr_vec;
  indptr_vec.reserve(static_cast<size_t>(num_rows) + 1);

  CountAllBands<IdType>(
      coo, ws, block_dim, stream, rows_per_band, num_bands, num_rows,
      &indptr_vec);

  // All bands counted: total must be exactly nnz (defense against
  // out-of-range or negative row ids, which no block would have counted).
  CHECK(static_cast<int64_t>(indptr_vec.back()) == nnz)
      << "COOToCSR: row ids outside [0, num_rows) detected " << "(counted "
      << indptr_vec.back() << " of " << nnz << " edges)";

  ASCEND_CALL(aclrtMemcpy(
      indptr->data, (num_rows + 1) * sizeof(IdType), indptr_vec.data(),
      (num_rows + 1) * sizeof(IdType), ACL_MEMCPY_HOST_TO_DEVICE));

  for (uint64_t band = 0; band < num_bands; ++band) {
    const int64_t band_begin = band * rows_per_band;
    const int64_t band_end =
        std::min<int64_t>(band_begin + rows_per_band, num_rows);
    const uint32_t band_rows = static_cast<uint32_t>(band_end - band_begin);
    const uint32_t active_blocks = std::min<uint32_t>(block_dim, band_rows);
    // Full-table upload for idle-block early exit (see count pass).
    std::vector<uint32_t> row_split_host =
        BuildBandRowSplit(band_begin, band_end, active_blocks);
    row_split_host.resize(block_dim + 1, row_split_host.back());
    ASCEND_CALL(aclrtMemcpy(
        ws.row_split_dev, (block_dim + 1) * sizeof(uint32_t),
        row_split_host.data(), (block_dim + 1) * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchScatterKernel<IdType>(
        block_dim, stream, coo.row->data, coo.col->data,
        has_data ? coo.data->data : nullptr, indptr->data, indices->data,
        data->data, ws.row_split_dev, ws.tiling_dev);
    ASCEND_CALL(aclrtSynchronizeStream(stream));
  }

  ws.Free();

  // col_sorted propagates from the input only when rows were already
  // sorted (no reordering happened relative to CPU semantics); the
  // counting sort itself never sorts within a row.
  const bool col_sorted = coo.row_sorted ? coo.col_sorted : false;
  return CSRMatrix(num_rows, coo.num_cols, indptr, indices, data, col_sorted);
}

}  // anonymous namespace

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  CHECK(coo.row->ctx.device_type == kDGLAscend)
      << "Expected Ascend device context";
  ASCEND_CALL(aclrtSetDevice(coo.row->ctx.device_id));

  if (!CacheDisabled()) {
    CooToCsrCacheKey key{coo.row->ctx.device_id,
                         coo.row->data,
                         coo.col->data,
                         aten::COOHasData(coo) ? coo.data->data : nullptr,
                         coo.num_rows,
                         coo.num_cols,
                         coo.row->shape[0],
                         coo.row->dtype.bits,
                         aten::COOHasData(coo),
                         coo.row_sorted,
                         coo.col_sorted};
    {
      std::lock_guard<std::mutex> lock(CacheMutex());
      ResetCacheIfForked();
      auto it = CacheMap().find(key);
      if (it != CacheMap().end()) {
        it->second->last_used = NextCacheClock();
        return it->second->csr;
      }
    }

    CSRMatrix csr = COOToCSRCountingSort<IdType>(coo);

    auto value = std::make_shared<CooToCsrCacheValue>();
    value->csr = csr;
    value->row = coo.row;
    value->col = coo.col;
    if (aten::COOHasData(coo)) value->data = coo.data;
    value->last_used = NextCacheClock();
    value->bytes = EstimateEntryBytes(coo, csr);
    if (value->bytes <= kCacheBudgetBytes) {
      std::lock_guard<std::mutex> lock(CacheMutex());
      ResetCacheIfForked();
      auto [it, inserted] = CacheMap().emplace(key, value);
      if (inserted) {
        EvictCacheLocked(kCacheBudgetBytes);
      } else {
        it->second->last_used = value->last_used;  // concurrent duplicate
      }
    }
    return csr;
  }
  return COOToCSRCountingSort<IdType>(coo);
}

template CSRMatrix COOToCSR<kDGLAscend, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDGLAscend, int64_t>(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#else  // !DGL_USE_ASCEND

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo) {
  LOG(FATAL) << "Ascend support is not compiled.";
  return {};
}

template CSRMatrix COOToCSR<kDGLAscend, int32_t>(COOMatrix coo);
template CSRMatrix COOToCSR<kDGLAscend, int64_t>(COOMatrix coo);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_USE_ASCEND
