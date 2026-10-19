#ifndef TEST_UNARY_ELEWISE_TEST_UTIL_H_
#define TEST_UNARY_ELEWISE_TEST_UTIL_H_

// UnaryElewise (arith::Neg, aten::Neg) test utilities.
//
// Authoritative precision assertion (single source of truth):
//   output   : tensor y (int32 / int64)
//   metric   : element-wise bit-exact (got[i] == expect[i] for all i), plus
//              mandatory meta checks shape=[n], dtype.bits in {32, 64},
//              ctx = kDGLAscend
//   threshold: mismatch count == 0 (any element mismatch fails the case)
//   verdict  : every case prints one "[UNARY-ELEWISE-VERDICT]" record line
//              (tensor / metric / measured / threshold / verdict) to stdout;
//              persist with `2>&1 | tee <log>` (auditable).
//
// Golden: host-side two's-complement negation via the unsigned equivalent
//   expect[i] = static_cast<T>(0 - static_cast<UnsignedT>(x[i]))
// bit-identical to the CPU reference arith::Neg::Call (`-t1` semantics,
// src/array/cpu/array_op_impl.cc) including INT_MIN wrap-around, while
// avoiding signed-negation UB.

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/ndarray.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "./common.h"

namespace unary_elewise_test {

using dgl::IdArray;
using dgl::runtime::NDArray;

// ---------------------------------------------------------------------------
// Golden: two's-complement negation, unsigned equivalent (INT_MIN wraps to
// itself), no signed-negation UB.
// ---------------------------------------------------------------------------
template <typename T>
inline T NegGolden(T x) {
  using U = typename std::make_unsigned<T>::type;
  return static_cast<T>(static_cast<U>(0) - static_cast<U>(x));
}

template <typename T>
inline std::vector<T> NegGoldenVec(const std::vector<T>& x) {
  std::vector<T> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = NegGolden(x[i]);
  return y;
}

// ---------------------------------------------------------------------------
// Full-value-domain random fill with a fixed seed. Values are injected as
// raw unsigned bit patterns, so extreme bit patterns (INT_MIN / INT_MAX /
// all-ones) occur naturally and generation is platform-independent
// (no dependence on distribution implementations).
// ---------------------------------------------------------------------------
inline uint64_t SeedFrom(const std::string& case_id) {
  // FNV-1a 64-bit over the case id: deterministic, documented per case.
  uint64_t h = 1469598103934665603ULL;
  for (char c : case_id) {
    h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
    h *= 1099511628211ULL;
  }
  return h;
}

template <typename T>
inline std::vector<T> RandomValues(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<T> v(n);
  for (size_t i = 0; i < n; ++i) {
    uint64_t bits = rng();
    if (sizeof(T) < sizeof(bits)) {
      bits &= (uint64_t{1} << (8 * sizeof(T))) - 1;
    }
    std::memcpy(&v[i], &bits, sizeof(T));
  }
  return v;
}

// ---------------------------------------------------------------------------
// Array builders (host fill -> CopyTo kDGLAscend).
// ---------------------------------------------------------------------------
template <typename T>
inline IdArray MakeNpuArray(const std::vector<T>& v) {
  IdArray cpu =
      dgl::aten::NewIdArray(static_cast<int64_t>(v.size()), CPU, sizeof(T) * 8);
  std::copy(v.begin(), v.end(), Ptr<T>(cpu));
  return cpu.CopyTo(NPU);
}

// Native NDArray with an arbitrary dtype on kDGLAscend ctx (bypasses
// NewIdArray's 32/64-bit gate). Used by illegal-dtype FATAL test inputs;
// content is never read because dispatch fails at the dtype gate.
inline NDArray MakeNpuArrayDType(int64_t n, uint8_t code, uint8_t bits,
                                 uint16_t lanes) {
  DGLDataType dtype;
  dtype.code = code;
  dtype.bits = bits;
  dtype.lanes = lanes;
  return NDArray::Empty({n}, dtype, NPU);
}

// ---------------------------------------------------------------------------
// Stream sync via the public DeviceAPI (ascend impl: aclrtSynchronizeStream
// on the current stream, src/runtime/ascend/ascend_device_api.cc).
// ---------------------------------------------------------------------------
inline void SyncAscend() {
  auto* api = dgl::runtime::DeviceAPI::Get(NPU);
  api->StreamSync(NPU, api->GetStream());
}

// ---------------------------------------------------------------------------
// Bit-exact verification of an Ascend output against the CPU golden vector.
// Implements the authoritative assertion (mismatch == 0) plus the mandatory
// meta checks, and prints the per-case verdict record line.
// ---------------------------------------------------------------------------
template <typename T>
inline ::testing::AssertionResult VerifyNegOutput(const std::string& case_id,
                                                  const NDArray& got,
                                                  const std::vector<T>& expect) {
  const int64_t n = static_cast<int64_t>(expect.size());
  const char* dtype_str = (sizeof(T) == 4) ? "i32" : "i64";

  bool meta_ok = true;
  std::string meta_detail;
  if (got->ndim != 1 || got->shape[0] != n) {
    meta_ok = false;
    meta_detail += " shape[" + std::to_string(got->ndim) + "d,";
    meta_detail += got->ndim > 0 ? std::to_string(got->shape[0]) : "?";
    meta_detail += "!=" + std::to_string(n) + "]";
  }
  if (got->dtype.code != kDGLInt || got->dtype.bits != sizeof(T) * 8 ||
      got->dtype.lanes != 1) {
    meta_ok = false;
    meta_detail += " dtype[code=" + std::to_string(got->dtype.code) +
                   ",bits=" + std::to_string(got->dtype.bits) + "]";
  }
  if (got->ctx.device_type != kDGLAscend || got->ctx.device_id != NPU.device_id) {
    meta_ok = false;
    meta_detail +=
        " ctx[device_type=" + std::to_string(got->ctx.device_type) + "]";
  }

  int64_t mismatch = 0;
  int64_t first_bad = -1;
  T got_first = 0;
  T expect_first = 0;
  if (n > 0 && meta_ok) {
    NDArray got_cpu = got.CopyTo(CPU);
    const T* got_data = Ptr<T>(got_cpu);
    for (int64_t i = 0; i < n; ++i) {
      if (got_data[i] != expect[i]) {
        if (first_bad < 0) {
          first_bad = i;
          got_first = got_data[i];
          expect_first = expect[i];
        }
        ++mismatch;
      }
    }
  }

  const bool pass = meta_ok && mismatch == 0;
  std::printf(
      "[UNARY-ELEWISE-VERDICT] case=%s tensor=y dtype=%s n=%lld "
      "metric=bit-exact mismatch=%lld/%lld threshold=mismatch=0 "
      "meta=%s%s verdict=%s\n",
      case_id.c_str(), dtype_str, static_cast<long long>(n),
      static_cast<long long>(mismatch), static_cast<long long>(n),
      meta_ok ? "ok" : "FAIL",
      meta_ok ? "" : meta_detail.c_str(), pass ? "PASS" : "FAIL");
  std::fflush(stdout);

  if (!pass) {
    ::testing::AssertionResult res = ::testing::AssertionFailure();
    res << case_id << ": authoritative bit-exact assertion failed"
        << " (threshold mismatch=0)";
    if (!meta_ok) res << "; meta:" << meta_detail;
    if (mismatch != 0) {
      res << "; mismatched=" << mismatch << " of " << n
          << "; first_bad_idx=" << first_bad
          << " got=" << static_cast<int64_t>(got_first)
          << " expect=" << static_cast<int64_t>(expect_first);
    }
    return res;
  }
  return ::testing::AssertionSuccess();
}

// Auxiliary cross-check (non-gating): the independent golden must agree
// bit-wise with the CPU reference implementation
// (dgl::aten::Neg on a kDGLCPU array, arith::Neg::Call `-t1` semantics).
template <typename T>
inline void AuxCrossCheckGoldenWithCpu(const std::string& case_id,
                                       const std::vector<T>& x) {
  IdArray cpu =
      dgl::aten::NewIdArray(static_cast<int64_t>(x.size()), CPU, sizeof(T) * 8);
  std::copy(x.begin(), x.end(), Ptr<T>(cpu));
  IdArray cpu_neg = dgl::aten::Neg(cpu);
  cpu_neg = cpu_neg.CopyTo(CPU);

  int64_t mismatch = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(x.size()); ++i) {
    if (Ptr<T>(cpu_neg)[i] != NegGolden(x[i])) ++mismatch;
  }
  std::printf(
      "[UNARY-ELEWISE-AUX] case=%s check=golden-vs-cpu-ref dtype=%s n=%lld "
      "mismatch=%lld (non-gating auxiliary cross-check) %s\n",
      case_id.c_str(), (sizeof(T) == 4) ? "i32" : "i64",
      static_cast<long long>(x.size()), static_cast<long long>(mismatch),
      mismatch == 0 ? "CONSISTENT" : "INCONSISTENT");
  std::fflush(stdout);
  ASSERT_EQ(mismatch, 0)
      << case_id << ": independent golden disagrees with CPU reference";
}

}  // namespace unary_elewise_test

#endif  // TEST_UNARY_ELEWISE_TEST_UTIL_H_
