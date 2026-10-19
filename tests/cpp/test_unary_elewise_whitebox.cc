// UnaryElewise white-box supplement cases.
//
// Case selection is source-driven (branch -> case mapping); execution is
// the same black-box call dgl::aten::Neg() with the authoritative bit-exact
// assertion. Branches NOT triggerable externally are covered by static
// review (CalcBlockDim fallback-8, n > UINT32_MAX, CUDA dispatch variant).
//
// Kernel structure covered:
//   i32 vector path (unary_elewise_kernel.cpp): per-core range from
//     UnaryBlockRange, tile loop capped at kUnaryTileElems=8192 -> DataCopy
//     (cnt==8192) vs DataCopyPad (cnt<8192); TQue depth 2 double buffering.
//   i64 lane-vectorized path (unary_elewise_i64_kernel.cpp): the int64
//     stream is addressed as 2n int32 lanes; one Sub(t, borrow, s) with the
//     static [0,-1,0,-1,...] parity pattern negates both lanes. A per-tile
//     zero probe (Cast+Abs+ReduceMin) routes any tile containing a zero
//     int32 lane to the exact rare-tile recompute path -- bit-exact for
//     every int64 value; random data keeps the hot path pure vector/MTE.
//     Tile = 1024 int64 elems (2048 lanes).
//   Splitting: per_core=ceil(n/block_num), blockFormer=round_up(per_core,
//     512), last core carries remainder, start>=n cores idle (tiling.h).
//     Idle cores require full-capacity launch (needed >= physical cores C);
//     with bd=needed, per_core<=1024 and bf<=1024 so bf*(bd-1)<n always
//     -> no idle.
// Host (unary_elewise.cc): CalcBlockDim gate n->needed=ceil(n/1024),
//   bd=min(needed, C); n==0 early return; n>UINT32_MAX CHECK.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "./common.h"
#include "./unary_elewise_test_util.h"

using namespace dgl;           // NOLINT
using namespace dgl::runtime;  // NOLINT

namespace {

using unary_elewise_test::MakeNpuArray;
using unary_elewise_test::RandomValues;

#ifdef DGL_USE_ASCEND

// Generic runner: random full-value-domain fill (fixed seed per case id),
// bit-exact against the independent CPU golden.
template <typename T>
void RunWhiteboxCase(const std::string& case_id, int64_t n) {
  auto x = RandomValues<T>(n, unary_elewise_test::SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

// ---- i32 vector path -------------------------------------------------------
// Single-core gate boundary (CalcBlockDim: needed=1 -> bd=1): threshold-1 /
// threshold / threshold+1 (multi-core onset). Threshold+1 additionally makes
// the last core carry a 1-element remainder (DataCopyPad minimal tail).
TEST(UnaryElewiseWhiteboxTest, WB_01_I32_N1023_SingleCore) {
  RunWhiteboxCase<int32_t>("WB-01", 1023);
}
TEST(UnaryElewiseWhiteboxTest, WB_02_I32_N1024_SingleCoreExactAlign) {
  RunWhiteboxCase<int32_t>("WB-02", 1024);
}
TEST(UnaryElewiseWhiteboxTest, WB_03_I32_N1025_MultiCoreOnset_1ElemTail) {
  RunWhiteboxCase<int32_t>("WB-03", 1025);
}
// Full-capacity launch (needed=41 >= C in {20,40}): bf=round_up(ceil(n/C),
// 512) makes bf*(bd-1) >= n -> idle cores (start>=n, loop skipped) plus a
// remainder core; exercises the end=min(...) selection on the last core.
TEST(UnaryElewiseWhiteboxTest, WB_04_I32_N40961_IdleCores_RemainderCore) {
  RunWhiteboxCase<int32_t>("WB-04", 40961);
}
// n = 8192 * 40: per-core == kTileElems exactly (C=40) -> DataCopy FULL tile
// path (:59-60/:76-77), single loop iteration, no pad, no idle; C=20 gives
// two full tiles per core (>=2 iterations -> double buffering in flight).
TEST(UnaryElewiseWhiteboxTest, WB_05_I32_N327680_ExactFullTiles) {
  RunWhiteboxCase<int32_t>("WB-05", 327680);
}
// Full tiles + 1-element DataCopyPad tail on the last active core
// (full/pad mixed sequence within one core range).
TEST(UnaryElewiseWhiteboxTest, WB_06_I32_N327681_FullTilesPlusPadTail) {
  RunWhiteboxCase<int32_t>("WB-06", 327681);
}

// ---- i64 lane-vectorized path ---------------------------------------------
TEST(UnaryElewiseWhiteboxTest, WB_07_I64_N1023_SingleCore) {
  RunWhiteboxCase<int64_t>("WB-07", 1023);
}
TEST(UnaryElewiseWhiteboxTest, WB_08_I64_N1024_SingleCoreExactAlign) {
  RunWhiteboxCase<int64_t>("WB-08", 1024);
}
TEST(UnaryElewiseWhiteboxTest, WB_09_I64_N1025_MultiCoreOnset_1ElemTail) {
  RunWhiteboxCase<int64_t>("WB-09", 1025);
}
TEST(UnaryElewiseWhiteboxTest, WB_10_I64_N40961_IdleCores_RemainderCore) {
  RunWhiteboxCase<int64_t>("WB-10", 40961);
}
TEST(UnaryElewiseWhiteboxTest, WB_11_I64_N327680_FullPerCoreRanges) {
  RunWhiteboxCase<int64_t>("WB-11", 327680);
}
TEST(UnaryElewiseWhiteboxTest, WB_12_I64_N327681_FullPerCorePlusTail) {
  RunWhiteboxCase<int64_t>("WB-12", 327681);
}

#else  // DGL_USE_ASCEND

TEST(UnaryElewiseWhiteboxTest, AscendNotCompiled) {
  std::printf(
      "[UNARY-ELEWISE-VERDICT] whitebox suite=disabled reason=DGL_USE_ASCEND "
      "is not defined\n");
  std::fflush(stdout);
}

#endif  // DGL_USE_ASCEND

}  // namespace
