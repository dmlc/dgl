// UnaryElewise (arith::Neg, aten::Neg) black-box gtest suite.
//
// All cases call dgl::aten::Neg() as a black box on kDGLAscend arrays and
// verify against the independent CPU golden bit-exactly (mismatch == 0),
// covering the fixed case table (L0 x2, L1 x13, L2 x7) plus supplements:
//   S-L2-08  int16 illegal dtype (macro.h:119 FATAL branch)
//   S-CROSS  full dtype x n-staircase cross (parameterized)
// Assertion format: unary_elewise_test_util.h.

#include <dmlc/logging.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "./common.h"
#include "./unary_elewise_test_util.h"

using namespace dgl;    // NOLINT
using namespace dgl::runtime;  // NOLINT

namespace {

using unary_elewise_test::MakeNpuArray;
using unary_elewise_test::NegGolden;
using unary_elewise_test::RandomValues;
using unary_elewise_test::SeedFrom;

#ifdef DGL_USE_ASCEND

TEST(UnaryElewiseTest, L0_01_Int32_N1024_Random) {
  const std::string case_id = "L0-01";
  const int64_t n = 1024;
  auto x = RandomValues<int32_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);

  unary_elewise_test::AuxCrossCheckGoldenWithCpu(case_id, x);

  IdArray x_npu = MakeNpuArray(x);
  IdArray y_npu = aten::Neg(x_npu);
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L0_02_Int64_N1024_Random) {
  const std::string case_id = "L0-02";
  const int64_t n = 1024;
  auto x = RandomValues<int64_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);

  unary_elewise_test::AuxCrossCheckGoldenWithCpu(case_id, x);

  IdArray x_npu = MakeNpuArray(x);
  IdArray y_npu = aten::Neg(x_npu);
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_01_Int32_N1_Single) {
  const std::string case_id = "L1-01";
  std::vector<int32_t> x = {7};
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_02_Int64_N1_Single) {
  const std::string case_id = "L1-02";
  std::vector<int64_t> x = {7};
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_03_Int32_N31_UnalignedTail) {
  const std::string case_id = "L1-03";
  const int64_t n = 31;
  auto x = RandomValues<int32_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_04_Int64_N33_UnalignedTail) {
  const std::string case_id = "L1-04";
  const int64_t n = 33;
  auto x = RandomValues<int64_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_05_Int32_N64_FullBlock) {
  const std::string case_id = "L1-05";
  const int64_t n = 64;
  auto x = RandomValues<int32_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_06_Int64_N65_BlockPlusOne) {
  const std::string case_id = "L1-06";
  const int64_t n = 65;
  auto x = RandomValues<int64_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_07_Int32_N2Pow20_Large) {
  const std::string case_id = "L1-07";
  const int64_t n = int64_t{1} << 20;
  auto x = RandomValues<int32_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_08_Int64_N2Pow20_Large) {
  const std::string case_id = "L1-08";
  const int64_t n = int64_t{1} << 20;
  auto x = RandomValues<int64_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_09_Int32_N2Pow20Plus1_MultiCoreRemainder) {
  const std::string case_id = "L1-09";
  const int64_t n = (int64_t{1} << 20) + 1;
  auto x = RandomValues<int32_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

// Directed prefix {0, 1, -1, INT64_MAX} (INT64_MIN intentionally excluded,
// it is covered by L2-04) followed by full-value-domain random remainder.
TEST(UnaryElewiseTest, L1_10_Int64_N4096_MixedValues) {
  const std::string case_id = "L1-10";
  const int64_t n = 4096;
  std::vector<int64_t> x = {0, 1, -1, std::numeric_limits<int64_t>::max()};
  auto rand_tail = RandomValues<int64_t>(n - x.size(), SeedFrom(case_id));
  x.insert(x.end(), rand_tail.begin(), rand_tail.end());
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

// Directed prefix {0, 1, -1, INT32_MAX} (INT32_MIN covered by L2-03) followed
// by full-value-domain random remainder.
TEST(UnaryElewiseTest, L1_11_Int32_N4096_MixedValues) {
  const std::string case_id = "L1-11";
  const int64_t n = 4096;
  std::vector<int32_t> x = {0, 1, -1, std::numeric_limits<int32_t>::max()};
  auto rand_tail = RandomValues<int32_t>(n - x.size(), SeedFrom(case_id));
  x.insert(x.end(), rand_tail.begin(), rand_tail.end());
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_12_Int32_N2048_AllZero) {
  const std::string case_id = "L1-12";
  const int64_t n = 2048;
  std::vector<int32_t> x(n, 0);
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L1_13_Int64_N2048_AllZero) {
  const std::string case_id = "L1-13";
  const int64_t n = 2048;
  std::vector<int64_t> x(n, 0);
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L2_01_Int32_N0_Empty) {
  const std::string case_id = "L2-01";
  std::vector<int32_t> x;
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L2_02_Int64_N0_Empty) {
  const std::string case_id = "L2-02";
  std::vector<int64_t> x;
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L2_03_Int32_N1_IntMinWrap) {
  const std::string case_id = "L2-03";
  std::vector<int32_t> x = {std::numeric_limits<int32_t>::min()};
  auto expect = unary_elewise_test::NegGoldenVec(x);
  ASSERT_EQ(expect[0], x[0]);  // wrap-around: -INT32_MIN == INT32_MIN
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L2_04_Int64_N1_IntMinWrap) {
  const std::string case_id = "L2-04";
  std::vector<int64_t> x = {std::numeric_limits<int64_t>::min()};
  auto expect = unary_elewise_test::NegGoldenVec(x);
  ASSERT_EQ(expect[0], x[0]);  // wrap-around: -INT64_MIN == INT64_MIN
  IdArray y_npu = aten::Neg(MakeNpuArray(x));
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
}

TEST(UnaryElewiseTest, L2_05_Int32_N2Pow22_Stress) {
  const std::string case_id = "L2-05";
  const int64_t n = int64_t{1} << 22;
  auto x = RandomValues<int32_t>(n, SeedFrom(case_id));
  auto expect = unary_elewise_test::NegGoldenVec(x);
  IdArray x_npu = MakeNpuArray(x);
  IdArray y_npu = aten::Neg(x_npu);
  EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));

  // Single-call duration record (informational, not a pass threshold).
  auto t0 = std::chrono::steady_clock::now();
  IdArray y_timed = aten::Neg(x_npu);
  unary_elewise_test::SyncAscend();
  auto t1 = std::chrono::steady_clock::now();
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  std::printf(
      "[UNARY-ELEWISE-PERF] case=L2-05 dtype=i32 n=%lld source=host-e2e "
      "single_call_us=%.1f bytes=%lld eff_GBps=%.2f "
      "note=record-only-not-a-gate\n",
      static_cast<long long>(n), us,
      static_cast<long long>(2 * n * static_cast<int64_t>(sizeof(int32_t))),
      (2.0 * n * sizeof(int32_t)) / us / 1e3);
  std::fflush(stdout);
}

// ---------------------------------------------------------------------------
// Illegal dtype FATAL tests (L2-06 / L2-07).
// ATEN_ID_TYPE_SWITCH checks the type CODE first (macro.h:111 CHECK_EQ
// "ID must be integer type"), so float32/float64 inputs fail there; the
// macro.h:119 "ID can only be int32 or int64" branch is covered separately
// by S-L2-08 (int16).
// Inputs are allocated on kDGLAscend ctx in SetUp; the statement only runs
// host-side dispatch and fails at the dtype gate.
//
// Mechanism: this repo's dmlc defaults are DMLC_LOG_FATAL_THROW=1 +
// DMLC_LOG_BEFORE_THROW=0, i.e. LOG(FATAL) throws dmlc::Error WITHOUT
// printing to stderr, and gtest catches escaping exceptions inside the
// EXPECT_DEATH child process, so the child never dies and EXPECT_DEATH
// cannot work here. The assertion is therefore in-process:
// EXPECT_THROW(dmlc::Error) plus an explicit check of the FATAL message
// text (dmlc::Error.what()).
// ---------------------------------------------------------------------------
class UnaryElewiseFatalMsgTest : public ::testing::Test {
 protected:
  void SetUp() override {
    f32_ = unary_elewise_test::MakeNpuArrayDType(
        4, kDGLFloat, 32, 1);
    f64_ = unary_elewise_test::MakeNpuArrayDType(
        4, kDGLFloat, 64, 1);
    i16_ = unary_elewise_test::MakeNpuArrayDType(
        4, kDGLInt, 16, 1);
  }

  NDArray f32_;
  NDArray f64_;
  NDArray i16_;
};

#if DMLC_LOG_FATAL_THROW

TEST_F(UnaryElewiseFatalMsgTest, L2_06_Float32FatalThrow) {
  EXPECT_THROW((void)aten::Neg(f32_), dmlc::Error);  // NOLINT
  try {
    (void)aten::Neg(f32_);
    FAIL() << "expected dmlc::Error (LOG(FATAL)) for float32 dtype";
  } catch (const dmlc::Error& e) {
    EXPECT_NE(std::string(e.what()).find("ID must be integer type"),
              std::string::npos)
        << "case=L2-06: FATAL message mismatch, actual: " << e.what();
  }
  std::printf(
      "[UNARY-ELEWISE-VERDICT] case=L2-06 tensor=n/a dtype=f32 n=4 "
      "metric=fatal-throw threshold=throws-dmlc-Error+what~\"ID must be "
      "integer type\"(macro.h:111) verdict=%s\n",
      (HasFailure()) ? "FAIL" : "PASS");
  std::fflush(stdout);
}

TEST_F(UnaryElewiseFatalMsgTest, L2_07_Float64FatalThrow) {
  EXPECT_THROW((void)aten::Neg(f64_), dmlc::Error);  // NOLINT
  try {
    (void)aten::Neg(f64_);
    FAIL() << "expected dmlc::Error (LOG(FATAL)) for float64 dtype";
  } catch (const dmlc::Error& e) {
    EXPECT_NE(std::string(e.what()).find("ID must be integer type"),
              std::string::npos)
        << "case=L2-07: FATAL message mismatch, actual: " << e.what();
  }
  std::printf(
      "[UNARY-ELEWISE-VERDICT] case=L2-07 tensor=n/a dtype=f64 n=4 "
      "metric=fatal-throw threshold=throws-dmlc-Error+what~\"ID must be "
      "integer type\"(macro.h:111) verdict=%s\n",
      (HasFailure()) ? "FAIL" : "PASS");
  std::fflush(stdout);
}

// int16 passes the code check at macro.h:111 and fails at the bits branch,
// macro.h:119.
TEST_F(UnaryElewiseFatalMsgTest, SL2_08_Int16FatalThrow) {
  EXPECT_THROW((void)aten::Neg(i16_), dmlc::Error);  // NOLINT
  try {
    (void)aten::Neg(i16_);
    FAIL() << "expected dmlc::Error (LOG(FATAL)) for int16 dtype";
  } catch (const dmlc::Error& e) {
    EXPECT_NE(std::string(e.what()).find("ID can only be int32 or int64"),
              std::string::npos)
        << "case=S-L2-08: FATAL message mismatch, actual: " << e.what();
  }
  std::printf(
      "[UNARY-ELEWISE-VERDICT] case=S-L2-08 tensor=n/a dtype=i16 n=4 "
      "metric=fatal-throw threshold=throws-dmlc-Error+what~\"ID can only be "
      "int32 or int64\"(macro.h:119) verdict=%s\n",
      (HasFailure()) ? "FAIL" : "PASS");
  std::fflush(stdout);
}

#else  // DMLC_LOG_FATAL_THROW

// If FATAL is configured to abort instead of throw, in-process assertion is
// impossible (the process terminates at the first illegal-dtype call).
TEST_F(UnaryElewiseFatalMsgTest, MsgChecksUnavailable) {
  std::printf(
      "[UNARY-ELEWISE-VERDICT] suite=UnaryElewiseFatalMsgTest skipped "
      "reason=DMLC_LOG_FATAL_THROW=0 (FATAL aborts in-process)\n");
  std::fflush(stdout);
}

#endif  // DMLC_LOG_FATAL_THROW

// ---------------------------------------------------------------------------
// Full dtype x n-staircase cross over the fixed length ladder, both dtypes,
// fixed-seed full-value-domain random fill; extra coverage beyond the fixed
// case table.
// ---------------------------------------------------------------------------
struct CrossParam {
  int64_t n;
  int bits;

  std::string Name() const {
    return (bits == 32 ? std::string("i32") : std::string("i64")) + "_n" +
           std::to_string(n);
  }
};

void PrintTo(const CrossParam& p, std::ostream* os) { *os << p.Name(); }

class UnaryElewiseCrossTest : public ::testing::TestWithParam<CrossParam> {};

TEST_P(UnaryElewiseCrossTest, SuppCrossBitExact) {
  const CrossParam p = GetParam();
  const std::string case_id = "S-CROSS-" + p.Name();
  if (p.bits == 32) {
    auto x = RandomValues<int32_t>(p.n, SeedFrom(case_id));
    auto expect = unary_elewise_test::NegGoldenVec(x);
    IdArray y_npu = aten::Neg(MakeNpuArray(x));
    EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
  } else {
    auto x = RandomValues<int64_t>(p.n, SeedFrom(case_id));
    auto expect = unary_elewise_test::NegGoldenVec(x);
    IdArray y_npu = aten::Neg(MakeNpuArray(x));
    EXPECT_TRUE(unary_elewise_test::VerifyNegOutput(case_id, y_npu, expect));
  }
}

INSTANTIATE_TEST_SUITE_P(
    Supp, UnaryElewiseCrossTest,
    ::testing::ValuesIn(std::vector<CrossParam>{
        {0, 32},
        {0, 64},
        {1, 32},
        {1, 64},
        {31, 32},
        {31, 64},
        {33, 32},
        {33, 64},
        {64, 32},
        {64, 64},
        {65, 32},
        {65, 64},
        {1024, 32},
        {1024, 64},
        {2048, 32},
        {2048, 64},
        {4096, 32},
        {4096, 64},
        {int64_t{1} << 20, 32},
        {int64_t{1} << 20, 64},
        {(int64_t{1} << 20) + 1, 32},
        {(int64_t{1} << 20) + 1, 64},
        {int64_t{1} << 22, 32},
        {int64_t{1} << 22, 64},
    }),
    [](const ::testing::TestParamInfo<CrossParam>& info) {
      return info.param.Name();
    });

#else  // DGL_USE_ASCEND

TEST(UnaryElewiseTest, AscendNotCompiled) {
  std::printf(
      "[UNARY-ELEWISE-VERDICT] suite=disabled reason=DGL_USE_ASCEND is not "
      "defined; build with USE_ASCEND=ON to enable the UnaryElewise cases\n");
  std::fflush(stdout);
}

#endif  // DGL_USE_ASCEND

}  // namespace
