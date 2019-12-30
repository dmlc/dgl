#include <gtest/gtest.h>
#include <dgl/random.h>
#include <dgl/sample_utils.h>
#include <vector>
#include <algorithm>
#include "./common.h"

using namespace dgl;
using namespace dgl::runtime;


TEST(SampleUtilsTest, TestWithReplacement) {
  RandomEngine re();
  re.SetSeed(42);
  _TestWithReplacement<int32_t, float>(&re);
  re.SetSeed(42);
  _TestWithReplacement<int32_t, double>(&re);
  re.SetSeed(42);
  _TestWithReplacement<int64_t, float)(&re);
  re.SetSeed(42);
  _TestWithReplacement<int64_t, double>(&re);
};

TEST(SampleUtilsTest, TestWithoutReplacementOrder) {

};

TEST(SampleUtilsTest, TestWithoutReplacementUnique) {

};

template <typename Idx,
    typename DType>
void _TestWithReplacement(RandomEngine *re) {
  Idx n_categories = 1000;
  Idx n_rolls = 100000;
  std::vector<DType> prob;
  std::vector<bool> counter(n_categories);
  DType accum = 0.;
  for (Idx i = 0; i < n_categories; ++i) {
    prob.push_back(re->Uniform<DType>());
    accum += prob.back();
  }
  for (Idx i = 0; i < n_categories; ++i)
    prob[i] /= accum;

  void _TestGivenSampler(BaseSampler *s, std::vector<bool>& counter) {
    std::fill(counter.begin(), counter.end(), 0);
    for (Idx i = 0; i < n_rolls; ++i) {
      Idx dice = as.draw();
      counter[dice]++;
    }
    for (Idx i = 0; i < n_categories; ++i)
      ASSERT_NEAR(static_cast<DType>(counter[i]) / n_rolls, prob[i], 1e-2);
  }

  AliasSampler<Idx, DType, true> as(&re, prob);
  CDFSampler<Idx, DType, true> cs(&re, prob);
  TreeSampler<Idx, DType, true> ts(&re, prob);
  _TestWithReplacement(&as);
  _TestWithReplacement(&cs);
  _TestWithReplacement(&ts);
}
