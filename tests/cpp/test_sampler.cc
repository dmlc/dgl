#include <gtest/gtest.h>
#include <dgl/sample_utils.h>
#include <vector>
#include <algorithm>
#include "./common.h"

using namespace dgl;

template <typename Idx, typename DType>
void _TestWithReplacement(RandomEngine *re) {
  Idx n_categories = 1000;
  Idx n_rolls = 100000;
  std::vector<DType> prob;
  DType accum = 0.;
  for (Idx i = 0; i < n_categories; ++i) {
    prob.push_back(re->Uniform<DType>());
    accum += prob.back();
  }
  for (Idx i = 0; i < n_categories; ++i)
    prob[i] /= accum;

  auto _test_given_sampler = [n_categories, n_rolls, &prob](
      BaseSampler<Idx, DType, true> *s) {
    std::vector<bool> counter(n_categories, 0);
    for (Idx i = 0; i < n_rolls; ++i) {
      Idx dice = s->draw();
      counter[dice]++;
    }
    for (Idx i = 0; i < n_categories; ++i)
      ASSERT_NEAR(static_cast<DType>(counter[i]) / n_rolls, prob[i], 1e-2);
  };

  AliasSampler<Idx, DType, true> as(re, prob);
  CDFSampler<Idx, DType, true> cs(re, prob);
  TreeSampler<Idx, DType, true> ts(re, prob);
  _test_given_sampler(&as);
  _test_given_sampler(&cs);
  _test_given_sampler(&ts);
}

TEST(SampleUtilsTest, TestWithReplacement) {
  RandomEngine* re = RandomEngine::ThreadLocal();
  re->SetSeed(42);
  _TestWithReplacement<int32_t, float>(re);
  re->SetSeed(42);
  _TestWithReplacement<int32_t, double>(re);
  re->SetSeed(42);
  _TestWithReplacement<int64_t, float>(re);
  re->SetSeed(42);
  _TestWithReplacement<int64_t, double>(re);
};

TEST(SampleUtilsTest, TestWithoutReplacementOrder) {

};

TEST(SampleUtilsTest, TestWithoutReplacementUnique) {

};
