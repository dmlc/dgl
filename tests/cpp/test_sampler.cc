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

  auto _check_given_sampler = [n_categories, n_rolls, &prob](
      BaseSampler<Idx, DType, true> *s) {
    std::vector<Idx> counter(n_categories, 0);
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
  _check_given_sampler(&as);
  _check_given_sampler(&cs);
  _check_given_sampler(&ts);
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

template <typename Idx, typename DType>
void _TestWithoutReplacementOrder(RandomEngine *re) {
  Idx N = 4;
  std::vector<DType> prob = {1e6, 1e-6, 1e-2, 1e2};
  std::vector<Idx> ground_truth = {0, 3, 2, 1};

  auto _check_given_sampler = [N, &prob, &ground_truth](
      BaseSampler<Idx, DType, false> *s) {
    for (Idx i = 0; i < N; ++i) {
      Idx dice = s->draw();
      ASSERT_EQ(dice, ground_truth[i]);
    }
  };

  AliasSampler<Idx, DType, false> as(re, prob);
  CDFSampler<Idx, DType, false> cs(re, prob);
  TreeSampler<Idx, DType, false> ts(re, prob);
  _check_given_sampler(&as);
  _check_given_sampler(&cs);
  _check_given_sampler(&ts);
}

TEST(SampleUtilsTest, TestWithoutReplacementOrder) {
  RandomEngine* re = RandomEngine::ThreadLocal();
  re->SetSeed(42);
  _TestWithoutReplacementOrder<int32_t, float>(re);
  re->SetSeed(42);
  _TestWithoutReplacementOrder<int32_t, double>(re);
  re->SetSeed(42);
  _TestWithoutReplacementOrder<int64_t, float>(re);
  re->SetSeed(42);
  _TestWithoutReplacementOrder<int64_t, double>(re);
};

template <typename Idx, typename DType>
void _TestWithoutReplacementUnique(RandomEngine *re) {
  Idx N = 100000;
  std::vector<DType> likelihood;
  for (Idx i = 0; i < N; ++i)
    likelihood.push_back(re->Uniform<DType>());

  auto _check_given_sampler = [N](
      BaseSampler<Idx, DType, false> *s) {
    std::vector<bool> visit(N, false);
    for (Idx i = 0; i < N; ++i) {
      Idx dice = s->draw();
      visit[dice] = true;
    }
    for (Idx i = 0; i < N; ++i)
      ASSERT_EQ(visit[i], true);
  };

  AliasSampler<Idx, DType, false> as(re, likelihood);
  CDFSampler<Idx, DType, false> cs(re, likelihood);
  TreeSampler<Idx, DType, false> ts(re, likelihood);
  _check_given_sampler(&as);
  _check_given_sampler(&cs);
  _check_given_sampler(&ts);
}

TEST(SampleUtilsTest, TestWithoutReplacementUnique) {
  RandomEngine* re = RandomEngine::ThreadLocal();
  re->SetSeed(42);
  _TestWithoutReplacementUnique<int32_t, float>(re);
  re->SetSeed(42);
  _TestWithoutReplacementUnique<int32_t, double>(re);
  re->SetSeed(42);
  _TestWithoutReplacementUnique<int64_t, float>(re);
  re->SetSeed(42);
  _TestWithoutReplacementUnique<int64_t, double>(re);
};
