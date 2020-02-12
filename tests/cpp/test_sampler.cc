#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "./common.h"
#include "../../src/random/cpu/sample_utils.h"

using namespace dgl;
using namespace dgl::aten;

// TODO: adapt this to Random::Choice

template <typename Idx, typename DType>
void _TestWithReplacement(RandomEngine *re) {
  Idx n_categories = 100;
  Idx n_rolls = 1000000;
  std::vector<DType> _prob;
  DType accum = 0.;
  for (Idx i = 0; i < n_categories; ++i) {
    _prob.push_back(re->Uniform<DType>());
    accum += _prob.back();
  }
  for (Idx i = 0; i < n_categories; ++i)
    _prob[i] /= accum;
  FloatArray prob = NDArray::FromVector(_prob);

  auto _check_given_sampler = [n_categories, n_rolls, &_prob](
      utils::BaseSampler<Idx, DType, true> *s) {
    std::vector<Idx> counter(n_categories, 0);
    for (Idx i = 0; i < n_rolls; ++i) {
      Idx dice = s->Draw();
      counter[dice]++;
    }
    for (Idx i = 0; i < n_categories; ++i)
      ASSERT_NEAR(static_cast<DType>(counter[i]) / n_rolls, _prob[i], 1e-2);
  };

  auto _check_random_choice = [n_categories, n_rolls, &_prob, prob]() {
    std::vector<int64_t> counter(n_categories, 0);
    for (Idx i = 0; i < n_rolls; ++i) {
      Idx dice = RandomEngine::ThreadLocal()->Choice<int64_t>(prob);
      counter[dice]++;
    }
    for (Idx i = 0; i < n_categories; ++i)
      ASSERT_NEAR(static_cast<DType>(counter[i]) / n_rolls, _prob[i], 1e-2);
  };

  utils::AliasSampler<Idx, DType, true> as(re, prob);
  utils::CDFSampler<Idx, DType, true> cs(re, prob);
  utils::TreeSampler<Idx, DType, true> ts(re, prob);
  _check_given_sampler(&as);
  _check_given_sampler(&cs);
  _check_given_sampler(&ts);
  _check_random_choice();
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
  // TODO(BarclayII): is there a reliable way to do this test?
  std::vector<DType> _prob = {1e6, 1e-6, 1e-2, 1e2};
  FloatArray prob = NDArray::FromVector(_prob);
  std::vector<Idx> ground_truth = {0, 3, 2, 1};

  auto _check_given_sampler = [&ground_truth](
      utils::BaseSampler<Idx, DType, false> *s) {
    for (size_t i = 0; i < ground_truth.size(); ++i) {
      Idx dice = s->Draw();
      ASSERT_EQ(dice, ground_truth[i]);
    }
  };

  utils::AliasSampler<Idx, DType, false> as(re, prob);
  utils::CDFSampler<Idx, DType, false> cs(re, prob);
  utils::TreeSampler<Idx, DType, false> ts(re, prob);
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
  Idx N = 1000000;
  std::vector<DType> _likelihood;
  for (Idx i = 0; i < N; ++i)
    _likelihood.push_back(re->Uniform<DType>());
  FloatArray likelihood = NDArray::FromVector(_likelihood);

  auto _check_given_sampler = [N](
      utils::BaseSampler<Idx, DType, false> *s) {
    std::vector<int> cnt(N, 0);
    for (Idx i = 0; i < N; ++i) {
      Idx dice = s->Draw();
      cnt[dice]++;
    }
    for (Idx i = 0; i < N; ++i)
      ASSERT_EQ(cnt[i], 1);
  };

  utils::AliasSampler<Idx, DType, false> as(re, likelihood);
  utils::CDFSampler<Idx, DType, false> cs(re, likelihood);
  utils::TreeSampler<Idx, DType, false> ts(re, likelihood);
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
