#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "../../src/random/cpu/sample_utils.h"
#include "./common.h"

using namespace dgl;
using namespace dgl::aten;

// TODO: adapt this to Random::Choice

template <typename Idx, typename DType>
void _TestWithReplacement(RandomEngine* re) {
  Idx n_categories = 100;
  Idx n_rolls = 1000000;
  std::vector<DType> _prob;
  DType accum = 0.;
  for (Idx i = 0; i < n_categories; ++i) {
    _prob.push_back(re->Uniform<DType>());
    accum += _prob.back();
  }
  for (Idx i = 0; i < n_categories; ++i) _prob[i] /= accum;
  FloatArray prob = NDArray::FromVector(_prob);

  auto _check_given_sampler = [n_categories, n_rolls,
                               &_prob](utils::BaseSampler<Idx>* s) {
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
void _TestWithoutReplacementOrder(RandomEngine* re) {
  // TODO(BarclayII): is there a reliable way to do this test?
  std::vector<DType> _prob = {1e6f, 1e-6f, 1e-2f, 1e2f};
  FloatArray prob = NDArray::FromVector(_prob);
  std::vector<Idx> ground_truth = {0, 3, 2, 1};

  auto _check_given_sampler = [&ground_truth](utils::BaseSampler<Idx>* s) {
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
void _TestWithoutReplacementUnique(RandomEngine* re) {
  Idx N = 1000000;
  std::vector<DType> _likelihood;
  for (Idx i = 0; i < N; ++i) _likelihood.push_back(re->Uniform<DType>());
  FloatArray likelihood = NDArray::FromVector(_likelihood);

  auto _check_given_sampler = [N](utils::BaseSampler<Idx>* s) {
    std::vector<int> cnt(N, 0);
    for (Idx i = 0; i < N; ++i) {
      Idx dice = s->Draw();
      cnt[dice]++;
    }
    for (Idx i = 0; i < N; ++i) ASSERT_EQ(cnt[i], 1);
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

template <typename Idx, typename DType>
void _TestChoice(RandomEngine* re) {
  re->SetSeed(42);
  std::vector<DType> prob_vec = {1., 0., 0., 0., 2., 2., 0., 0.};
  FloatArray prob = FloatArray::FromVector(prob_vec);
  {
    for (int k = 0; k < 1000; ++k) {
      Idx x = re->Choice<Idx>(prob);
      ASSERT_TRUE(x == 0 || x == 4 || x == 5);
    }
  }
  // num = 0
  {
    IdArray rst = re->Choice<Idx, DType>(0, prob, true);
    ASSERT_EQ(rst->shape[0], 0);
  }
  // w/ replacement
  {
    IdArray rst = re->Choice<Idx, DType>(1000, prob, true);
    ASSERT_EQ(rst->shape[0], 1000);
    for (int64_t i = 0; i < 1000; ++i) {
      Idx x = static_cast<Idx*>(rst->data)[i];
      ASSERT_TRUE(x == 0 || x == 4 || x == 5);
    }
  }
  // w/o replacement
  {
    IdArray rst = re->Choice<Idx, DType>(3, prob, false);
    ASSERT_EQ(rst->shape[0], 3);
    std::set<Idx> idxset;
    for (int64_t i = 0; i < 3; ++i) {
      Idx x = static_cast<Idx*>(rst->data)[i];
      idxset.insert(x);
    }
    ASSERT_EQ(idxset.size(), 3);
    ASSERT_EQ(idxset.count(0), 1);
    ASSERT_EQ(idxset.count(4), 1);
    ASSERT_EQ(idxset.count(5), 1);
  }
}

TEST(RandomTest, TestChoice) {
  RandomEngine* re = RandomEngine::ThreadLocal();
  _TestChoice<int32_t, float>(re);
  _TestChoice<int64_t, float>(re);
  _TestChoice<int32_t, double>(re);
  _TestChoice<int64_t, double>(re);
}

template <typename Idx>
void _TestUniformChoice(RandomEngine* re) {
  re->SetSeed(42);
  // num == 0
  {
    IdArray rst = re->UniformChoice<Idx>(0, 100, true);
    ASSERT_EQ(rst->shape[0], 0);
  }
  // w/ replacement
  {
    IdArray rst = re->UniformChoice<Idx>(1000, 100, true);
    ASSERT_EQ(rst->shape[0], 1000);
    for (int64_t i = 0; i < 1000; ++i) {
      Idx x = static_cast<Idx*>(rst->data)[i];
      ASSERT_TRUE(x >= 0 && x < 100);
    }
  }
  // w/o replacement
  {
    IdArray rst = re->UniformChoice<Idx>(99, 100, false);
    ASSERT_EQ(rst->shape[0], 99);
    std::set<Idx> idxset;
    for (int64_t i = 0; i < 99; ++i) {
      Idx x = static_cast<Idx*>(rst->data)[i];
      ASSERT_TRUE(x >= 0 && x < 100);
      idxset.insert(x);
    }
    ASSERT_EQ(idxset.size(), 99);
  }
}

TEST(RandomTest, TestUniformChoice) {
  RandomEngine* re = RandomEngine::ThreadLocal();
  _TestUniformChoice<int32_t>(re);
  _TestUniformChoice<int64_t>(re);
  _TestUniformChoice<int32_t>(re);
  _TestUniformChoice<int64_t>(re);
}

template <typename Idx, typename FloatType>
void _TestBiasedChoice(RandomEngine* re) {
  re->SetSeed(42);
  // num == 0
  {
    Idx split[] = {0, 1, 2};
    FloatArray bias = NDArray::FromVector(std::vector<FloatType>({1, 3}));
    IdArray rst = re->BiasedChoice<Idx, FloatType>(0, split, bias, true);
    ASSERT_EQ(rst->shape[0], 0);
  }
  // basic test
  {
    Idx sample_num = 100000;
    Idx population = 1000000;
    Idx split[] = {0, population / 2, population};
    FloatArray bias = NDArray::FromVector(std::vector<FloatType>({1, 3}));

    IdArray rst =
        re->BiasedChoice<Idx, FloatType>(sample_num, split, bias, true);
    auto rst_data = static_cast<Idx*>(rst->data);
    Idx larger = 0;
    for (Idx i = 0; i < sample_num; ++i)
      if (rst_data[i] >= population / 2) larger++;
    ASSERT_LE(fabs((double)larger / sample_num - 0.75), 1e-2);
  }
  // without replacement
  {
    Idx sample_num = 500;
    Idx population = 1000;
    Idx split[] = {0, sample_num, population};
    FloatArray bias = NDArray::FromVector(std::vector<FloatType>({1, 0}));

    IdArray rst =
        re->BiasedChoice<Idx, FloatType>(sample_num, split, bias, false);
    auto rst_data = static_cast<Idx*>(rst->data);

    std::set<Idx> idxset;
    for (int64_t i = 0; i < sample_num; ++i) {
      Idx x = rst_data[i];
      ASSERT_LT(x, sample_num);
      idxset.insert(x);
    }
    ASSERT_EQ(idxset.size(), sample_num);
  }
}

TEST(RandomTest, TestBiasedChoice) {
  RandomEngine* re = RandomEngine::ThreadLocal();
  _TestBiasedChoice<int32_t, float>(re);
  _TestBiasedChoice<int64_t, float>(re);
  _TestBiasedChoice<int32_t, double>(re);
  _TestBiasedChoice<int64_t, double>(re);
}
