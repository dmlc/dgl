/*!
 *  Copyright (c) 2019 by Contributors
 * \file random/choice.cc
 * \brief Non-uniform discrete sampling implementation
 */

#include <dgl/random.h>
#include <dgl/array.h>
#include <vector>
#include <numeric>
#include "sample_utils.h"

namespace dgl {

template<typename IdxType>
IdxType RandomEngine::Choice(FloatArray prob) {
  IdxType ret = 0;
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, ValueType, "probability", {
    // TODO(minjie): allow choosing different sampling algorithms
    utils::TreeSampler<IdxType, ValueType, true> sampler(this, prob);
    ret = sampler.Draw();
  });
  return ret;
}

template int32_t RandomEngine::Choice<int32_t>(FloatArray);
template int64_t RandomEngine::Choice<int64_t>(FloatArray);


template<typename IdxType, typename FloatType>
void RandomEngine::Choice(int64_t num, FloatArray prob, IdxType* out, bool replace) {
  const int64_t N = prob->shape[0];
  if (!replace)
    CHECK_LE(num, N) << "Cannot take more sample than population when 'replace=false'";
  if (num == N && !replace)
    std::iota(out, out + num, 0);

  utils::BaseSampler<IdxType>* sampler = nullptr;
  if (replace) {
    sampler = new utils::TreeSampler<IdxType, FloatType, true>(this, prob);
  } else {
    sampler = new utils::TreeSampler<IdxType, FloatType, false>(this, prob);
  }
  for (int64_t i = 0; i < num; ++i)
    out[i] = sampler->Draw();
  delete sampler;
}

template void RandomEngine::Choice<int32_t, float>(
    int64_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, float>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, double>(
    int64_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, double>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);

template <typename IdxType>
void RandomEngine::UniformChoice(int64_t num, int64_t population, IdxType* out, bool replace) {
  if (!replace)
    CHECK_LE(num, population) << "Cannot take more sample than population when 'replace=false'";
  if (replace) {
    for (int64_t i = 0; i < num; ++i)
      out[i] = RandInt(population);
  } else {
    if (num < population / 2) {
      // use hash set
      // In the best scenario, time complexity is O(num), i.e., no conflict.
      //
      // Let k be num / population, the expected number of extra sampling steps is roughly
      // k^2 / (1-k) * population, which means in the worst case scenario,
      // the time complexity is O(population^2). Therefore, we use k=1/2 as the threshold,
      // which gives an expected 1/2*population number of extra steps.
      std::unordered_set<IdxType> selected;
      while (selected.size() < num) {
        selected.insert(RandInt(population));
      }
      std::copy(selected.begin(), selected.end(), out);
    } else {
      // reservoir algorithm
      // time: O(population), space: O(num)
      for (int64_t i = 0; i < num; ++i)
        out[i] = i;
      for (uint64_t i = num; i < population; ++i) {
        const int64_t j = RandInt(i);
        if (j < num)
          out[j] = i;
      }
    }
  }
}

template void RandomEngine::UniformChoice<int32_t>(
    int64_t num, int64_t population, int32_t* out, bool replace);
template void RandomEngine::UniformChoice<int64_t>(
    int64_t num, int64_t population, int64_t* out, bool replace);

};  // namespace dgl
