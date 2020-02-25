/*!
 *  Copyright (c) 2019 by Contributors
 * \file random/choice.cc
 * \brief Non-uniform discrete sampling implementation
 */

#include <dgl/random.h>
#include <dgl/array.h>
#include <vector>
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
IdArray RandomEngine::Choice(int64_t num, FloatArray prob, bool replace) {
  const int64_t N = prob->shape[0];
  if (!replace)
    CHECK_LE(num, N) << "Cannot take more sample than population when 'replace=false'";
  if (num == N && !replace)
    return aten::Range(0, N, sizeof(IdxType) * 8, DLContext{kDLCPU, 0});

  const DLDataType dtype{kDLInt, sizeof(IdxType) * 8, 1};
  IdArray ret = IdArray::Empty({num}, dtype, DLContext{kDLCPU, 0});
  IdxType* ret_data = static_cast<IdxType*>(ret->data);
  utils::BaseSampler<IdxType>* sampler = nullptr;
  if (replace) {
    sampler = new utils::TreeSampler<IdxType, FloatType, true>(this, prob);
  } else {
    sampler = new utils::TreeSampler<IdxType, FloatType, false>(this, prob);
  }
  for (int64_t i = 0; i < num; ++i)
    ret_data[i] = sampler->Draw();
  delete sampler;
  return ret;
}

template IdArray RandomEngine::Choice<int32_t, float>(
    int64_t num, FloatArray prob, bool replace);
template IdArray RandomEngine::Choice<int64_t, float>(
    int64_t num, FloatArray prob, bool replace);
template IdArray RandomEngine::Choice<int32_t, double>(
    int64_t num, FloatArray prob, bool replace);
template IdArray RandomEngine::Choice<int64_t, double>(
    int64_t num, FloatArray prob, bool replace);

template <typename IdxType>
IdArray RandomEngine::UniformChoice(int64_t num, int64_t population, bool replace) {
  if (!replace)
    CHECK_LE(num, population) << "Cannot take more sample than population when 'replace=false'";
  const DLDataType dtype{kDLInt, sizeof(IdxType) * 8, 1};
  IdArray ret = IdArray::Empty({num}, dtype, DLContext{kDLCPU, 0});
  IdxType* ret_data = static_cast<IdxType*>(ret->data);
  if (replace) {
    for (int64_t i = 0; i < num; ++i)
      ret_data[i] = RandInt(population);
  } else {
    // time: O(population), space: O(num)
    for (int64_t i = 0; i < num; ++i)
      ret_data[i] = i;
    for (uint64_t i = num; i < population; ++i) {
      const int64_t j = RandInt(i);
      if (j < num)
        ret_data[j] = i;
    }
  }
  return ret;
}

template IdArray RandomEngine::UniformChoice<int32_t>(
    int64_t num, int64_t population, bool replace);
template IdArray RandomEngine::UniformChoice<int64_t>(
    int64_t num, int64_t population, bool replace);

};  // namespace dgl
