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


template<typename IdxType>
IdArray RandomEngine::Choice(int64_t num, FloatArray prob, bool replace) {
  const int64_t N = prob->shape[0];
  if (!replace)
    CHECK_LE(num, N) << "Cannot take more sample than population when 'replace=false'";
  if (num == N)
    return aten::Range(0, N, sizeof(IdxType) * 8, DLContext{kDLCPU, 0});

  const DLDataType dtype{kDLInt, sizeof(IdxType) * 8, 1};
  IdArray ret = IdArray::Empty({num}, dtype, DLContext{kDLCPU, 0});
  IdxType* ret_data = static_cast<IdxType*>(ret->data);
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, ValueType, "probability", {
    // TODO(minjie): allow choosing different sampling algorithms
    utils::BaseSampler<IdxType>* sampler = nullptr;
    if (replace) {
      sampler = new utils::TreeSampler<IdxType, ValueType, true>(this, prob);
    } else {
      sampler = new utils::TreeSampler<IdxType, ValueType, false>(this, prob);
    }
    for (int64_t i = 0; i < num; ++i)
      ret_data[i] = sampler->Draw();
    delete sampler;
  });
  return ret;
}

template IdArray RandomEngine::Choice<int32_t>(
    int64_t num, FloatArray prob, bool replace);
template IdArray RandomEngine::Choice<int64_t>(
    int64_t num, FloatArray prob, bool replace);

};  // namespace dgl
