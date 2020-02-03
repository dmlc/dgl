/*!
 *  Copyright (c) 2019 by Contributors
 * \file random/choice.cc
 * \brief Non-uniform discrete sampling implementation
 */

#include <dgl/random.h>
#include <algorithm>
#include <utility>
#include <queue>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>
#include "sample_utils.h"

namespace dgl {

template<typename IdxType>
IdxType RandomEngine::Choice(FloatArray prob) {
  IdxType ret;
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, ValueType, "probability", {
    utils::TreeSampler<IdxType, ValueType, true> sampler(this, prob);
    ret = sampler.Draw();
  });
  return ret;
}

template int32_t RandomEngine::Choice<int32_t>(FloatArray);
template int64_t RandomEngine::Choice<int64_t>(FloatArray);

};  // namespace dgl
