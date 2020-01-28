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

template<typename IdxType, typename ValueType>
IdxType RandomEngine::Choice(const std::vector<ValueType> &prob) {
  utils::TreeSampler<IdxType, ValueType, true> sampler(this, prob);
  return sampler.Draw();
}

template int32_t RandomEngine::Choice<int32_t, float>(const std::vector<float> &);
template int64_t RandomEngine::Choice<int64_t, float>(const std::vector<float> &);
template int32_t RandomEngine::Choice<int32_t, double>(const std::vector<double> &);
template int64_t RandomEngine::Choice<int64_t, double>(const std::vector<double> &);

};  // namespace dgl
