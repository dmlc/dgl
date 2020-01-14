/*!
 *  Copyright (c) 2019 by Contributors
 * \file random/choice.cc
 * \brief Non-uniform discrete sampling implementation
 */

#include <dgl/sample_utils.h>
#include <dgl/random.h>
#include <algorithm>
#include <utility>
#include <queue>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <limits>
#include <vector>
#include "random.h"

namespace dgl {

template<typename IdxType>
IdxType RandomEngine::ChoiceImpl<kDLCPU>(FloatArray prob) {
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, DType, {
    TreeSampler<IdxType, DType, true> sampler(this, static_cast<DType *>(prob->data), prob->shape[0]);
    return sampler.Draw();
  });
}

template int32_t RandomEngine::ChoiceImpl<kDLCPU, int32_t>(FloatArray prob);
template int64_t RandomEngine::ChoiceImpl<kDLCPU, int64_t>(FloatArray prob);

};  // namespace dgl
