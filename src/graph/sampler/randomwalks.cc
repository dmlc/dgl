/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/randomwalks.cc
 * \brief Dispatcher of different DGL random walks by device and data type
 */

#include <dgl/runtime/object.h>
#include <dgl/array.h>
#include <utility>
#include "../../c_api_common.h"
#include "randomwalk.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray etypes,
    const FloatArray prob) {
  CHECK_IDARRAY(seeds);
  CHECK_IDARRAY(etypes, 1);
  CHECK_FLOATARRAY(prob, 1);

  int64_t num_seeds = seeds->shape[0];
  int64_t trace_length = etypes->shape[0] + 1;

  std::pair<IdArray, TypeArray> result;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    ATEN_ID_TYPE_SWITCH(seeds->dtype, Idx, {
      ATEN_ID_TYPE_SWITCH(etypes->dtype, Type, {
        result = impl::RandomWalk<XPU, Idx, Type>(hg, seeds, etypes, prob);
      });
    });
  });

  return result;
}

};  // namespace sampling

};  // namespace dgl
