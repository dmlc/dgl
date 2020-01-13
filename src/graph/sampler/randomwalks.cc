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

  std::pair<IdArray, TypeArray> result;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    result = impl::RandomWalkImpl<XPU>(hg, seeds, etypes, prob);
  });

  return result;
}

};  // namespace sampling

};  // namespace dgl
