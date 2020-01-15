/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/randomwalks.cc
 * \brief Dispatcher of different DGL random walks by device and data type
 */

#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dgl/array.h>
#include <utility>
#include "../../c_api_common.h"
#include "randomwalks_impl.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {

namespace sampling {

std::pair<IdArray, TypeArray> RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray etypes,
    const List<FloatArray> &prob) {
  EXPECT_INT(seeds, "seeds");
  EXPECT_INT(etypes, "etypes");
  EXPECT_NDIM(seeds, 1, "seeds");
  EXPECT_NDIM(etypes, 1, "etypes");
  for (FloatArray p : prob) {
    EXPECT_FLOAT(p, "probability");
    if (p->ndim != 0) {
      EXPECT_NDIM(p, 1, "probability");
    }
  }

  std::pair<IdArray, TypeArray> result;
  ATEN_XPU_SWITCH(hg->Context().device_type, XPU, {
    result = impl::RandomWalkImpl<XPU>(hg, seeds, etypes, prob);
  });

  return result;
}

};  // namespace sampling

};  // namespace dgl
