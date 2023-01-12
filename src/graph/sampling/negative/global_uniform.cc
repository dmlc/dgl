/**
 *  Copyright (c) 2021 by Contributors
 * @file graph/sampling/negative/global_uniform.cc
 * @brief Global uniform negative sampling.
 */

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/sampling/negative.h>

#include <utility>

#include "../../../c_api_common.h"

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

std::pair<IdArray, IdArray> GlobalUniformNegativeSampling(
    HeteroGraphPtr hg, dgl_type_t etype, int64_t num_samples, int num_trials,
    bool exclude_self_loops, bool replace, double redundancy) {
  auto format = hg->SelectFormat(etype, CSC_CODE | CSR_CODE);
  if (format == SparseFormat::kCSC) {
    CSRMatrix csc = hg->GetCSCMatrix(etype);
    CSRSort_(&csc);
    std::pair<IdArray, IdArray> result = CSRGlobalUniformNegativeSampling(
        csc, num_samples, num_trials, exclude_self_loops, replace, redundancy);
    // reverse the pair since it is CSC
    return {result.second, result.first};
  } else if (format == SparseFormat::kCSR) {
    CSRMatrix csr = hg->GetCSRMatrix(etype);
    CSRSort_(&csr);
    return CSRGlobalUniformNegativeSampling(
        csr, num_samples, num_trials, exclude_self_loops, replace, redundancy);
  } else {
    LOG(FATAL)
        << "COO format is not supported in global uniform negative sampling";
    return {IdArray(), IdArray()};
  }
}

DGL_REGISTER_GLOBAL("sampling.negative._CAPI_DGLGlobalUniformNegativeSampling")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      HeteroGraphRef hg = args[0];
      dgl_type_t etype = args[1];
      CHECK_LE(etype, hg->NumEdgeTypes()) << "invalid edge type " << etype;
      int64_t num_samples = args[2];
      int num_trials = args[3];
      bool exclude_self_loops = args[4];
      bool replace = args[5];
      double redundancy = args[6];
      List<Value> result;
      std::pair<IdArray, IdArray> ret = GlobalUniformNegativeSampling(
          hg.sptr(), etype, num_samples, num_trials, exclude_self_loops,
          replace, redundancy);
      result.push_back(Value(MakeValue(ret.first)));
      result.push_back(Value(MakeValue(ret.second)));
      *rv = result;
    });

};  // namespace sampling
};  // namespace dgl
