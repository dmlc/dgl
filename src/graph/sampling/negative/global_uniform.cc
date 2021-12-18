/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/sampling/negative/global_uniform.cc
 * \brief Global uniform negative sampling.
 */

#include <dgl/array.h>
#include <dgl/sampling/negative.h>
#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <utility>

using namespace dgl::runtime;
using namespace dgl::aten;

namespace dgl {
namespace sampling {

std::pair<IdArray, IdArray> GlobalUniformNegativeSampling(
    HeteroGraphPtr hg,
    dgl_type_t etype,
    int64_t num_samples,
    int num_trials,
    bool exclude_self_loops) {
  dgl_format_code_t allowed = hg->GetAllowedFormats();

  CHECK(FORMAT_HAS_CSR(allowed) || FORMAT_HAS_CSC(allowed)) <<
    "Global uniform negative sampling requires either CSR or CSC format to work.";

  switch (hg->SelectFormat(etype, CSC_CODE | CSR_CODE)) {
   case SparseFormat::kCSC:
    CSRMatrix csc = hg->GetCSCMatrix(etype);
    CSRSort_(&csc);
    std::pair<IdArray, IdArray> result = CSRGlobalUniformNegativeSampling(
        csc, num_samples, num_trials, exclude_self_loops);
    // reverse the pair since it is CSC
    return {pair.second, pair.first};
   case SparseFormat::kCSR:
    CSRMatrix csr = hg->GetCSRMatrix(etype);
    CSRSort_(&csr);
    return CSRGlobalUniformNegativeSampling(csr, num_samples, num_trials, exclude_self_loops);
  }
  // NOTREACHED
  return {IdArray(), IdArray()};
}

DGL_REGISTER_GLOBAL("sampling.negative._CAPI_DGLGlobalUniformNegativeSampling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    CHECK_LE(etype, hg->NumEdgeTypes()) << "invalid edge type " << etype;
    int64_t num_samples = args[2];
    int num_trials = args[3];
    bool exclude_self_loops = args[4];
    List<Value> result;
    std::pair<IdArray, IdArray> ret = GlobalUniformNegativeSampling(
        hg, etype, num_samples, num_trials, exclude_self_loops);
    result.push_back(Value(MakeValue(ret.first)));
    result.push_back(Value(MakeValue(ret.second)));
    *rv = result;
  });

};  // namespace sampling
};  // namespace dgl
