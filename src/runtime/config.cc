/*!
 *  Copyright (c) 2019 by Contributors
 * \file runtime/config.cc
 * \brief DGL runtime config
 */

#include <dgl/runtime/config.h>
#include <dgl/runtime/registry.h>

using namespace dgl::runtime;

namespace dgl {
namespace runtime {

void Config::EnableLibxsmm(bool b) { libxsmm_ = b; }

bool Config::IsLibxsmmAvailable() const { return libxsmm_; }

DGL_REGISTER_GLOBAL("global_config._CAPI_DGLConfigSetLibxsmm")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      bool use_libxsmm = args[0];
      dgl::runtime::Config::Global()->EnableLibxsmm(use_libxsmm);
    });

DGL_REGISTER_GLOBAL("global_config._CAPI_DGLConfigGetLibxsmm")
    .set_body([](DGLArgs args, DGLRetValue* rv) {
      *rv = dgl::runtime::Config::Global()->IsLibxsmmAvailable();
    });

}  // namespace runtime
}  // namespace dgl
