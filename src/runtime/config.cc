/**
 *  Copyright (c) 2019 by Contributors
 * @file runtime/config.cc
 * @brief DGL runtime config
 */

#include <dgl/runtime/config.h>
#include <dgl/runtime/registry.h>
#if !defined(_WIN32) && defined(USE_LIBXSMM)
#include <libxsmm_source.h>
#endif

using namespace dgl::runtime;

namespace dgl {
namespace runtime {

Config::Config() {
#if !defined(_WIN32) && defined(USE_LIBXSMM)
  int cpu_id = libxsmm_cpuid_x86();
  // Enable libxsmm on AVX machines by default
  libxsmm_ = LIBXSMM_X86_AVX2 <= cpu_id && cpu_id <= LIBXSMM_X86_ALLFEAT;
#else
  libxsmm_ = false;
#endif
}

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
