/*!
 *  Copyright (c) 2022 by Contributors
 * \file capi_profiler.cc
 * \brief C API profiler implementation
 */

#include <chrono>
#include <dgl/runtime/capi_profiler.h>
#include <dgl/runtime/registry.h>
#include <dgl/packed_func_ext.h>
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {
namespace runtime {

void CApiProfiler::Start() {
  if (recording_) {
    LOG(WARNING) << "Profiling has been started already";
  }
  recording_ = true;
}

void CApiProfiler::Stop() {
  if (!recording_) {
    LOG(WARNING) << "Profiling hasn't been started";
  }
  recording_ = false;
}

void CApiProfiler::PrintStats() {
  std::vector< std::pair< std::string, Stats > > stats(capi_stats_.begin(), capi_stats_.end());
  std::sort(stats.begin(), stats.end(), []( const auto& a, const auto& b ) {
    return a.second.total_time > b.second.total_time;
  });

  for (auto& v : stats) {
    std::cout << v.first << " " << v.second.total_time / 1000000.0f << " ms, count: " << v.second.count << std::endl;
  }
}

bool CApiProfiler::recording_{ false };

DGL_REGISTER_GLOBAL("utils.profiler._CAPI_DGLStartProfiling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CApiProfiler::Get().Start();
  });

DGL_REGISTER_GLOBAL("utils.profiler._CAPI_DGLStopProfiling")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CApiProfiler::Get().Stop();
  });

DGL_REGISTER_GLOBAL("utils.profiler._CAPI_DGLPrintStats")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CApiProfiler::Get().PrintStats();
  });

}
}