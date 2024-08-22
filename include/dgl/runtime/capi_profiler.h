/*!
 *  Copyright (c) 2022 by Contributors
 * \file capi_profiler.h
 * \brief C API profiler
 */
#ifndef CAPI_PROFILER_H_
#define CAPI_PROFILER_H_

#include <chrono>
#include <dmlc/logging.h>
#include <iostream>
#include "packed_func.h"

using std::chrono::nanoseconds;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

namespace dgl {
namespace runtime {

class CApiProfiler {
 public:
  struct Stats {
    uint64_t count;
    uint64_t total_time;
  };

  static CApiProfiler& singleton() {
    static CApiProfiler singleton_;
    return singleton_;
  }

  static auto& Get() {
    return singleton();
  }

  static auto WrapCAPI(PackedFunc::FType f, std::string& fname) {
    return [&, f](DGLArgs args, DGLRetValue* rv) {
      if (!recording_) {
        f(args, rv);
        return;
      }

      auto tic = high_resolution_clock::now();
      f(args, rv);
      auto toc = high_resolution_clock::now();
      auto tm = duration_cast<nanoseconds>(toc - tic).count();

      auto &stats = CApiProfiler::Get().capi_stats_;
      if (stats.find(fname) == stats.end()) {
        stats[fname] = { .count = 1, .total_time = tm };
      } else {
        stats[fname].count += 1;
        stats[fname].total_time += tm;
      }
    };
  }

  void PrintStats();
  void Start();
  void Stop();

 private:
  CApiProfiler() = default;

  static bool recording_;

  std::map<std::string, Stats> capi_stats_{};
};

}
}

#endif // CAPI_PROFILER_H_