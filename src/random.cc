/*!
 *  Copyright (c) 2017 by Contributors
 * \file random.cc
 * \brief Random number generator interfaces
 */

#include <dmlc/omp.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/random.h>
#include <thread>
#include <random>

using namespace dgl::runtime;

namespace dgl {

namespace {

static std::hash<std::thread::id> kThreadIdHasher;

uint32_t GetThreadId() {
  return kThreadIdHasher(std::this_thread::get_id());
}

};

RandomEngine::RandomEngine() {
  std::random_device rd;
  SetSeed(rd());
}

RandomEngine::RandomEngine(uint32_t seed) {
  SetSeed(seed);
}

void RandomEngine::SetSeed(uint32_t seed) {
  rng_.seed(seed + GetThreadId());
}

template<typename T>
T RandomEngine::RandInt(T upper) {
  return RandInt<T>(0, upper);
}

template<typename T>
T RandomEngine::RandInt(T lower, T upper) {
  CHECK_LT(lower, upper);
  std::uniform_int_distribution<T> dist(lower, upper - 1);
  return dist(rng_);
}

template<typename T>
T RandomEngine::Uniform() {
  return Uniform<T>(0., 1.);
}

template<typename T>
T RandomEngine::Uniform(T lower, T upper) {
  CHECK_LT(lower, upper);
  std::uniform_real_distribution<T> dist(lower, upper);
  return dist(rng_);
}

RandomEngine *RandomEngine::ThreadLocal() {
  return dmlc::ThreadLocalStore<RandomEngine>::Get();
}

DGL_REGISTER_GLOBAL("rng._CAPI_SetSeed")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    int seed = args[0];
#pragma omp parallel for
    for (int i = 0; i < omp_get_max_threads(); ++i)
      RandomEngine::ThreadLocal()->SetSeed(seed);
  });

// explicit instantiations
template int32_t RandomEngine::RandInt<int32_t>(int32_t);
template uint32_t RandomEngine::RandInt<uint32_t>(uint32_t);
template int64_t RandomEngine::RandInt<int64_t>(int64_t);
template uint64_t RandomEngine::RandInt<uint64_t>(uint64_t);

template int32_t RandomEngine::RandInt<int32_t>(int32_t, int32_t);
template uint32_t RandomEngine::RandInt<uint32_t>(uint32_t, uint32_t);
template int64_t RandomEngine::RandInt<int64_t>(int64_t, int64_t);
template uint64_t RandomEngine::RandInt<uint64_t>(uint64_t, uint64_t);

template float RandomEngine::Uniform<float>();
template double RandomEngine::Uniform<double>();

template float RandomEngine::Uniform<float>(float, float);
template double RandomEngine::Uniform<double>(double, double);

};  // namespace dgl
