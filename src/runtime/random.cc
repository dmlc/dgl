/*!
 *  Copyright (c) 2017 by Contributors
 * \file random.cc
 * \brief Random number generator interfaces
 */

#include <dgl/runtime/registry.h>
#include <dgl/runtime/random.h>

namespace dgl {
namespace runtime {

void Random::SetSeed(int seed) {
  GetInstance().Seed(seed);
}

template<typename IntType>
IntType Random::RandInt(IntType upper) {
  CHECK_LT(0, upper);
  auto *rng = GetInstance().Get(omp_get_thread_num());
  std::uniform_int_distribution<IntType> dist(0, upper - 1);
  return dist(*rng);
}

template int32_t Random::RandInt<int32_t>(int32_t);
template uint32_t Random::RandInt<uint32_t>(uint32_t);
template int64_t Random::RandInt<int64_t>(int64_t);
template uint64_t Random::RandInt<uint64_t>(uint64_t);

template<typename IntType>
IntType Random::RandInt(IntType lower, IntType upper) {
  CHECK_LT(lower, upper);
  auto *rng = GetInstance().Get(omp_get_thread_num());
  std::uniform_int_distribution<IntType> dist(lower, upper - 1);
  return dist(*rng);
}

template int32_t Random::RandInt<int32_t>(int32_t, int32_t);
template uint32_t Random::RandInt<uint32_t>(uint32_t, uint32_t);
template int64_t Random::RandInt<int64_t>(int64_t, int64_t);
template uint64_t Random::RandInt<uint64_t>(uint64_t, uint64_t);

template<typename FloatType>
FloatType Random::Uniform() {
  auto *rng = GetInstance().Get(omp_get_thread_num());
  std::uniform_real_distribution<FloatType> dist(0.f, 1.f);
  return dist(*rng);
}

template float Random::Uniform<float>();
template double Random::Uniform<double>();

template<typename FloatType>
FloatType Random::Uniform(FloatType lower, FloatType upper) {
  CHECK_LT(lower, upper);
  auto *rng = GetInstance().Get(omp_get_thread_num());
  std::uniform_real_distribution<FloatType> dist(lower, upper);
  return dist(*rng);
}

template float Random::Uniform<float>(float, float);
template double Random::Uniform<double>(double, double);

Random::Random() {
  Resize(omp_get_max_threads());
  Seed(std::chrono::system_clock::now().time_since_epoch().count());
}

std::mt19937_64 *Random::Get(int i) {
  return &rngs_[i];
}

void Random::Seed(int seed) {
  meta_rng_.seed(seed);
  for (auto &rng : rngs_)
    rng.seed(meta_rng_());
}

Random &Random::GetInstance() {
  static Random random;
  return random;
}

void Random::Resize(int size) {
  if (size <= rngs_.size())
    return;
  for (int i = rngs_.size(); i < size; ++i)
    rngs_.push_back(std::mt19937_64(meta_rng_()));
}

DGL_REGISTER_GLOBAL("rng._CAPI_SetSeed")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    int seed = args[0];
    Random::SetSeed(seed);
  });

};  // namespace runtime
};  // namespace dgl
