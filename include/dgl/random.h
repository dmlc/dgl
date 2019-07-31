/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/random.h
 * \brief Random number generators
 */

#ifndef DGL_RANDOM_H_
#define DGL_RANDOM_H_

#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#include <random>
#include <thread>

namespace dgl {

using namespace dgl::runtime;

namespace {

inline uint32_t GetThreadId() {
  static std::hash<std::thread::id> kThreadIdHasher;
  return kThreadIdHasher(std::this_thread::get_id());
}

};  // namespace

/*!
 * \brief Thread-local Random Number Generator class
 */
class RandomEngine {
 public:
  /*! \brief Constructor with default seed */
  RandomEngine() {
    std::random_device rd;
    SetSeed(rd());
  }

  /*! \brief Constructor with given seed */
  explicit RandomEngine(uint32_t seed) {
    SetSeed(seed);
  }

  /*! \brief Get the thread-local random number generator instance */
  static RandomEngine *ThreadLocal() {
    return dmlc::ThreadLocalStore<RandomEngine>::Get();
  }

  /*!
   * \brief Set the seed of this random number generator
   */
  void SetSeed(uint32_t seed) {
    rng_.seed(seed + GetThreadId());
  }

  /*!
   * \brief Generate a uniform random integer in [0, upper)
   */
  template<typename T>
  T RandInt(T upper) {
    return RandInt<T>(0, upper);
  }

  /*!
   * \brief Generate a uniform random integer in [lower, upper)
   */
  template<typename T>
  T RandInt(T lower, T upper) {
    CHECK_LT(lower, upper);
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(rng_);
  }

  /*!
   * \brief Generate a uniform random float in [0, 1)
   */
  template<typename T>
  T Uniform() {
    return Uniform<T>(0., 1.);
  }

  /*!
   * \brief Generate a uniform random float in [lower, upper)
   */
  template<typename T>
  T Uniform(T lower, T upper) {
    CHECK_LT(lower, upper);
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(rng_);
  }

 private:
  std::mt19937 rng_;
};

};  // namespace dgl

#endif  // DGL_RANDOM_H_
