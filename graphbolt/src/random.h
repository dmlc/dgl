
/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file random.h
 * @brief Random Engine class.
 */
#ifndef GRAPHBOLT_RANDOM_H_
#define GRAPHBOLT_RANDOM_H_

#include <dmlc/thread_local.h>

#include <pcg_random.hpp>
#include <random>
#include <thread>

namespace graphbolt {

namespace {

// Get a unique integer ID representing this thread.
inline uint32_t GetThreadId() {
  static int num_threads = 0;
  static std::mutex mutex;
  static thread_local int id = -1;

  if (id == -1) {
    std::lock_guard<std::mutex> guard(mutex);
    id = num_threads;
    num_threads++;
  }
  return id;
}

};  // namespace

/**
 * @brief Thread-local Random Number Generator class.
 */
class RandomEngine {
 public:
  /** @brief Constructor with default seed. */
  RandomEngine() {
    std::random_device rd;
    SetSeed(rd());
  }

  /** @brief Constructor with given seed. */
  explicit RandomEngine(uint64_t seed, uint64_t stream = GetThreadId()) {
    SetSeed(seed, stream);
  }

  /** @brief Get the thread-local random number generator instance. */
  static RandomEngine* ThreadLocal() {
    return dmlc::ThreadLocalStore<RandomEngine>::Get();
  }

  /** @brief Set the seed. */
  void SetSeed(uint64_t seed, uint64_t stream = GetThreadId()) {
    rng_.seed(seed, stream);
  }

  /**
   * @brief Generate a uniform random integer in [low, high).
   */
  template <typename T>
  T RandInt(T lower, T upper) {
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(rng_);
  }

 private:
  pcg32 rng_;
};

namespace labor {

template <typename T>
inline T uniform_random(int64_t random_seed, int64_t t) {
  pcg32 ng(random_seed, t);
  std::uniform_real_distribution<T> uni;
  return uni(ng);
}

template <typename T>
inline T invcdf(T u, int64_t n, T rem) {
  constexpr T one = 1;
  return rem * (one - std::pow(one - u, one / n));
}

template <typename T>
inline T jth_sorted_uniform_random(
    int64_t random_seed, int64_t t, int64_t c, int64_t j, T& rem, int64_t n) {
  const auto u = uniform_random<T>(random_seed, t + j * c);
  // https://mathematica.stackexchange.com/a/256707
  rem -= invcdf(u, n, rem);
  return 1 - rem;
}

};  // namespace labor

}  // namespace graphbolt

#endif  // GRAPHBOLT_RANDOM_H_
