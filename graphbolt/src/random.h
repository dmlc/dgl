
/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file random.h
 * @brief Random Engine class.
 */
#ifndef GRAPHBOLT_RANDOM_H_
#define GRAPHBOLT_RANDOM_H_

#include <mutex>
#include <optional>
#include <pcg_random.hpp>
#include <random>
#include <thread>

namespace graphbolt {

/**
 * @brief Thread-local Random Number Generator class.
 */
class RandomEngine {
 public:
  /** @brief Constructor with default seed. */
  RandomEngine();

  /** @brief Constructor with given seed. */
  explicit RandomEngine(uint64_t seed);
  explicit RandomEngine(uint64_t seed, uint64_t stream);

  /** @brief Get the thread-local random number generator instance. */
  static RandomEngine* ThreadLocal();

  /** @brief Set the seed. */
  void SetSeed(uint64_t seed);
  void SetSeed(uint64_t seed, uint64_t stream);

  /** @brief Protect manual seed accesses. */
  static std::mutex manual_seed_mutex;

  /** @brief Manually fix the seed. */
  static std::optional<uint64_t> manual_seed;
  static void SetManualSeed(int64_t seed);

  /**
   * @brief Generate a uniform random integer in [low, high).
   */
  template <typename T>
  T RandInt(T lower, T upper) {
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(rng_);
  }

  /**
   * @brief Generate a uniform random real number in [low, high).
   */
  template <typename T>
  T Uniform(T lower, T upper) {
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(rng_);
  }

  /**
   * @brief Generate random non-negative floating-point values according to
   * exponential distribution. Probability density function: P(x|λ) = λe^(-λx).
   */
  template <typename T>
  T Exponential(T lambda) {
    std::exponential_distribution<T> dist(lambda);
    return dist(rng_);
  }

 private:
  pcg32 rng_;
};

}  // namespace graphbolt

#endif  // GRAPHBOLT_RANDOM_H_
