/**
 *  Copyright (c) 2023 by Contributors
 * @file random_engine.h
 * @brief Random engine used for sampling.
 */

#ifndef GRAPHBOLT_RANDOM_ENGINE_H_
#define GRAPHBOLT_RANDOM_ENGINE_H_

#include <random>

namespace graphbolt {
namespace sampling {

class RandomEngine {
 public:
  // Get the instance of the RandomEngine (Singleton).
  static RandomEngine& getInstance() {
    static RandomEngine instance;
    return instance;
  }

  /**
   * Generates a random integer within [min, max].
   *
   * @tparam T The type of the integer.
   * @param min The lower bound (inclusive) of the range.
   * @param max The upper bound (inclusive) of the range.
   * @return A random integer within [min, max].
   */
  template <typename T>
  T RandInt(T min, T max);

  template <typename T>
  T RandInt(T max) {
    return RandInt<T>(0, max);
  }

  /**
   * @brief Pick random integers from population by uniform distribution.
   *
   * If replace is false, num must not be larger than population.
   *
   * @tparam T Return integer type
   * @param num Number of integers to choose
   * @param population Total number of elements to choose from.
   * @param out The output buffer to write selected indices.
   * @param replace Whether the sample is with or without replacement. Default
   * is True, meaning that a value of a can be selected multiple times.
   */
  template <typename T>
  void UniformChoice(T num, T population, T* out, bool replace = true);

  /**
   * @brief Pick random integers between 0 to N-1 according to given
   * probabilities
   *
   * If replace is false, the number of picked integers must not larger than N.
   *
   * @tparam T Id type
   * @tparam ProbType Probability value type
   * @param num Number of integers to choose
   * @param prob Array of N unnormalized probability of each element.  Must be
   *        non-negative.
   * @param out The output buffer to write selected indices.
   * @param replace Whether the sample is with or without replacement. Default
   * is True, meaning that a value of a can be selected multiple times.
   */
  template <typename T, typename ProbType>
  void Choice(T num, ProbType* prob, T len, T* out, bool replace = true);

 private:
  std::mt19937 rng_;

  // Private constructor to prevent instantiation
  RandomEngine() : rng_(std::random_device()()) {}
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_RANDOM_ENGINE_H_
