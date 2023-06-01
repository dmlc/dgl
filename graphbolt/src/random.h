/**
 *  Copyright (c) 2023 by Contributors
 * @file graphbolt/src/random.h
 * @brief Common macros for graphbolt package.
 */

#ifndef GRAPHBOLT_RANDOM_H_
#define GRAPHBOLT_RANDOM_H_

#include <random>

namespace graphbolt {
namespace sampling {

class RandomEngine {
 public:
  // Get the instance of the RandomEngine (Singleton)
  static RandomEngine& getInstance() {
    static RandomEngine instance;
    return instance;
  }

  /**
   * @brief Generate a uniform random integer in [lower, upper)
   */
  template <typename T>
  T RandInt(T lower, T upper);

  template <typename T>
  T RandInt(T upper) {
    return RandInt<T>(0, upper);
  }

  /**
   * @brief Pick random integers from population by uniform distribution.
   *
   * If replace is false, num must not be larger than population.
   *
   * @tparam IdxType Return integer type
   * @param num Number of integers to choose
   * @param population Total number of elements to choose from.
   * @param out The output buffer to write selected indices.
   * @param replace If true, choose with replacement.
   */
  template <typename IdxType>
  void UniformChoice(
      IdxType num, IdxType population, IdxType* out, bool replace = true);

  /**
   * @brief Pick random integers between 0 to N-1 according to given
   * probabilities
   *
   * If replace is false, the number of picked integers must not larger than N.
   *
   * @tparam IdxType Id type
   * @tparam ProbType Probability value type
   * @param num Number of integers to choose
   * @param prob Array of N unnormalized probability of each element.  Must be
   *        non-negative.
   * @param out The output buffer to write selected indices.
   * @param replace If true, choose with replacement.
   */
  template <typename IdxType, typename ProbType>
  void Choice(
      IdxType num, ProbType* prob, IdxType len, IdxType* out,
      bool replace = true);

 private:
  std::mt19937 rng_;

  // Private constructor to prevent instantiation
  RandomEngine() : rng_(std::random_device()()) {}
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_RANDOM_H_
