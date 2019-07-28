/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/random.h
 * \brief Random number generators
 */

#ifndef DGL_RANDOM_H_
#define DGL_RANDOM_H_

#include <random>
#include <vector>

namespace dgl {

/*!
 * \brief Thread-local Random Number Generator class
 */
class RandomEngine {
 public:
  /*! \brief Constructor with default seed */
  RandomEngine();

  /*! \brief Constructor with given seed */
  explicit RandomEngine(uint32_t seed);

  /*! \brief Get the thread-local random number generator instance */
  static RandomEngine *ThreadLocal();

  /*!
   * \brief Set the seed of this random number generator
   */
  void SetSeed(uint32_t seed);

  /*!
   * \brief Generate a uniform random integer in [0, upper)
   */
  template<typename T>
  T RandInt(T upper);

  /*!
   * \brief Generate a uniform random integer in [lower, upper)
   */
  template<typename T>
  T RandInt(T lower, T upper);

  /*!
   * \brief Generate a uniform random float in [0, 1)
   */
  template<typename T>
  T Uniform();

  /*!
   * \brief Generate a uniform random float in [lower, upper)
   */
  template<typename T>
  T Uniform(T lower, T upper);

 private:
  std::mt19937 rng_;
};

};  // namespace dgl

#endif  // DGL_RANDOM_H_
