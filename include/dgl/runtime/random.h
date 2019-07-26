/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/runtime/random.h
 * \brief Random number generators
 */

#ifndef DGL_RUNTIME_RANDOM_H_
#define DGL_RUNTIME_RANDOM_H_

#include <dmlc/logging.h>
#include <dmlc/omp.h>
#include <random>
#include <vector>
#include <chrono>

namespace dgl {
namespace runtime {

/*!
 * \brief OpenMP thread-safe Random Number Generator.
 *
 * This is only intended for OpenMP!  It does so by creating omp_get_num_threads()
 * random number generators.
 */
class Random {
 public:
  /*!
   * \brief Set the seed of the DGL C random number generator.
   * \param seed The seed.
   */
  static void SetSeed(int seed);

  /*!
   * \brief Generate a random integer uniformly from 0 (inclusive) to upper (exclusive).
   * \param upper The upper bound (exclusive)
   */
  template<typename IntType>
  static IntType RandInt(IntType upper);

  /*!
   * \brief Generate a random integer uniformly from lower (inclusive) to upper (exclusive).
   * \param lower The lower bound (inclusive)
   * \param upper The upper bound (exclusive)
   */
  template<typename IntType>
  static IntType RandInt(IntType lower, IntType upper);

  /*!
   * \brief Generate a random float uniformly from 0 (inclusive) to 1 (exclusive).
   */
  template<typename FloatType>
  static FloatType Uniform();

  /*!
   * \brief Generate a random float uniformly from lower (inclusive) to upper (exclusive).
   * \param lower The lower bound (inclusive)
   * \param upper The upper bound (exclusive)
   */
  template<typename FloatType>
  static FloatType Uniform(FloatType lower, FloatType upper);

 private:
  /*! \brief Private constructor that initializes seed by time */
  Random();

  /*! \brief Get the RNG for thread ID i */
  std::mt19937_64 *Get(int i);

  /*! \brief Get the singleton instance */
  static Random &GetInstance();

  /*! \brief Set the seed of meta RNG, then reinitialize the thread-specific RNGs */
  void Seed(int seed);

  /*! \brief Resize the RNG array */
  void Resize(int size);

  /*! \brief Meta RNG that seeds the thread-specific RNGs */
  std::mt19937 meta_rng_;
  /*! \brief Thread-specific RNGs that have omp_get_num_threads() elements */
  std::vector<std::mt19937_64> rngs_;
};

};  // namespace runtime
};  // namespace dgl

#endif  // DGL_RUNTIME_RANDOM_H_
