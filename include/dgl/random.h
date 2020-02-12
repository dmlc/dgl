/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/random.h
 * \brief Random number generators
 */

#ifndef DGL_RANDOM_H_
#define DGL_RANDOM_H_

#include <dgl/array.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#include <random>
#include <thread>
#include <vector>

namespace dgl {

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
    // Although the result is in [lower, upper), we allow lower == upper as in
    // www.cplusplus.com/reference/random/uniform_real_distribution/uniform_real_distribution/
    CHECK_LE(lower, upper);
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(rng_);
  }

  /*!
   * \brief Pick a random integer between 0 to N-1 according to given probabilities
   * \tparam IdxType Return integer type
   * \param prob Array of N unnormalized probability of each element.  Must be non-negative.
   * \return An integer randomly picked from 0 to N-1.
   */
  template<typename IdxType>
  IdxType Choice(FloatArray prob);

  /*!
   * \brief Pick random integers between 0 to N-1 according to given probabilities
   * 
   * If replace is false, the number of picked integers must not larger than N.
   *
   * \tparam IdxType Id type
   * \tparam FloatType Probability value type
   * \param num Number of integers to choose
   * \param prob Array of N unnormalized probability of each element.  Must be non-negative.
   * \param replace If true, choose with replacement.
   * \return Integer array
   */
  template <typename IdxType, typename FloatType>
  IdArray Choice(int64_t num, FloatArray prob, bool replace = true);

  /*!
   * \brief Pick random integers from population by uniform distribution.
   *
   * If replace is false, num must not be larger than population.
   *
   * \tparam IdxType Return integer type
   * \param num Number of integers to choose
   * \param population Total number of elements to choose from.
   * \param replace If true, choose with replacement.
   * \return Integer array
   */
  template <typename IdxType>
  IdArray UniformChoice(int64_t num, int64_t population, bool replace = true);

 private:
  std::default_random_engine rng_;
};

};  // namespace dgl

#endif  // DGL_RANDOM_H_
