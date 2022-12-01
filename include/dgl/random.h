/**
 * Copyright (c) 2017 by Contributors
 * @file dgl/random.h
 * @brief Random number generators
 */

#ifndef DGL_RANDOM_H_
#define DGL_RANDOM_H_

#include <dgl/array.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>

#include <random>
#include <thread>
#include <vector>

#include <pcg_random.hpp>

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

/**
 * @brief Thread-local Random Number Generator class
 */
class RandomEngine {
 public:
  /** @brief Constructor with default seed */
  RandomEngine() {
    std::random_device rd;
    SetSeed(rd());
  }

  /** @brief Constructor with given seed */
  explicit RandomEngine(uint64_t seed, uint64_t stream = GetThreadId()) {
    SetSeed(seed, stream);
  }

  /** @brief Get the thread-local random number generator instance */
  static RandomEngine* ThreadLocal() {
    return dmlc::ThreadLocalStore<RandomEngine>::Get();
  }

  /**
   * @brief Set the seed of this random number generator
   */
  void SetSeed(uint64_t seed, uint64_t stream = GetThreadId()) {
    rng_.seed(seed, stream);
  }

  /**
   * @brief Generate an arbitrary random 32-bit integer.
   */
  int32_t RandInt32() { return static_cast<int32_t>(rng_()); }

  /**
   * @brief Generate a uniform random integer in [0, upper)
   */
  template <typename T>
  T RandInt(T upper) {
    return RandInt<T>(0, upper);
  }

  /**
   * @brief Generate a uniform random integer in [lower, upper)
   */
  template <typename T>
  T RandInt(T lower, T upper) {
    CHECK_LT(lower, upper);
    std::uniform_int_distribution<T> dist(lower, upper - 1);
    return dist(rng_);
  }

  /**
   * @brief Generate a uniform random float in [0, 1)
   */
  template <typename T>
  T Uniform() {
    return Uniform<T>(0., 1.);
  }

  /**
   * @brief Generate a uniform random float in [lower, upper)
   */
  template <typename T>
  T Uniform(T lower, T upper) {
    // Although the result is in [lower, upper), we allow lower == upper as in
    // www.cplusplus.com/reference/random/uniform_real_distribution/uniform_real_distribution/
    CHECK_LE(lower, upper);
    std::uniform_real_distribution<T> dist(lower, upper);
    return dist(rng_);
  }

  /**
   * @brief Pick a random integer between 0 to N-1 according to given
   *        probabilities.
   * @tparam IdxType Return integer type.
   * @param prob Array of N unnormalized probability of each element. Must be
   *        non-negative.
   * @return An integer randomly picked from 0 to N-1.
   */
  template <typename IdxType>
  IdxType Choice(FloatArray prob);

  /**
   * @brief Pick random integers between 0 to N-1 according to given
   * probabilities
   *
   * If replace is false, the number of picked integers must not larger than N.
   *
   * @tparam IdxType Id type
   * @tparam FloatType Probability value type
   * @param num Number of integers to choose
   * @param prob Array of N unnormalized probability of each element.  Must be
   *        non-negative.
   * @param out The output buffer to write selected indices.
   * @param replace If true, choose with replacement.
   */
  template <typename IdxType, typename FloatType>
  void Choice(IdxType num, FloatArray prob, IdxType* out, bool replace = true);

  /**
   * @brief Pick random integers between 0 to N-1 according to given
   * probabilities
   *
   * If replace is false, the number of picked integers must not larger than N.
   *
   * @tparam IdxType Id type
   * @tparam FloatType Probability value type
   * @param num Number of integers to choose
   * @param prob Array of N unnormalized probability of each element.  Must be
   *        non-negative.
   * @param replace If true, choose with replacement.
   * @return Picked indices
   */
  template <typename IdxType, typename FloatType>
  IdArray Choice(IdxType num, FloatArray prob, bool replace = true) {
    const DGLDataType dtype{kDGLInt, sizeof(IdxType) * 8, 1};
    IdArray ret = IdArray::Empty({num}, dtype, prob->ctx);
    Choice<IdxType, FloatType>(
        num, prob, static_cast<IdxType*>(ret->data), replace);
    return ret;
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
   * @brief Pick random integers from population by uniform distribution.
   *
   * If replace is false, num must not be larger than population.
   *
   * @tparam IdxType Return integer type
   * @param num Number of integers to choose
   * @param population Total number of elements to choose from.
   * @param replace If true, choose with replacement.
   * @return Picked indices
   */
  template <typename IdxType>
  IdArray UniformChoice(IdxType num, IdxType population, bool replace = true) {
    const DGLDataType dtype{kDGLInt, sizeof(IdxType) * 8, 1};
    // TODO(minjie): only CPU implementation right now
    IdArray ret = IdArray::Empty({num}, dtype, DGLContext{kDGLCPU, 0});
    UniformChoice<IdxType>(
        num, population, static_cast<IdxType*>(ret->data), replace);
    return ret;
  }

  /**
   * @brief Pick random integers with different probability for different
   * segments.
   *
   * For example, if split=[0, 4, 10] and bias=[1.5, 1], it means to pick some
   * integers from 0 to 9, which is divided into two segments. 0-3 are in the
   * first segment and the rest belongs to the second. The weight(bias) of each
   * candidate in the first segment is upweighted to 1.5.
   *
   *  candidate | 0 1 2 3 | 4 5 6 7 8 9 |
   *  split       ^         ^            ^
   *  bias      |   1.5   |      1      |
   *
   *
   * The complexity of this operator is O(k * log(T)) where k is the number of
   * integers we want to pick, and T is the number of segments. It is much
   * faster compared with assigning probability for each candidate, of which the
   * complexity is O(k * log(N)) where N is the number of all candidates.
   *
   * If replace is false, num must not be larger than population.
   *
   * @tparam IdxType Return integer type
   * @param num Number of integers to choose
   * @param split Array of T+1 split positions of different segments(including
   *        start and end)
   * @param bias Array of T weight of each segments.
   * @param out The output buffer to write selected indices.
   * @param replace If true, choose with replacement.
   */
  template <typename IdxType, typename FloatType>
  void BiasedChoice(
      IdxType num, const IdxType* split, FloatArray bias, IdxType* out,
      bool replace = true);

  /**
   * @brief Pick random integers with different probability for different
   * segments.
   *
   * If replace is false, num must not be larger than population.
   *
   * @tparam IdxType Return integer type
   * @param num Number of integers to choose
   * @param split Split positions of different segments
   * @param bias Weights of different segments
   * @param replace If true, choose with replacement.
   */
  template <typename IdxType, typename FloatType>
  IdArray BiasedChoice(
      IdxType num, const IdxType* split, FloatArray bias, bool replace = true) {
    const DGLDataType dtype{kDGLInt, sizeof(IdxType) * 8, 1};
    IdArray ret = IdArray::Empty({num}, dtype, DGLContext{kDGLCPU, 0});
    BiasedChoice<IdxType, FloatType>(
        num, split, bias, static_cast<IdxType*>(ret->data), replace);
    return ret;
  }

 private:
  pcg32 rng_;
};

};  // namespace dgl

#endif  // DGL_RANDOM_H_
