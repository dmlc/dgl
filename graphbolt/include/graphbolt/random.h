// Add for RandomEngine.
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

}  // namespace graphbolt