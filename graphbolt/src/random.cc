#include "random.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace graphbolt {
namespace sampling {

template <typename T>
T RandomEngine::RandInt(T lower, T upper) {
  std::uniform_int_distribution<T> dist(lower, upper - 1);
  return dist(rng_);
}

template <typename IdxType>
void RandomEngine::UniformChoice(
    IdxType num, IdxType population, IdxType* out, bool replace) {
  if (replace) {
    for (IdxType i = 0; i < num; ++i) out[i] = RandInt(population);
  } else {
    if (num < population / 10) {
      if (num && num < 64) {
        *out = RandInt(population);
        auto b = out + 1;
        auto e = b + num - 1;
        while (b != e) {
          // put the new value at the end
          *b = RandInt(population);
          // Check if a new value doesn't exist in current range(out,b)
          // otherwise get a new value until we haven't unique range of
          // elements.
          auto it = std::find(out, b, *b);
          if (it != b) continue;
          ++b;
        }
      } else {
        // use hash set
        // In the best scenario, time complexity is O(num), i.e., no conflict.
        //
        // Let k be num / population, the expected number of extra sampling
        // steps is roughly k^2 / (1-k) * population, which means in the worst
        // case scenario, the time complexity is O(population^2). In practice,
        // we use 1/10 since std::unordered_set is pretty slow.
        std::unordered_set<IdxType> selected;
        while (static_cast<IdxType>(selected.size()) < num) {
          selected.insert(RandInt(population));
        }
        std::copy(selected.begin(), selected.end(), out);
      }

    } else {
      // In this case, `num >= population / 10`. To reduce the computation
      // overhead, we should reduce the number of random number generations.
      // Even though reservior algorithm is more memory effficient (it has
      // O(num) memory complexity), it generates O(population) random numbers,
      // which is computationally expensive. This algorithm has memory
      // complexity of O(population) but generates much fewer random numbers
      // O(num). In the case of `num >= population/10`, we don't need to worry
      // about memory complexity because `num` is usually small. So is
      // `population`. Allocating a small piece of memory is very efficient.
      std::vector<IdxType> seq(population);
      for (size_t i = 0; i < seq.size(); i++) seq[i] = i;
      for (IdxType i = 0; i < num; i++) {
        IdxType j = RandInt(i, population);
        std::swap(seq[i], seq[j]);
      }
      // Save the randomly sampled numbers.
      for (IdxType i = 0; i < num; i++) {
        out[i] = seq[i];
      }
    }
  }
}

template void RandomEngine::UniformChoice<int32_t>(
    int32_t num, int32_t population, int32_t* out, bool replace);
template void RandomEngine::UniformChoice<int64_t>(
    int64_t num, int64_t population, int64_t* out, bool replace);

// Perform a biased choice using probabilities for each element with optional
// replacement
template <typename IdxType, typename ProbType>
void RandomEngine::Choice(
    IdxType num, ProbType* prob, IdxType len, IdxType* out,
    bool replace) {
  std::discrete_distribution<IdxType> dist(prob, prob + len);
  if (replace) {
    for (auto i = 0; i < num; i++) out[i] = dist(rng_);
  } else {
    // Naive implementation, will be changed later.
    std::vector<ProbType> prob_vec(prob, prob + len);
    for (auto i = 0; i < num; i++) {
      out[i] = dist(rng_);
      prob_vec[out[i]] = 0;
      dist = std::discrete_distribution<IdxType>(prob_vec.begin(), prob_vec.end());
    }
  }
}

template void RandomEngine::Choice<int32_t, float>(
    int32_t num, float* prob, int32_t len, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, float>(
    int64_t num, float* prob, int64_t len, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, double>(
    int32_t num, double* prob, int32_t len, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, double>(
    int64_t num, double* prob, int64_t len, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, int8_t>(
    int32_t num, int8_t* prob, int32_t len, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, int8_t>(
    int64_t num, int8_t* prob, int64_t len, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, uint8_t>(
    int32_t num, uint8_t* prob, int32_t len, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, uint8_t>(
    int64_t num, uint8_t* prob, int64_t len, int64_t* out, bool replace);

}  // namespace sampling
}  // namespace graphbolt