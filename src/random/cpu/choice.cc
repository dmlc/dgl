/**
 *  Copyright (c) 2019 by Contributors
 * @file random/choice.cc
 * @brief Non-uniform discrete sampling implementation
 */

#include <dgl/array.h>
#include <dgl/random.h>

#include <numeric>
#include <vector>

#include "sample_utils.h"

namespace dgl {

template <typename IdxType>
IdxType RandomEngine::Choice(FloatArray prob) {
  IdxType ret = 0;
  ATEN_FLOAT_TYPE_SWITCH(prob->dtype, ValueType, "probability", {
    // TODO(minjie): allow choosing different sampling algorithms
    utils::TreeSampler<IdxType, ValueType, true> sampler(this, prob);
    ret = sampler.Draw();
  });
  return ret;
}

template int32_t RandomEngine::Choice<int32_t>(FloatArray);
template int64_t RandomEngine::Choice<int64_t>(FloatArray);

template <typename IdxType, typename FloatType>
void RandomEngine::Choice(
    IdxType num, FloatArray prob, IdxType* out, bool replace) {
  const IdxType N = prob->shape[0];
  if (!replace)
    CHECK_LE(num, N)
        << "Cannot take more sample than population when 'replace=false'";
  if (num == N && !replace) std::iota(out, out + num, 0);

  utils::BaseSampler<IdxType>* sampler = nullptr;
  if (replace) {
    sampler = new utils::TreeSampler<IdxType, FloatType, true>(this, prob);
  } else {
    sampler = new utils::TreeSampler<IdxType, FloatType, false>(this, prob);
  }
  for (IdxType i = 0; i < num; ++i) out[i] = sampler->Draw();
  delete sampler;
}

template void RandomEngine::Choice<int32_t, float>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, float>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, double>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, double>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, int8_t>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, int8_t>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);
template void RandomEngine::Choice<int32_t, uint8_t>(
    int32_t num, FloatArray prob, int32_t* out, bool replace);
template void RandomEngine::Choice<int64_t, uint8_t>(
    int64_t num, FloatArray prob, int64_t* out, bool replace);

template <typename IdxType>
void RandomEngine::UniformChoice(
    IdxType num, IdxType population, IdxType* out, bool replace) {
  CHECK_GE(num, 0) << "The numbers to sample should be non-negative.";
  CHECK_GE(population, 0) << "The population size should be non-negative.";
  if (!replace)
    CHECK_LE(num, population)
        << "Cannot take more sample than population when 'replace=false'";
  if (replace) {
    for (IdxType i = 0; i < num; ++i) out[i] = RandInt(population);
  } else {
    if (num <
        population / 10) {  // TODO(minjie): may need a better threshold here
      // if set of numbers is small (up to 128) use linear search to verify
      // uniqueness this operation is cheaper for CPU.
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

template <typename IdxType, typename FloatType>
void RandomEngine::BiasedChoice(
    IdxType num, const IdxType* split, FloatArray bias, IdxType* out,
    bool replace) {
  const int64_t num_tags = bias->shape[0];
  const FloatType* bias_data = static_cast<FloatType*>(bias->data);
  IdxType total_node_num = 0;
  FloatArray prob = NDArray::Empty({num_tags}, bias->dtype, bias->ctx);
  FloatType* prob_data = static_cast<FloatType*>(prob->data);
  for (int64_t tag = 0; tag < num_tags; ++tag) {
    int64_t tag_num_nodes = split[tag + 1] - split[tag];
    total_node_num += tag_num_nodes;
    FloatType tag_bias = bias_data[tag];
    prob_data[tag] = tag_num_nodes * tag_bias;
  }
  if (replace) {
    auto sampler = utils::TreeSampler<IdxType, FloatType, true>(this, prob);
    for (IdxType i = 0; i < num; ++i) {
      const int64_t tag = sampler.Draw();
      const IdxType tag_num_nodes = split[tag + 1] - split[tag];
      out[i] = RandInt(tag_num_nodes) + split[tag];
    }
  } else {
    utils::TreeSampler<int64_t, FloatType, false> sampler(
        this, prob, bias_data);
    CHECK_GE(total_node_num, num)
        << "Cannot take more sample than population when 'replace=false'";
    // we use hash set here. Maybe in the future we should support reservoir
    // algorithm
    std::vector<std::unordered_set<IdxType>> selected(num_tags);
    for (IdxType i = 0; i < num; ++i) {
      const int64_t tag = sampler.Draw();
      bool inserted = false;
      const IdxType tag_num_nodes = split[tag + 1] - split[tag];
      IdxType selected_node;
      while (!inserted) {
        CHECK_LT(selected[tag].size(), tag_num_nodes)
            << "Cannot take more sample than population when 'replace=false'";
        selected_node = RandInt(tag_num_nodes);
        inserted = selected[tag].insert(selected_node).second;
      }
      out[i] = selected_node + split[tag];
    }
  }
}

template void RandomEngine::BiasedChoice<int32_t, float>(
    int32_t, const int32_t*, FloatArray, int32_t*, bool);
template void RandomEngine::BiasedChoice<int32_t, double>(
    int32_t, const int32_t*, FloatArray, int32_t*, bool);
template void RandomEngine::BiasedChoice<int64_t, float>(
    int64_t, const int64_t*, FloatArray, int64_t*, bool);
template void RandomEngine::BiasedChoice<int64_t, double>(
    int64_t, const int64_t*, FloatArray, int64_t*, bool);

};  // namespace dgl
