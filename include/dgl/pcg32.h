/*!
 *   Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *   For additional information about the PCG random number generation scheme,
 *   including its license and other licensing options, visit
 *
 *         http://www.pcg-random.org
 *
 *
 * @file dgl/pcg32.h
 * @brief An implementation of pcg32 PRNG, derived from
 *        https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
 */

#ifndef DGL_PCG32_H_
#define DGL_PCG32_H_

#include <cinttypes>
#include <limits>

struct pcg32 {
  using result_type = std::uint32_t;

  static const std::uint64_t default_stream = 721347520444481703ULL;

  constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }

  constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }

  result_type operator()() {
    std::uint64_t oldstate = state;
    state = oldstate * 6364136223846793005ULL + inc;
    std::uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    std::uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  void seed(std::uint64_t seed, std::uint64_t stream = default_stream) {
    state = 0;
    inc = 2 * stream + 1;
    this->operator()();
    state += seed;
    this->operator()();
  }

  pcg32() : state(0x853c49e6748fea9bULL), inc(1442695040888963407ULL) {}
  
  pcg32(std::uint64_t seed, std::uint64_t stream = default_stream) {
    this->seed(seed, stream);
  }

 private:
  std::uint64_t state;
  std::uint64_t inc;
};

#endif  // DGL_PCG32_H_
