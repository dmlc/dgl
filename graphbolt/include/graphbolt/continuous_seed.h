/**
 *   Copyright (c) 2023, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *   All rights reserved.
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
 *
 * @file graphbolt/continuous_seed.h
 * @brief CPU and CUDA implementation for continuous random seeds
 */
#ifndef GRAPHBOLT_CONTINUOUS_SEED_H_
#define GRAPHBOLT_CONTINUOUS_SEED_H_

#include <torch/script.h>

#include <cmath>

#ifdef __CUDACC__
#include <curand_kernel.h>
#else
#include <pcg_random.hpp>
#include <random>
#endif  // __CUDA_ARCH__

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524401
#endif  // M_SQRT1_2

namespace graphbolt {

class continuous_seed {
  uint64_t s[2];
  float c[2];

 public:
  /* implicit */ continuous_seed(const int64_t seed) {  // NOLINT
    s[0] = s[1] = seed;
    c[0] = c[1] = 0;
  }

  continuous_seed(torch::Tensor seed_arr, float r) {
    auto seed = seed_arr.data_ptr<int64_t>();
    s[0] = seed[0];
    s[1] = seed[seed_arr.size(0) - 1];
    const auto pi = std::acos(-1.0);
    c[0] = std::cos(pi * r / 2);
    c[1] = std::sin(pi * r / 2);
  }

  uint64_t get_seed(int i) const { return s[i != 0]; }

#ifdef __CUDACC__
  __device__ inline float uniform(const uint64_t t) const {
    const uint64_t kCurandSeed = 999961;  // Could be any random number.
    curandStatePhilox4_32_10_t rng;
    curand_init(kCurandSeed, s[0], t, &rng);
    float rnd;
    if (s[0] != s[1]) {
      rnd = c[0] * curand_normal(&rng);
      curand_init(kCurandSeed, s[1], t, &rng);
      rnd += c[1] * curand_normal(&rng);
      rnd = normcdff(rnd);
    } else {
      rnd = curand_uniform(&rng);
    }
    return rnd;
  }
#else
  inline float uniform(const uint64_t t) const {
    pcg32 ng0(s[0], t);
    float rnd;
    if (s[0] != s[1]) {
      std::normal_distribution<float> norm;
      rnd = c[0] * norm(ng0);
      pcg32 ng1(s[1], t);
      norm.reset();
      rnd += c[1] * norm(ng1);
      rnd = std::erfc(-rnd * static_cast<float>(M_SQRT1_2)) / 2.0f;
    } else {
      std::uniform_real_distribution<float> uni;
      rnd = uni(ng0);
    }
    return rnd;
  }
#endif  // __CUDA_ARCH__
};

class single_seed {
  uint64_t seed_;

 public:
  /* implicit */ single_seed(const int64_t seed) : seed_(seed) {}  // NOLINT

  single_seed(torch::Tensor seed_arr)
      : seed_(seed_arr.data_ptr<int64_t>()[0]) {}

#ifdef __CUDACC__
  __device__ inline float uniform(const uint64_t id) const {
    const uint64_t kCurandSeed = 999961;  // Could be any random number.
    curandStatePhilox4_32_10_t rng;
    curand_init(kCurandSeed, seed_, id, &rng);
    return curand_uniform(&rng);
  }
#else
  inline float uniform(const uint64_t id) const {
    pcg32 ng0(seed_, id);
    std::uniform_real_distribution<float> uni;
    return uni(ng0);
  }
#endif  // __CUDA_ARCH__
};

}  // namespace graphbolt

#endif  // GRAPHBOLT_CONTINUOUS_SEED_H_
