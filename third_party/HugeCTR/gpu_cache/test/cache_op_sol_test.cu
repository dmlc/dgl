/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <omp.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <hps/embedding_cache_gpu.hpp>
#include <iostream>
#include <nv_gpu_cache.hpp>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The key generator
template <typename T, typename set_hasher = MurmurHash3_32<T>>
class KeyGenerator {
 public:
  KeyGenerator() : gen_(rd_()) {}
  KeyGenerator(T min, T max) : gen_(rd_()), distribution_(min, max) {}

  void fill_unique(T* data, size_t keys_per_set, size_t num_of_set, T empty_value) {
    if (keys_per_set == 0 || num_of_set == 0) {
      return;
    }
    assert(distribution_.max() - distribution_.min() >= keys_per_set * num_of_set);

    std::unordered_set<T> set;
    std::vector<size_t> set_sz(num_of_set, 0);
    size_t sz = 0;
    while (sz < keys_per_set * num_of_set) {
      T x = distribution_(gen_);
      if (x == empty_value) {
        continue;
      }
      auto res = set.insert(x);
      if (res.second) {
        size_t src_set = set_hasher::hash(x) % num_of_set;
        if (set_sz[src_set] < keys_per_set) {
          data[src_set * keys_per_set + set_sz[src_set]] = x;
          set_sz[src_set]++;
          sz++;
        }
      }
    }
    assert(sz == keys_per_set * num_of_set);
    for (size_t i = 0; i < num_of_set; i++) {
      assert(set_sz[i] == keys_per_set);
    }
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<T> distribution_;
};
// The random number generator
template <typename T>
class IntGenerator {
 public:
  IntGenerator() : gen_(rd_()) {}
  IntGenerator(T min, T max) : gen_(rd_()), distribution_(min, max) {}

  void fill_unique(T* data, size_t len, T empty_value) {
    if (len == 0) {
      return;
    }
    assert(distribution_.max() - distribution_.min() >= len);

    std::unordered_set<T> set;
    size_t sz = 0;
    while (sz < len) {
      T x = distribution_(gen_);
      if (x == empty_value) {
        continue;
      }
      auto res = set.insert(x);
      if (res.second) {
        data[sz++] = x;
      }
    }
    assert(sz == set.size());
    assert(sz == len);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::uniform_int_distribution<T> distribution_;
};

template <typename T>
class IntGenerator_normal {
 public:
  IntGenerator_normal() : gen_(rd_()) {}
  IntGenerator_normal(double mean, double dev) : gen_(rd_()), distribution_(mean, dev) {}

  void fill_unique(T* data, size_t len, T min, T max) {
    if (len == 0) {
      return;
    }

    std::unordered_set<T> set;
    size_t sz = 0;
    while (sz < len) {
      T x = (T)(abs(distribution_(gen_)));
      if (x < min || x > max) {
        continue;
      }
      auto res = set.insert(x);
      if (res.second) {
        data[sz++] = x;
      }
    }
    assert(sz == set.size());
    assert(sz == len);
  }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<double> distribution_;
};

// Utility to fill len embedding vector
template <typename KeyType>
void fill_vec(const KeyType* keys, float* vals, size_t embedding_vec_size, size_t len,
              float ratio) {
  for (size_t i = 0; i < len; ++i) {
    for (size_t j = 0; j < embedding_vec_size; ++j) {
      vals[i * embedding_vec_size + j] = (float)(ratio * keys[i]);
    }
  }
}

// Floating-point compare function
template <typename T>
bool is_near(T a, T b) {
  double diff = abs(a - b);
  bool ret = diff <= std::min(a, b) * 1e-6;
  if (!ret) {
    std::cerr << "error: " << a << " != " << b << "; diff = " << diff << std::endl;
  }
  return ret;
}

// Check correctness of result
template <typename KeyType>
void check_result(const KeyType* keys, const float* vals, size_t embedding_vec_size, size_t len,
                  float ratio) {
  for (size_t i = 0; i < len; ++i) {
    for (size_t j = 0; j < embedding_vec_size; ++j) {
      assert(is_near(vals[i * embedding_vec_size + j], (float)(ratio * keys[i])));
    }
  }
}

// Compare two sequence of keys and check whether they are the same(but with different order)
template <typename KeyType>
void compare_key(const KeyType* sequence_a, const KeyType* sequence_b, size_t len) {
  // Temp buffers for sorting
  KeyType* sequence_a_copy = (KeyType*)malloc(len * sizeof(KeyType));
  KeyType* sequence_b_copy = (KeyType*)malloc(len * sizeof(KeyType));
  // Copy data to temp buffers
  memcpy(sequence_a_copy, sequence_a, len * sizeof(KeyType));
  memcpy(sequence_b_copy, sequence_b, len * sizeof(KeyType));
  // Sort both arrays
  std::sort(sequence_a_copy, sequence_a_copy + len);
  std::sort(sequence_b_copy, sequence_b_copy + len);

  // Linearly compare elements
  for (size_t i = 0; i < len; i++) {
    assert(sequence_a_copy[i] == sequence_b_copy[i]);
  }
  // Free temp buffers
  free(sequence_a_copy);
  free(sequence_b_copy);
}

/* Timing funtion */
double W_time() {
  timeval marker;
  gettimeofday(&marker, NULL);
  return ((double)(marker.tv_sec) * 1e6 + (double)(marker.tv_usec)) * 1e-6;
}

using key_type = uint32_t;
using ref_counter_type = uint64_t;

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: " << argv[0]
              << " embedding_table_size cache_capacity_in_set embedding_vec_size query_length "
                 "iter_round num_threads cache_type"
              << std::endl;
    return -1;
  }

  // Arguments
  const size_t emb_size = atoi(argv[1]);
  const size_t cache_capacity_in_set = atoi(argv[2]);
  const size_t embedding_vec_size = atoi(argv[3]);
  const size_t query_length = atoi(argv[4]);
  const size_t iter_round = atoi(argv[5]);
  const size_t num_threads = atoi(argv[6]);
  const size_t cache_type = atoi(argv[7]);

  // Since cache is designed for single-gpu, all threads just use GPU 0
  CUDA_CHECK(cudaSetDevice(0));

  // Host side buffers shared between threads
  key_type* h_keys;  // Buffer holding all keys in embedding table
  float* h_vals;     // Buffer holding all values in embedding table

  // host-only buffers placed in normal host memory
  h_keys = (key_type*)malloc(emb_size * sizeof(key_type));
  h_vals = (float*)malloc(emb_size * embedding_vec_size * sizeof(float));

  // The empty key to be used
  const key_type empty_key = std::numeric_limits<key_type>::max();
  gpu_cache::gpu_cache_api<key_type>* cache = nullptr;

  // The cache to be used, by default the set hasher is based on MurMurHash and slab hasher is based
  // on Mod.

  if (cache_type == 0) {
    using Cache_ =
        gpu_cache::gpu_cache<key_type, ref_counter_type, empty_key, SET_ASSOCIATIVITY, SLAB_SIZE>;
    cache = new Cache_(cache_capacity_in_set, embedding_vec_size);
  } else {
    cache = new HugeCTR::EmbeddingCacheWrapper<key_type>(cache_capacity_in_set, embedding_vec_size);
  }

  // For random unique keys generation
  IntGenerator<key_type> gen_key;
  float increase = 0.1f;

  // Start 1st test
  std::cout << "****************************************" << std::endl;
  std::cout << "****************************************" << std::endl;
  std::cout << "Start Single-GPU Thread-safe Query and Replace API test " << std::endl;

  // Timimg variables
  double time_a;
  double time_b;

  time_a = W_time();

  std::cout << "Filling data" << std::endl;
  // Generating random unique keys
  gen_key.fill_unique(h_keys, emb_size, empty_key);
  // Filling values vector according to the keys
  fill_vec(h_keys, h_vals, embedding_vec_size, emb_size, increase);

  // Elapsed wall time
  time_b = W_time() - time_a;
  std::cout << "The Elapsed time for filling data is: " << time_b << "sec." << std::endl;

  // Insert <k,v> pairs to embedding table (CPU hashtable)
  time_a = W_time();

  std::cout << "Filling embedding table" << std::endl;
  std::unordered_map<key_type, std::vector<float>> h_emb_table;
  for (size_t i = 0; i < emb_size; i++) {
    std::vector<float> emb_vec(embedding_vec_size);
    for (size_t j = 0; j < embedding_vec_size; j++) {
      emb_vec[j] = h_vals[i * embedding_vec_size + j];
    }
    h_emb_table.emplace(h_keys[i], emb_vec);
  }

  // Elapsed wall time
  time_b = W_time() - time_a;
  std::cout << "The Elapsed time for filling embedding table is: " << time_b << "sec." << std::endl;

  // Free value buffer
  free(h_vals);

#pragma omp parallel default(none)                                                           \
    shared(h_keys, cache, h_emb_table, increase, embedding_vec_size, query_length, emb_size, \
           iter_round, std::cout, cache_type) num_threads(num_threads)
  {
    // The thread ID for this thread
    int thread_id = omp_get_thread_num();
    printf("Worker %d starts testing cache.\n", thread_id);
    // Since cache is designed for single-gpu, all threads just use GPU 0
    CUDA_CHECK(cudaSetDevice(0));

    // Thread-private host side buffers
    size_t* h_query_keys_index;  // Buffer holding index for keys to be queried
    key_type* h_query_keys;      // Buffer holding keys to be queried
    float* h_vals_retrieved;     // Buffer holding values retrieved from query
    key_type* h_missing_keys;    // Buffer holding missing keys from query
    float* h_missing_vals;       // Buffer holding values for missing keys
    uint64_t* h_missing_index;   // Buffers holding missing index from query
    size_t h_missing_len;        // missing length

    // Thread-private device side buffers
    key_type* d_query_keys;     // Buffer holding keys to be queried
    float* d_vals_retrieved;    // Buffer holding values retrieved from query
    key_type* d_missing_keys;   // Buffer holding missing keys from query
    float* d_missing_vals;      // Buffer holding values for missing keys
    uint64_t* d_missing_index;  // Buffers holding missing index from query
    size_t* d_missing_len;      // missing length

    // host-only buffers placed in normal host memory
    h_query_keys_index = (size_t*)malloc(query_length * sizeof(size_t));
    // host-device interactive buffers placed in pinned memory
    CUDA_CHECK(cudaHostAlloc((void**)&h_query_keys, query_length * sizeof(key_type),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_vals_retrieved,
                             query_length * embedding_vec_size * sizeof(float),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_missing_keys, query_length * sizeof(key_type),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_missing_vals,
                             query_length * embedding_vec_size * sizeof(float),
                             cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc((void**)&h_missing_index, query_length * sizeof(uint64_t),
                             cudaHostAllocPortable));

    // Allocate device side buffers
    CUDA_CHECK(cudaMalloc((void**)&d_query_keys, query_length * sizeof(key_type)));
    CUDA_CHECK(
        cudaMalloc((void**)&d_vals_retrieved, query_length * embedding_vec_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_keys, query_length * sizeof(key_type)));
    CUDA_CHECK(
        cudaMalloc((void**)&d_missing_vals, query_length * embedding_vec_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_index, query_length * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_missing_len, sizeof(size_t)));

    // Thread-private CUDA stream, all threads just use the #0 device
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Timimg variables
    double time_1;
    double time_2;

    /******************************************************************************
    *******************************************************************************
    ********************************Test start*************************************
    *******************************************************************************
    *******************************************************************************/

    // Normal-distribution random number generator
    size_t foot_print = emb_size - 1;  // Memory footprint for access the keys within the key buffer
    double mean = (double)(emb_size / 2);     // mean for normal distribution
    double dev = (double)(2 * query_length);  // dev for normal distribution
    // IntGenerator<size_t> uni_gen(0, foot_print);
    // Normal-distribution random number generator
    IntGenerator_normal<size_t> normal_gen(mean, dev);

    // Start normal distribution cache test
    printf("Worker %d : normal distribution test start.\n", thread_id);
    for (size_t i = 0; i < iter_round; i++) {
      // Generate random index with normal-distribution
      normal_gen.fill_unique(h_query_keys_index, query_length, 0, foot_print);
      // Select keys from total keys buffer with the index
      for (size_t j = 0; j < query_length; j++) {
        h_query_keys[j] = h_keys[h_query_keys_index[j]];
        // std::cout << h_query_keys[j] << " ";
      }
      std::cout << std::endl;

      // Copy the keys to GPU memory
      CUDA_CHECK(cudaMemcpyAsync(d_query_keys, h_query_keys, query_length * sizeof(key_type),
                                 cudaMemcpyHostToDevice, stream));
      // Wait for stream to complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Record time
      time_1 = W_time();
      // Get pairs from hashtable
      cache->Query(d_query_keys, query_length, d_vals_retrieved, d_missing_index, d_missing_keys,
                   d_missing_len, stream);
      // Wait for stream to complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Elapsed wall time
      time_2 = W_time() - time_1;
      printf("Worker %d : The Elapsed time for %zu round normal-distribution query is: %f sec.\n",
             thread_id, i, time_2);

      // Copy the data back to host
      CUDA_CHECK(cudaMemcpyAsync(h_vals_retrieved, d_vals_retrieved,
                                 query_length * embedding_vec_size * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_missing_index, d_missing_index, query_length * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_missing_keys, d_missing_keys, query_length * sizeof(key_type),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(&h_missing_len, d_missing_len, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      printf("Worker %d : %zu round : Missing key: %zu. Hit rate: %f %%.\n", thread_id, i,
             h_missing_len, 100.0f - (((float)h_missing_len / (float)query_length) * 100.0f));

      // Get missing key from embedding table
      // Insert missing values into the retrieved value buffer
      // Record time
      time_1 = W_time();
      for (size_t missing_idx = 0; missing_idx < h_missing_len; missing_idx++) {
        auto result = h_emb_table.find(h_missing_keys[missing_idx]);
        for (size_t emb_vec_idx = 0; emb_vec_idx < embedding_vec_size; emb_vec_idx++) {
          h_missing_vals[missing_idx * embedding_vec_size + emb_vec_idx] =
              (result->second)[emb_vec_idx];
          h_vals_retrieved[h_missing_index[missing_idx] * embedding_vec_size + emb_vec_idx] =
              (result->second)[emb_vec_idx];
        }
      }
      // Elapsed wall time
      time_2 = W_time() - time_1;
      printf(
          "Worker %d : The Elapsed time for %zu round normal-distribution fetching missing data "
          "is: %f sec.\n",
          thread_id, i, time_2);

      // Copy the missing value to device
      CUDA_CHECK(cudaMemcpyAsync(d_missing_vals, h_missing_vals,
                                 query_length * embedding_vec_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_vals_retrieved, h_vals_retrieved,
                                 query_length * embedding_vec_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // Record time
      time_1 = W_time();
      // Replace the missing <k,v> pairs into the cache
      if (cache_type == 0)
        cache->Replace(d_missing_keys, h_missing_len, d_missing_vals, stream);
      else
        cache->Replace(d_query_keys, query_length, d_vals_retrieved, stream);
      // Wait for stream to complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Elapsed wall time
      time_2 = W_time() - time_1;
      printf("Worker %d : The Elapsed time for %zu round normal-distribution replace is: %f sec.\n",
             thread_id, i, time_2);

      // Verification: Check for correctness for retrieved pairs
      check_result(h_query_keys, h_vals_retrieved, embedding_vec_size, query_length, increase);
      printf(
          "Worker %d : Result check for %zu round normal-distribution query+replace "
          "successfully!\n",
          thread_id, i);
    }

    printf("Worker %d : All Finished!\n", thread_id);

    // Clean-up
    cudaStreamDestroy(stream);
    free(h_query_keys_index);
    CUDA_CHECK(cudaFreeHost(h_query_keys));
    CUDA_CHECK(cudaFreeHost(h_vals_retrieved));
    CUDA_CHECK(cudaFreeHost(h_missing_keys));
    CUDA_CHECK(cudaFreeHost(h_missing_vals));
    CUDA_CHECK(cudaFreeHost(h_missing_index));

    CUDA_CHECK(cudaFree(d_query_keys));
    CUDA_CHECK(cudaFree(d_vals_retrieved));
    CUDA_CHECK(cudaFree(d_missing_keys));
    CUDA_CHECK(cudaFree(d_missing_vals));
    CUDA_CHECK(cudaFree(d_missing_index));
    CUDA_CHECK(cudaFree(d_missing_len));
  }

  // 1st test Clean-up
  free(h_keys);
  delete cache;

  // Start 2nd test
  std::cout << "****************************************" << std::endl;
  std::cout << "****************************************" << std::endl;
  std::cout << "Start Single-GPU Thread-safe Update and Dump API test " << std::endl;

  // The key and value buffer that contains all the keys and values to be inserted into the cache
  h_keys =
      (key_type*)malloc(SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type));
  h_vals = (float*)malloc(SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set *
                          embedding_vec_size * sizeof(float));
  float* h_new_vals = (float*)malloc(SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set *
                                     embedding_vec_size * sizeof(float));

  // The cache object to be tested
  if (cache_type == 0) {
    using Cache_ =
        gpu_cache::gpu_cache<key_type, ref_counter_type, empty_key, SET_ASSOCIATIVITY, SLAB_SIZE>;
    cache = new Cache_(cache_capacity_in_set, embedding_vec_size);
  } else {
    cache = new HugeCTR::EmbeddingCacheWrapper<key_type>(cache_capacity_in_set, embedding_vec_size);
  }

  // Key generator
  KeyGenerator<key_type> cache_key_gen;
  // Generating random unique keys
  cache_key_gen.fill_unique(h_keys, SLAB_SIZE * SET_ASSOCIATIVITY, cache_capacity_in_set,
                            empty_key);
  // Filling values vector according to the keys
  fill_vec(h_keys, h_vals, embedding_vec_size,
           SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set, increase);
  // Set new value
  increase = 1.0f;
  // Filling values vector according to the keys
  fill_vec(h_keys, h_new_vals, embedding_vec_size,
           SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set, increase);

  // Host-side buffers
  // Buffers holding keys and values to be inserted, each time insert 1 slab to every slabset
  key_type* h_insert_keys;
  float* h_insert_vals;
  // Buffers holding keys and values dumped and retrieved from the cache
  key_type* h_dump_keys;
  float* h_vals_retrieved;
  size_t h_dump_counter;
  size_t h_missing_len;
  key_type* h_acc_keys;

  // Device-side buffers
  key_type* d_keys;
  float* d_vals;
  // Buffers holding keys and values to be inserted, each time insert 1 slab to every slabset
  key_type* d_insert_keys;
  float* d_insert_vals;
  // Buffers holding keys and values dumped and retrieved from the cache
  key_type* d_dump_keys;
  float* d_vals_retrieved;
  size_t* d_dump_counter;
  uint64_t* d_missing_index;
  key_type* d_missing_keys;
  size_t* d_missing_len;

  CUDA_CHECK(cudaHostAlloc((void**)&h_insert_keys,
                           SLAB_SIZE * cache_capacity_in_set * sizeof(key_type),
                           cudaHostAllocPortable));
  CUDA_CHECK(cudaHostAlloc((void**)&h_insert_vals,
                           SLAB_SIZE * cache_capacity_in_set * embedding_vec_size * sizeof(float),
                           cudaHostAllocPortable));
  CUDA_CHECK(cudaHostAlloc((void**)&h_dump_keys,
                           SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type),
                           cudaHostAllocPortable));
  CUDA_CHECK(cudaHostAlloc(
      (void**)&h_vals_retrieved,
      SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * embedding_vec_size * sizeof(float),
      cudaHostAllocPortable));
  CUDA_CHECK(cudaHostAlloc((void**)&h_acc_keys,
                           SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type),
                           cudaHostAllocPortable));

  CUDA_CHECK(cudaMalloc((void**)&d_keys,
                        SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc((void**)&d_vals, SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set *
                                             embedding_vec_size * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc((void**)&d_insert_keys, SLAB_SIZE * cache_capacity_in_set * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc((void**)&d_insert_vals,
                        SLAB_SIZE * cache_capacity_in_set * embedding_vec_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_dump_keys,
                        SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc(
      (void**)&d_vals_retrieved,
      SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * embedding_vec_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_dump_counter, sizeof(size_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_missing_index,
                        SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc((void**)&d_missing_keys,
                        SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type)));
  CUDA_CHECK(cudaMalloc((void**)&d_missing_len, sizeof(size_t)));

  // CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Copy all keys and values from host to device
  CUDA_CHECK(cudaMemcpyAsync(
      d_keys, h_keys, SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * sizeof(key_type),
      cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(
      d_vals, h_new_vals,
      SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set * embedding_vec_size * sizeof(float),
      cudaMemcpyHostToDevice, stream));
  // Wait for stream to complete
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Each time insert 1 slab per slabset into the cache and check result
  for (size_t i = 0; i < SET_ASSOCIATIVITY; i++) {
    // Prepare the keys and values to be inserted
    for (size_t j = 0; j < cache_capacity_in_set; j++) {
      memcpy(h_insert_keys + j * SLAB_SIZE,
             h_keys + j * SLAB_SIZE * SET_ASSOCIATIVITY + i * SLAB_SIZE,
             SLAB_SIZE * sizeof(key_type));
      memcpy(h_insert_vals + j * SLAB_SIZE * embedding_vec_size,
             h_vals + (j * SLAB_SIZE * SET_ASSOCIATIVITY + i * SLAB_SIZE) * embedding_vec_size,
             SLAB_SIZE * embedding_vec_size * sizeof(float));
    }
    // Copy the selected keys to accumulate buffer
    memcpy(h_acc_keys + SLAB_SIZE * cache_capacity_in_set * i, h_insert_keys,
           SLAB_SIZE * cache_capacity_in_set * sizeof(key_type));

    // Copy the <k,v> pairs from host to device
    CUDA_CHECK(cudaMemcpyAsync(d_insert_keys, h_insert_keys,
                               SLAB_SIZE * cache_capacity_in_set * sizeof(key_type),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_insert_vals, h_insert_vals,
                        SLAB_SIZE * cache_capacity_in_set * embedding_vec_size * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
    // Insert the <k,v> pairs into the cache
    cache->Replace(d_insert_keys, SLAB_SIZE * cache_capacity_in_set, d_insert_vals, stream);
    // Wait for stream to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Record time
    time_a = W_time();
    // Update the new values to the cache(including missing keys)
    cache->Update(d_keys, SLAB_SIZE * SET_ASSOCIATIVITY * cache_capacity_in_set, d_vals, stream,
                  SLAB_SIZE);
    // Wait for stream to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Elapsed wall time
    time_b = W_time() - time_a;
    printf("The Elapsed time for %zu round update is: %f sec.\n", i, time_b);
    bool check_dump = false;
    if (check_dump) {
      // Record time
      time_a = W_time();
      // Dump the keys from the cache
      cache->Dump(d_dump_keys, d_dump_counter, 0, cache_capacity_in_set, stream);
      // Wait for stream to complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Elapsed wall time
      time_b = W_time() - time_a;
      printf("The Elapsed time for %zu round dump is: %f sec.\n", i, time_b);

      // Copy the dump counter from device to host
      CUDA_CHECK(cudaMemcpyAsync(&h_dump_counter, d_dump_counter, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream));
      // Wait for stream to complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Check the dump counter
      assert(h_dump_counter == SLAB_SIZE * cache_capacity_in_set * (i + 1));
      // Query all the dumped keys from the cache
      cache->Query(d_dump_keys, h_dump_counter, d_vals_retrieved, d_missing_index, d_missing_keys,
                   d_missing_len, stream);
      // Copy result from device to host
      CUDA_CHECK(cudaMemcpyAsync(h_dump_keys, d_dump_keys, h_dump_counter * sizeof(key_type),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_vals_retrieved, d_vals_retrieved,
                                 h_dump_counter * embedding_vec_size * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(&h_missing_len, d_missing_len, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream));
      // Wait for stream to complete
      CUDA_CHECK(cudaStreamSynchronize(stream));
      // Check result
      assert(h_missing_len == 0);
      compare_key(h_dump_keys, h_acc_keys, h_dump_counter);
      check_result(h_dump_keys, h_vals_retrieved, embedding_vec_size, h_dump_counter, increase);
    }
  }

  printf("Update and Dump API test all finished!\n");

  // 2nd test clean-up
  CUDA_CHECK(cudaStreamDestroy(stream));
  free(h_keys);
  free(h_vals);
  free(h_new_vals);

  CUDA_CHECK(cudaFreeHost(h_insert_keys));
  CUDA_CHECK(cudaFreeHost(h_insert_vals));
  CUDA_CHECK(cudaFreeHost(h_dump_keys));
  CUDA_CHECK(cudaFreeHost(h_vals_retrieved));
  CUDA_CHECK(cudaFreeHost(h_acc_keys));

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_vals));
  CUDA_CHECK(cudaFree(d_insert_keys));
  CUDA_CHECK(cudaFree(d_insert_vals));
  CUDA_CHECK(cudaFree(d_dump_keys));
  CUDA_CHECK(cudaFree(d_vals_retrieved));
  CUDA_CHECK(cudaFree(d_dump_counter));
  CUDA_CHECK(cudaFree(d_missing_index));
  CUDA_CHECK(cudaFree(d_missing_keys));
  CUDA_CHECK(cudaFree(d_missing_len));

  delete cache;

  return 0;
}
