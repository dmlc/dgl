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

#include <cooperative_groups.h>

#include <nv_gpu_cache.hpp>

namespace cg = cooperative_groups;

// Overload CUDA atomic for other 64bit unsigned/signed integer type
__forceinline__ __device__ long atomicAdd(long* address, long val) {
  return (long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ long long atomicAdd(long long* address, long long val) {
  return (long long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__forceinline__ __device__ unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return (unsigned long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

namespace gpu_cache {

#ifdef LIBCUDACXX_VERSION
template <int warp_size>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float, float* d_dst,
                                               const float* d_src) {
#pragma unroll
  for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
    d_dst[i] = d_src[i];
  }
}
#else
template <int warp_size>
__forceinline__ __device__ void warp_tile_copy(const size_t lane_idx,
                                               const size_t emb_vec_size_in_float,
                                               volatile float* d_dst, volatile float* d_src) {

#pragma unroll
  for (size_t i = lane_idx; i < emb_vec_size_in_float; i += warp_size) {
    d_dst[i] = d_src[i];
  }
}
#endif

#ifdef LIBCUDACXX_VERSION
// Will be called by multiple thread_block_tile((sub-)warp) on the same mutex
// Expect only one thread_block_tile return to execute critical section at any time
template <typename mutex, int warp_size>
__forceinline__ __device__ void warp_lock_mutex(const cg::thread_block_tile<warp_size>& warp_tile,
                                                mutex& set_mutex) {
  // The first thread of this (sub-)warp to acquire the lock
  if (warp_tile.thread_rank() == 0) {
    set_mutex.acquire();
  }
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
}

// The (sub-)warp holding the mutex will unlock the mutex after finishing the critical section on a
// set Expect any following (sub-)warp that acquire the mutex can see its modification done in the
// critical section
template <typename mutex, int warp_size>
__forceinline__ __device__ void warp_unlock_mutex(const cg::thread_block_tile<warp_size>& warp_tile,
                                                  mutex& set_mutex) {
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
  // The first thread of this (sub-)warp to release the lock
  if (warp_tile.thread_rank() == 0) {
    set_mutex.release();
  }
}
#else
// Will be called by multiple thread_block_tile((sub-)warp) on the same mutex
// Expect only one thread_block_tile return to execute critical section at any time
template <int warp_size>
__forceinline__ __device__ void warp_lock_mutex(const cg::thread_block_tile<warp_size>& warp_tile,
                                                volatile int& set_mutex) {
  // The first thread of this (sub-)warp to acquire the lock
  if (warp_tile.thread_rank() == 0) {
    while (0 == atomicCAS((int*)&set_mutex, 1, 0))
      ;
  }
  __threadfence();
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
}

// The (sub-)warp holding the mutex will unlock the mutex after finishing the critical section on a
// set Expect any following (sub-)warp that acquire the mutex can see its modification done in the
// critical section
template <int warp_size>
__forceinline__ __device__ void warp_unlock_mutex(const cg::thread_block_tile<warp_size>& warp_tile,
                                                  volatile int& set_mutex) {
  __threadfence();
  warp_tile.sync();  // Synchronize the threads in the (sub-)warp. Execution barrier + memory fence
  // The first thread of this (sub-)warp to release the lock
  if (warp_tile.thread_rank() == 0) {
    atomicExch((int*)&set_mutex, 1);
  }
}
#endif

// The (sub-)warp doing all reduction to find the slot with min slot_counter
// The slot with min slot_counter is the LR slot.
template <typename ref_counter_type, int warp_size>
__forceinline__ __device__ void warp_min_reduction(
    const cg::thread_block_tile<warp_size>& warp_tile, ref_counter_type& min_slot_counter_val,
    size_t& slab_distance, size_t& slot_distance) {
  const size_t lane_idx = warp_tile.thread_rank();
  slot_distance = lane_idx;

  for (size_t i = (warp_tile.size() >> 1); i > 0; i = i >> 1) {
    ref_counter_type input_slot_counter_val = warp_tile.shfl_xor(min_slot_counter_val, (int)i);
    size_t input_slab_distance = warp_tile.shfl_xor(slab_distance, (int)i);
    size_t input_slot_distance = warp_tile.shfl_xor(slot_distance, (int)i);

    if (input_slot_counter_val == min_slot_counter_val) {
      if (input_slab_distance == slab_distance) {
        if (input_slot_distance < slot_distance) {
          slot_distance = input_slot_distance;
        }
      } else if (input_slab_distance < slab_distance) {
        slab_distance = input_slab_distance;
        slot_distance = input_slot_distance;
      }
    } else if (input_slot_counter_val < min_slot_counter_val) {
      min_slot_counter_val = input_slot_counter_val;
      slab_distance = input_slab_distance;
      slot_distance = input_slot_distance;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef LIBCUDACXX_VERSION
// Kernel to initialize the GPU cache
// Init every entry of the cache with <unused_key, value> pair
template <typename slabset, typename ref_counter_type, typename atomic_ref_counter_type,
          typename key_type, typename mutex>
__global__ void init_cache(slabset* keys, ref_counter_type* slot_counter,
                           atomic_ref_counter_type* global_counter, const size_t num_slot,
                           const key_type empty_key, mutex* set_mutex,
                           const size_t capacity_in_set) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_slot) {
    // Set the key of this slot to unused key
    // Flatten the cache
    key_type* key_slot = (key_type*)keys;
    key_slot[idx] = empty_key;

    // Clear the counter for this slot
    slot_counter[idx] = 0;
  }
  // First CUDA thread clear the global counter
  if (idx == 0) {
    new (global_counter) atomic_ref_counter_type(0);
  }

  // First capacity_in_set CUDA thread initialize mutex
  if (idx < capacity_in_set) {
    new (set_mutex + idx) mutex(1);
  }
}

template <typename atomic_ref_counter_type, typename mutex>
__global__ void destruct_kernel(atomic_ref_counter_type* global_counter, mutex* set_mutex,
                                const size_t capacity_in_set) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // First CUDA thread destruct the global_counter
  if (idx == 0) {
    global_counter->~atomic_ref_counter_type();
  }
  // First capacity_in_set CUDA thread destruct the set mutex
  if (idx < capacity_in_set) {
    (set_mutex + idx)->~mutex();
  }
}
#else
// Kernel to initialize the GPU cache
// Init every entry of the cache with <unused_key, value> pair
template <typename slabset, typename ref_counter_type, typename key_type>
__global__ void init_cache(slabset* keys, ref_counter_type* slot_counter,
                           ref_counter_type* global_counter, const size_t num_slot,
                           const key_type empty_key, int* set_mutex, const size_t capacity_in_set) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_slot) {
    // Set the key of this slot to unused key
    // Flatten the cache
    key_type* key_slot = (key_type*)keys;
    key_slot[idx] = empty_key;

    // Clear the counter for this slot
    slot_counter[idx] = 0;
  }
  // First CUDA thread clear the global counter
  if (idx == 0) {
    global_counter[idx] = 0;
  }

  // First capacity_in_set CUDA thread initialize mutex
  if (idx < capacity_in_set) {
    set_mutex[idx] = 1;
  }
}
#endif

// Kernel to update global counter
// Resolve distance overflow issue as well
#ifdef LIBCUDACXX_VERSION
template <typename atomic_ref_counter_type>
__global__ void update_kernel_overflow_ignore(atomic_ref_counter_type* global_counter,
                                              size_t* d_missing_len) {
  // Update global counter
  global_counter->fetch_add(1, cuda::std::memory_order_relaxed);
  *d_missing_len = 0;
}
#else
template <typename ref_counter_type>
__global__ void update_kernel_overflow_ignore(ref_counter_type* global_counter,
                                              size_t* d_missing_len) {
  // Update global counter
  atomicAdd(global_counter, 1);
  *d_missing_len = 0;
}
#endif

#ifdef LIBCUDACXX_VERSION
// Kernel to read from cache
// Also update locality information for touched slot
template <typename key_type, typename ref_counter_type, typename atomic_ref_counter_type,
          typename slabset, typename set_hasher, typename slab_hasher, typename mutex,
          key_type empty_key, int set_associativity, int warp_size>
__global__ void get_kernel(const key_type* d_keys, const size_t len, float* d_values,
                           const size_t embedding_vec_size, uint64_t* d_missing_index,
                           key_type* d_missing_keys, size_t* d_missing_len,
                           const atomic_ref_counter_type* global_counter,
                           ref_counter_type* slot_counter, const size_t capacity_in_set,
                           const slabset* keys, const float* vals, mutex* set_mutex,
                           const size_t task_per_warp_tile) {
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const size_t warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const size_t key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  key_type key;
  // The dst slabset and the dst slab inside this set
  size_t src_set;
  size_t src_slab;
  // The variable that contains the missing key
  key_type missing_key;
  // The variable that contains the index for the missing key
  uint64_t missing_index;
  // The counter for counting the missing key in this warp
  uint8_t warp_missing_counter = 0;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = set_hasher::hash(key) % capacity_in_set;
      src_slab = slab_hasher::hash(key) % set_associativity;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task and the global index to all lane in the warp_tile
    key_type next_key = warp_tile.shfl(key, next_lane);
    size_t next_idx = warp_tile.shfl(key_idx, next_lane);
    size_t next_set = warp_tile.shfl(src_set, next_lane);
    size_t next_slab = warp_tile.shfl(src_slab, next_lane);

    // Counter to record how many slab have been searched
    size_t counter = 0;

    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex<mutex, warp_size>(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched, mark missing task, task is
      // completed
      if (counter >= set_associativity) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      key_type read_key = keys[next_set].set_[next_slab].slab_[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found, mark hit task, copy the founded data, the task is completed
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;
        if (lane_idx == (size_t)next_lane) {
          slot_counter[found_offset] = global_counter->load(cuda::std::memory_order_relaxed);
          active = false;
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  d_values + next_idx * embedding_vec_size,
                                  vals + found_offset * embedding_vec_size);

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key, if found empty key, mark missing task, task is
      // completed
      if (warp_tile.ballot(read_key == empty_key) != 0) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // Not found in this slab, the task is not completed, goto searching next slab
      counter++;
      next_slab = (next_slab + 1) % set_associativity;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex<mutex, warp_size>(warp_tile, set_mutex[next_set]);
  }

  // After warp_tile complete the working queue, save the result for output
  // First thread of the warp_tile accumulate the missing length to global variable
  size_t warp_position;
  if (lane_idx == 0) {
    warp_position = atomicAdd(d_missing_len, (size_t)warp_missing_counter);
  }
  warp_position = warp_tile.shfl(warp_position, 0);

  if (lane_idx < warp_missing_counter) {
    d_missing_keys[warp_position + lane_idx] = missing_key;
    d_missing_index[warp_position + lane_idx] = missing_index;
  }
}
#else
// Kernel to read from cache
// Also update locality information for touched slot
template <typename key_type, typename ref_counter_type, typename slabset, typename set_hasher,
          typename slab_hasher, key_type empty_key, int set_associativity, int warp_size>
__global__ void get_kernel(const key_type* d_keys, const size_t len, float* d_values,
                           const size_t embedding_vec_size, uint64_t* d_missing_index,
                           key_type* d_missing_keys, size_t* d_missing_len,
                           ref_counter_type* global_counter,
                           volatile ref_counter_type* slot_counter, const size_t capacity_in_set,
                           volatile slabset* keys, volatile float* vals, volatile int* set_mutex,
                           const size_t task_per_warp_tile) {
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const size_t warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const size_t key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  key_type key;
  // The dst slabset and the dst slab inside this set
  size_t src_set;
  size_t src_slab;
  // The variable that contains the missing key
  key_type missing_key;
  // The variable that contains the index for the missing key
  uint64_t missing_index;
  // The counter for counting the missing key in this warp
  uint8_t warp_missing_counter = 0;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = set_hasher::hash(key) % capacity_in_set;
      src_slab = slab_hasher::hash(key) % set_associativity;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task and the global index to all lane in the warp_tile
    key_type next_key = warp_tile.shfl(key, next_lane);
    size_t next_idx = warp_tile.shfl(key_idx, next_lane);
    size_t next_set = warp_tile.shfl(src_set, next_lane);
    size_t next_slab = warp_tile.shfl(src_slab, next_lane);

    // Counter to record how many slab have been searched
    size_t counter = 0;

    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex<warp_size>(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched, mark missing task, task is
      // completed
      if (counter >= set_associativity) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      key_type read_key = ((volatile key_type*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found, mark hit task, copy the founded data, the task is completed
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;
        if (lane_idx == (size_t)next_lane) {
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
          active = false;
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  (volatile float*)(d_values + next_idx * embedding_vec_size),
                                  (volatile float*)(vals + found_offset * embedding_vec_size));

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key, if found empty key, mark missing task, task is
      // completed
      if (warp_tile.ballot(read_key == empty_key) != 0) {
        if (lane_idx == warp_missing_counter) {
          missing_key = next_key;
          missing_index = next_idx;
        }

        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        warp_missing_counter++;
        active_mask = warp_tile.ballot(active);
        break;
      }

      // Not found in this slab, the task is not completed, goto searching next slab
      counter++;
      next_slab = (next_slab + 1) % set_associativity;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex<warp_size>(warp_tile, set_mutex[next_set]);
  }

  // After warp_tile complete the working queue, save the result for output
  // First thread of the warp_tile accumulate the missing length to global variable
  size_t warp_position;
  if (lane_idx == 0) {
    warp_position = atomicAdd(d_missing_len, (size_t)warp_missing_counter);
  }
  warp_position = warp_tile.shfl(warp_position, 0);

  if (lane_idx < warp_missing_counter) {
    d_missing_keys[warp_position + lane_idx] = missing_key;
    d_missing_index[warp_position + lane_idx] = missing_index;
  }
}
#endif

#ifdef LIBCUDACXX_VERSION
// Kernel to insert or replace the <k,v> pairs into the cache
template <typename key_type, typename slabset, typename ref_counter_type, typename mutex,
          typename atomic_ref_counter_type, typename set_hasher, typename slab_hasher,
          key_type empty_key, int set_associativity, int warp_size,
          ref_counter_type max_ref_counter_type = std::numeric_limits<ref_counter_type>::max(),
          size_t max_slab_distance = std::numeric_limits<size_t>::max()>
__global__ void insert_replace_kernel(const key_type* d_keys, const float* d_values,
                                      const size_t embedding_vec_size, const size_t len,
                                      slabset* keys, float* vals, ref_counter_type* slot_counter,
                                      mutex* set_mutex,
                                      const atomic_ref_counter_type* global_counter,
                                      const size_t capacity_in_set,
                                      const size_t task_per_warp_tile) {
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const size_t warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const size_t key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  key_type key;
  // The dst slabset and the dst slab inside this set
  size_t src_set;
  size_t src_slab;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = set_hasher::hash(key) % capacity_in_set;
      src_slab = slab_hasher::hash(key) % set_associativity;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task, the global index and the src slabset and slab to all lane in a warp_tile
    key_type next_key = warp_tile.shfl(key, next_lane);
    size_t next_idx = warp_tile.shfl(key_idx, next_lane);
    size_t next_set = warp_tile.shfl(src_set, next_lane);
    size_t next_slab = warp_tile.shfl(src_slab, next_lane);
    size_t first_slab = next_slab;

    // Counter to record how many slab have been searched
    size_t counter = 0;

    // Variable to keep the min slot counter during the probing
    ref_counter_type min_slot_counter_val = max_ref_counter_type;
    // Variable to keep the slab distance for slot with min counter
    size_t slab_distance = max_slab_distance;
    // Variable to keep the slot distance for slot with min counter within the slab
    size_t slot_distance;
    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex<mutex, warp_size>(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched
      // and no empty slots or target slots are found. Replace with LRU
      if (counter >= set_associativity) {
        // (sub)Warp all-reduction, the reduction result store in all threads
        warp_min_reduction<ref_counter_type, warp_size>(warp_tile, min_slot_counter_val,
                                                        slab_distance, slot_distance);

        // Calculate the position of LR slot
        size_t target_slab = (first_slab + slab_distance) % set_associativity;
        size_t slot_index =
            (next_set * set_associativity + target_slab) * warp_size + slot_distance;

        // Replace the LR slot
        if (lane_idx == (size_t)next_lane) {
          keys[next_set].set_[target_slab].slab_[slot_distance] = key;
          slot_counter[slot_index] = global_counter->load(cuda::std::memory_order_relaxed);
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  vals + slot_index * embedding_vec_size,
                                  d_values + next_idx * embedding_vec_size);

        // Replace complete, mark this task completed
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      key_type read_key = keys[next_set].set_[next_slab].slab_[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found target key, the insertion/replace is no longer needed.
      // Refresh the slot, the task is completed
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;
        if (lane_idx == (size_t)next_lane) {
          slot_counter[found_offset] = global_counter->load(cuda::std::memory_order_relaxed);
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key.
      // If found empty key, do insertion,the task is complete
      found_lane = __ffs(warp_tile.ballot(read_key == empty_key)) - 1;
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;

        if (lane_idx == (size_t)next_lane) {
          keys[next_set].set_[next_slab].slab_[found_lane] = key;
          slot_counter[found_offset] = global_counter->load(cuda::std::memory_order_relaxed);
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  vals + found_offset * embedding_vec_size,
                                  d_values + next_idx * embedding_vec_size);

        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // If no target or unused slot found in this slab,
      // Refresh LR info, continue probing
      ref_counter_type read_slot_counter =
          slot_counter[(next_set * set_associativity + next_slab) * warp_size + lane_idx];
      if (read_slot_counter < min_slot_counter_val) {
        min_slot_counter_val = read_slot_counter;
        slab_distance = counter;
      }

      counter++;
      next_slab = (next_slab + 1) % set_associativity;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex<mutex, warp_size>(warp_tile, set_mutex[next_set]);
  }
}
#else
// Kernel to insert or replace the <k,v> pairs into the cache
template <typename key_type, typename slabset, typename ref_counter_type, typename set_hasher,
          typename slab_hasher, key_type empty_key, int set_associativity, int warp_size,
          ref_counter_type max_ref_counter_type = std::numeric_limits<ref_counter_type>::max(),
          size_t max_slab_distance = std::numeric_limits<size_t>::max()>
__global__ void insert_replace_kernel(const key_type* d_keys, const float* d_values,
                                      const size_t embedding_vec_size, const size_t len,
                                      volatile slabset* keys, volatile float* vals,
                                      volatile ref_counter_type* slot_counter,
                                      volatile int* set_mutex, ref_counter_type* global_counter,
                                      const size_t capacity_in_set,
                                      const size_t task_per_warp_tile) {
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const size_t warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const size_t key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  key_type key;
  // The dst slabset and the dst slab inside this set
  size_t src_set;
  size_t src_slab;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = set_hasher::hash(key) % capacity_in_set;
      src_slab = slab_hasher::hash(key) % set_associativity;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task, the global index and the src slabset and slab to all lane in a warp_tile
    key_type next_key = warp_tile.shfl(key, next_lane);
    size_t next_idx = warp_tile.shfl(key_idx, next_lane);
    size_t next_set = warp_tile.shfl(src_set, next_lane);
    size_t next_slab = warp_tile.shfl(src_slab, next_lane);
    size_t first_slab = next_slab;

    // Counter to record how many slab have been searched
    size_t counter = 0;

    // Variable to keep the min slot counter during the probing
    ref_counter_type min_slot_counter_val = max_ref_counter_type;
    // Variable to keep the slab distance for slot with min counter
    size_t slab_distance = max_slab_distance;
    // Variable to keep the slot distance for slot with min counter within the slab
    size_t slot_distance;
    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex<warp_size>(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched
      // and no empty slots or target slots are found. Replace with LRU
      if (counter >= set_associativity) {
        // (sub)Warp all-reduction, the reduction result store in all threads
        warp_min_reduction<ref_counter_type, warp_size>(warp_tile, min_slot_counter_val,
                                                        slab_distance, slot_distance);

        // Calculate the position of LR slot
        size_t target_slab = (first_slab + slab_distance) % set_associativity;
        size_t slot_index =
            (next_set * set_associativity + target_slab) * warp_size + slot_distance;

        // Replace the LR slot
        if (lane_idx == (size_t)next_lane) {
          ((volatile key_type*)(keys[next_set].set_[target_slab].slab_))[slot_distance] = key;
          slot_counter[slot_index] = atomicAdd(global_counter, 0);
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  (volatile float*)(vals + slot_index * embedding_vec_size),
                                  (volatile float*)(d_values + next_idx * embedding_vec_size));

        // Replace complete, mark this task completed
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      key_type read_key = ((volatile key_type*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found target key, the insertion/replace is no longer needed.
      // Refresh the slot, the task is completed
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;
        if (lane_idx == (size_t)next_lane) {
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key.
      // If found empty key, do insertion,the task is complete
      found_lane = __ffs(warp_tile.ballot(read_key == empty_key)) - 1;
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;

        if (lane_idx == (size_t)next_lane) {
          ((volatile key_type*)(keys[next_set].set_[next_slab].slab_))[found_lane] = key;
          slot_counter[found_offset] = atomicAdd(global_counter, 0);
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  (volatile float*)(vals + found_offset * embedding_vec_size),
                                  (volatile float*)(d_values + next_idx * embedding_vec_size));

        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // If no target or unused slot found in this slab,
      // Refresh LR info, continue probing
      ref_counter_type read_slot_counter =
          slot_counter[(next_set * set_associativity + next_slab) * warp_size + lane_idx];
      if (read_slot_counter < min_slot_counter_val) {
        min_slot_counter_val = read_slot_counter;
        slab_distance = counter;
      }

      counter++;
      next_slab = (next_slab + 1) % set_associativity;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex<warp_size>(warp_tile, set_mutex[next_set]);
  }
}
#endif

#ifdef LIBCUDACXX_VERSION
// Kernel to update the existing keys in the cache
// Will not change the locality information
template <typename key_type, typename slabset, typename set_hasher, typename slab_hasher,
          typename mutex, key_type empty_key, int set_associativity, int warp_size>
__global__ void update_kernel(const key_type* d_keys, const size_t len, const float* d_values,
                              const size_t embedding_vec_size, const size_t capacity_in_set,
                              const slabset* keys, float* vals, mutex* set_mutex,
                              const size_t task_per_warp_tile) {
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const size_t warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const size_t key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  key_type key;
  // The dst slabset and the dst slab inside this set
  size_t src_set;
  size_t src_slab;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = set_hasher::hash(key) % capacity_in_set;
      src_slab = slab_hasher::hash(key) % set_associativity;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task and the global index to all lane in the warp_tile
    key_type next_key = warp_tile.shfl(key, next_lane);
    size_t next_idx = warp_tile.shfl(key_idx, next_lane);
    size_t next_set = warp_tile.shfl(src_set, next_lane);
    size_t next_slab = warp_tile.shfl(src_slab, next_lane);

    // Counter to record how many slab have been searched
    size_t counter = 0;

    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex<mutex, warp_size>(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched, mark missing task, do nothing, task
      // complete
      if (counter >= set_associativity) {
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      key_type read_key = keys[next_set].set_[next_slab].slab_[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found, mark hit task, update the value, the task is completed
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  vals + found_offset * embedding_vec_size,
                                  d_values + next_idx * embedding_vec_size);

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key, if found empty key, mark missing task, do nothing,
      // task is completed
      if (warp_tile.ballot(read_key == empty_key) != 0) {
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Not found in this slab, the task is not completed, goto searching next slab
      counter++;
      next_slab = (next_slab + 1) % set_associativity;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex<mutex, warp_size>(warp_tile, set_mutex[next_set]);
  }
}
#else
// Kernel to update the existing keys in the cache
// Will not change the locality information
template <typename key_type, typename slabset, typename set_hasher, typename slab_hasher,
          key_type empty_key, int set_associativity, int warp_size>
__global__ void update_kernel(const key_type* d_keys, const size_t len, const float* d_values,
                              const size_t embedding_vec_size, const size_t capacity_in_set,
                              volatile slabset* keys, volatile float* vals, volatile int* set_mutex,
                              const size_t task_per_warp_tile) {
  // Lane(thread) ID within a warp_tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile global ID
  const size_t warp_tile_global_idx =
      (blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank();
  // The index of key for this thread
  const size_t key_idx = (warp_tile_global_idx * task_per_warp_tile) + lane_idx;
  // The assigned key for this lane(thread)
  key_type key;
  // The dst slabset and the dst slab inside this set
  size_t src_set;
  size_t src_slab;
  // Active flag: whether current lane(thread) has unfinished task
  bool active = false;
  if (lane_idx < task_per_warp_tile) {
    if (key_idx < len) {
      active = true;
      key = d_keys[key_idx];
      src_set = set_hasher::hash(key) % capacity_in_set;
      src_slab = slab_hasher::hash(key) % set_associativity;
    }
  }

  // Lane participate in warp_tile ballot to produce warp-level work queue
  unsigned active_mask = warp_tile.ballot(active);

  // The warp-level outer loop: finish all the tasks within the work queue
  while (active_mask != 0) {
    // Next task in the work quere, start from lower index lane(thread)
    int next_lane = __ffs(active_mask) - 1;
    // Broadcast the task and the global index to all lane in the warp_tile
    key_type next_key = warp_tile.shfl(key, next_lane);
    size_t next_idx = warp_tile.shfl(key_idx, next_lane);
    size_t next_set = warp_tile.shfl(src_set, next_lane);
    size_t next_slab = warp_tile.shfl(src_slab, next_lane);

    // Counter to record how many slab have been searched
    size_t counter = 0;

    // Working queue before task started
    const unsigned old_active_mask = active_mask;

    // Lock the slabset before operating the slabset
    warp_lock_mutex<warp_size>(warp_tile, set_mutex[next_set]);

    // The warp-level inner loop: finish a single task in the work queue
    while (active_mask == old_active_mask) {
      // When all the slabs inside a slabset have been searched, mark missing task, do nothing, task
      // complete
      if (counter >= set_associativity) {
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // The warp_tile read out the slab
      key_type read_key = ((volatile key_type*)(keys[next_set].set_[next_slab].slab_))[lane_idx];

      // Compare the slab data with the target key
      int found_lane = __ffs(warp_tile.ballot(read_key == next_key)) - 1;

      // If found, mark hit task, update the value, the task is completed
      if (found_lane >= 0) {
        size_t found_offset = (next_set * set_associativity + next_slab) * warp_size + found_lane;
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        warp_tile_copy<warp_size>(lane_idx, embedding_vec_size,
                                  (volatile float*)(vals + found_offset * embedding_vec_size),
                                  (volatile float*)(d_values + next_idx * embedding_vec_size));

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Compare the slab data with empty key, if found empty key, mark missing task, do nothing,
      // task is completed
      if (warp_tile.ballot(read_key == empty_key) != 0) {
        if (lane_idx == (size_t)next_lane) {
          active = false;
        }

        active_mask = warp_tile.ballot(active);
        break;
      }

      // Not found in this slab, the task is not completed, goto searching next slab
      counter++;
      next_slab = (next_slab + 1) % set_associativity;
    }

    // Unlock the slabset after operating the slabset
    warp_unlock_mutex<warp_size>(warp_tile, set_mutex[next_set]);
  }
}
#endif

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename slabset, typename mutex, key_type empty_key,
          int set_associativity, int warp_size>
__global__ void dump_kernel(key_type* d_keys, size_t* d_dump_counter, const slabset* keys,
                            mutex* set_mutex, const size_t start_set_index,
                            const size_t end_set_index) {
  // Block-level counter used by all warp tiles within a block
  __shared__ uint32_t block_acc;
  // Initialize block-level counter
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();
  // Lane(thread) ID within a warp tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile target slabset id
  const size_t set_idx =
      ((blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank()) + start_set_index;
  // Keys dump from cache
  key_type read_key[set_associativity];
  // Lane(thread) offset for storing each key
  uint32_t thread_key_offset[set_associativity];
  // Warp offset for storing each key
  uint32_t warp_key_offset;
  // Block offset for storing each key
  __shared__ size_t block_key_offset;

  // Warp tile dump target slabset
  if (set_idx < end_set_index) {
    // Lock the slabset before operating the slabset
    warp_lock_mutex<mutex, warp_size>(warp_tile, set_mutex[set_idx]);

    // The warp tile read out the slabset
    for (unsigned slab_id = 0; slab_id < set_associativity; slab_id++) {
      // The warp tile read out a slab
      read_key[slab_id] = keys[set_idx].set_[slab_id].slab_[lane_idx];
    }

    // Finish dumping the slabset, unlock the slabset
    warp_unlock_mutex<mutex, warp_size>(warp_tile, set_mutex[set_idx]);

    // Each lane(thread) within the warp tile calculate the offset to store its keys
    uint32_t warp_tile_total_keys = 0;
    for (unsigned slab_id = 0; slab_id < set_associativity; slab_id++) {
      unsigned valid_mask = warp_tile.ballot(read_key[slab_id] != empty_key);
      thread_key_offset[slab_id] =
          __popc(valid_mask & ((1U << lane_idx) - 1U)) + warp_tile_total_keys;
      warp_tile_total_keys = warp_tile_total_keys + __popc(valid_mask);
    }

    // Each warp tile request a unique place from the block-level counter
    if (lane_idx == 0) {
      warp_key_offset = atomicAdd(&block_acc, warp_tile_total_keys);
    }
    warp_key_offset = warp_tile.shfl(warp_key_offset, 0);
  }

  // Each block request a unique place in global memory output buffer
  __syncthreads();
  if (threadIdx.x == 0) {
    block_key_offset = atomicAdd(d_dump_counter, (size_t)block_acc);
  }
  __syncthreads();

  // Warp tile store the (non-empty)keys back to output buffer
  if (set_idx < end_set_index) {
    for (unsigned slab_id = 0; slab_id < set_associativity; slab_id++) {
      if (read_key[slab_id] != empty_key) {
        d_keys[block_key_offset + warp_key_offset + thread_key_offset[slab_id]] = read_key[slab_id];
      }
    }
  }
}
#else
template <typename key_type, typename slabset, key_type empty_key, int set_associativity,
          int warp_size>
__global__ void dump_kernel(key_type* d_keys, size_t* d_dump_counter, volatile slabset* keys,
                            volatile int* set_mutex, const size_t start_set_index,
                            const size_t end_set_index) {
  // Block-level counter used by all warp tiles within a block
  __shared__ uint32_t block_acc;
  // Initialize block-level counter
  if (threadIdx.x == 0) {
    block_acc = 0;
  }
  __syncthreads();
  // Lane(thread) ID within a warp tile
  cg::thread_block_tile<warp_size> warp_tile =
      cg::tiled_partition<warp_size>(cg::this_thread_block());
  const size_t lane_idx = warp_tile.thread_rank();
  // Warp tile target slabset id
  const size_t set_idx =
      ((blockIdx.x * (blockDim.x / warp_size)) + warp_tile.meta_group_rank()) + start_set_index;
  // Keys dump from cache
  key_type read_key[set_associativity];
  // Lane(thread) offset for storing each key
  uint32_t thread_key_offset[set_associativity];
  // Warp offset for storing each key
  uint32_t warp_key_offset;
  // Block offset for storing each key
  __shared__ size_t block_key_offset;

  // Warp tile dump target slabset
  if (set_idx < end_set_index) {
    // Lock the slabset before operating the slabset
    warp_lock_mutex<warp_size>(warp_tile, set_mutex[set_idx]);

    // The warp tile read out the slabset
    for (unsigned slab_id = 0; slab_id < set_associativity; slab_id++) {
      // The warp tile read out a slab
      read_key[slab_id] = ((volatile key_type*)(keys[set_idx].set_[slab_id].slab_))[lane_idx];
    }

    // Finish dumping the slabset, unlock the slabset
    warp_unlock_mutex<warp_size>(warp_tile, set_mutex[set_idx]);

    // Each lane(thread) within the warp tile calculate the offset to store its keys
    uint32_t warp_tile_total_keys = 0;
    for (unsigned slab_id = 0; slab_id < set_associativity; slab_id++) {
      unsigned valid_mask = warp_tile.ballot(read_key[slab_id] != empty_key);
      thread_key_offset[slab_id] =
          __popc(valid_mask & ((1U << lane_idx) - 1U)) + warp_tile_total_keys;
      warp_tile_total_keys = warp_tile_total_keys + __popc(valid_mask);
    }

    // Each warp tile request a unique place from the block-level counter
    if (lane_idx == 0) {
      warp_key_offset = atomicAdd(&block_acc, warp_tile_total_keys);
    }
    warp_key_offset = warp_tile.shfl(warp_key_offset, 0);
  }

  // Each block request a unique place in global memory output buffer
  __syncthreads();
  if (threadIdx.x == 0) {
    block_key_offset = atomicAdd(d_dump_counter, (size_t)block_acc);
  }
  __syncthreads();

  // Warp tile store the (non-empty)keys back to output buffer
  if (set_idx < end_set_index) {
    for (unsigned slab_id = 0; slab_id < set_associativity; slab_id++) {
      if (read_key[slab_id] != empty_key) {
        d_keys[block_key_offset + warp_key_offset + thread_key_offset[slab_id]] = read_key[slab_id];
      }
    }
  }
}
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
          slab_hasher>::gpu_cache(const size_t capacity_in_set, const size_t embedding_vec_size)
    : capacity_in_set_(capacity_in_set), embedding_vec_size_(embedding_vec_size) {
  // Check parameter
  if (capacity_in_set_ == 0) {
    printf("Error: Invalid value for capacity_in_set.\n");
    return;
  }
  if (embedding_vec_size_ == 0) {
    printf("Error: Invalid value for embedding_vec_size.\n");
    return;
  }
  if (set_associativity <= 0) {
    printf("Error: Invalid value for set_associativity.\n");
    return;
  }
  if (warp_size != 1 && warp_size != 2 && warp_size != 4 && warp_size != 8 && warp_size != 16 &&
      warp_size != 32) {
    printf("Error: Invalid value for warp_size.\n");
    return;
  }

  // Get the current CUDA dev
  CUDA_CHECK(cudaGetDevice(&dev_));

  // Calculate # of slot
  num_slot_ = capacity_in_set_ * set_associativity * warp_size;

  // Allocate GPU memory for cache
  CUDA_CHECK(cudaMalloc((void**)&keys_, sizeof(slabset) * capacity_in_set_));
  CUDA_CHECK(cudaMalloc((void**)&vals_, sizeof(float) * embedding_vec_size_ * num_slot_));
  CUDA_CHECK(cudaMalloc((void**)&slot_counter_, sizeof(ref_counter_type) * num_slot_));
  CUDA_CHECK(cudaMalloc((void**)&global_counter_, sizeof(atomic_ref_counter_type)));

  // Allocate GPU memory for set mutex
  CUDA_CHECK(cudaMalloc((void**)&set_mutex_, sizeof(mutex) * capacity_in_set_));

  // Initialize the cache, set all entry to unused <K,V>
  init_cache<<<((num_slot_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
      keys_, slot_counter_, global_counter_, num_slot_, empty_key, set_mutex_, capacity_in_set_);

  // Wait for initialization to finish
  CUDA_CHECK(cudaStreamSynchronize(0));
  CUDA_CHECK(cudaGetLastError());
}
#else
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
          slab_hasher>::gpu_cache(const size_t capacity_in_set, const size_t embedding_vec_size)
    : capacity_in_set_(capacity_in_set), embedding_vec_size_(embedding_vec_size) {
  // Check parameter
  if (capacity_in_set_ == 0) {
    printf("Error: Invalid value for capacity_in_set.\n");
    return;
  }
  if (embedding_vec_size_ == 0) {
    printf("Error: Invalid value for embedding_vec_size.\n");
    return;
  }
  if (set_associativity <= 0) {
    printf("Error: Invalid value for set_associativity.\n");
    return;
  }
  if (warp_size != 1 && warp_size != 2 && warp_size != 4 && warp_size != 8 && warp_size != 16 &&
      warp_size != 32) {
    printf("Error: Invalid value for warp_size.\n");
    return;
  }

  // Get the current CUDA dev
  CUDA_CHECK(cudaGetDevice(&dev_));

  // Calculate # of slot
  num_slot_ = capacity_in_set_ * set_associativity * warp_size;

  // Allocate GPU memory for cache
  CUDA_CHECK(cudaMalloc((void**)&keys_, sizeof(slabset) * capacity_in_set_));
  CUDA_CHECK(cudaMalloc((void**)&vals_, sizeof(float) * embedding_vec_size_ * num_slot_));
  CUDA_CHECK(cudaMalloc((void**)&slot_counter_, sizeof(ref_counter_type) * num_slot_));
  CUDA_CHECK(cudaMalloc((void**)&global_counter_, sizeof(ref_counter_type)));

  // Allocate GPU memory for set mutex
  CUDA_CHECK(cudaMalloc((void**)&set_mutex_, sizeof(int) * capacity_in_set_));

  // Initialize the cache, set all entry to unused <K,V>
  init_cache<<<((num_slot_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
      keys_, slot_counter_, global_counter_, num_slot_, empty_key, set_mutex_, capacity_in_set_);

  // Wait for initialization to finish
  CUDA_CHECK(cudaStreamSynchronize(0));
  CUDA_CHECK(cudaGetLastError());
}
#endif

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
          slab_hasher>::~gpu_cache() {
  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;

  // Check device
  dev_restorer.check_device(dev_);

  // Destruct CUDA std object
  destruct_kernel<<<((capacity_in_set_ - 1) / BLOCK_SIZE_) + 1, BLOCK_SIZE_>>>(
      global_counter_, set_mutex_, capacity_in_set_);
  // Wait for destruction to finish
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Free GPU memory for cache
  CUDA_CHECK(cudaFree(keys_));
  CUDA_CHECK(cudaFree(vals_));
  CUDA_CHECK(cudaFree(slot_counter_));
  CUDA_CHECK(cudaFree(global_counter_));
  // Free GPU memory for set mutex
  CUDA_CHECK(cudaFree(set_mutex_));
}
#else
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
          slab_hasher>::~gpu_cache() noexcept(false) {
  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;

  // Check device
  dev_restorer.check_device(dev_);

  // Free GPU memory for cache
  CUDA_CHECK(cudaFree(keys_));
  CUDA_CHECK(cudaFree(vals_));
  CUDA_CHECK(cudaFree(slot_counter_));
  CUDA_CHECK(cudaFree(global_counter_));
  // Free GPU memory for set mutex
  CUDA_CHECK(cudaFree(set_mutex_));
}
#endif

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Query(const key_type* d_keys, const size_t len, float* d_values,
                                   uint64_t* d_missing_index, key_type* d_missing_keys,
                                   size_t* d_missing_len, cudaStream_t stream,
                                   const size_t task_per_warp_tile) {
  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Check if it is a valid query
  if (len == 0) {
    // Set the d_missing_len to 0 before return
    CUDA_CHECK(cudaMemsetAsync(d_missing_len, 0, sizeof(size_t), stream));
    return;
  }

  // Update the global counter as user perform a new(most recent) read operation to the cache
  // Resolve distance overflow issue as well.
  update_kernel_overflow_ignore<atomic_ref_counter_type>
      <<<1, 1, 0, stream>>>(global_counter_, d_missing_len);

  // Read from the cache
  // Touch and refresh the hitting slot
  const size_t keys_per_block = (BLOCK_SIZE_ / warp_size) * task_per_warp_tile;
  const size_t grid_size = ((len - 1) / keys_per_block) + 1;
  get_kernel<key_type, ref_counter_type, atomic_ref_counter_type, slabset, set_hasher, slab_hasher,
             mutex, empty_key, set_associativity, warp_size><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      d_keys, len, d_values, embedding_vec_size_, d_missing_index, d_missing_keys, d_missing_len,
      global_counter_, slot_counter_, capacity_in_set_, keys_, vals_, set_mutex_,
      task_per_warp_tile);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#else
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Query(const key_type* d_keys, const size_t len, float* d_values,
                                   uint64_t* d_missing_index, key_type* d_missing_keys,
                                   size_t* d_missing_len, cudaStream_t stream,
                                   const size_t task_per_warp_tile) {
  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Check if it is a valid query
  if (len == 0) {
    // Set the d_missing_len to 0 before return
    CUDA_CHECK(cudaMemsetAsync(d_missing_len, 0, sizeof(size_t), stream));
    return;
  }

  // Update the global counter as user perform a new(most recent) read operation to the cache
  // Resolve distance overflow issue as well.
  update_kernel_overflow_ignore<ref_counter_type>
      <<<1, 1, 0, stream>>>(global_counter_, d_missing_len);

  // Read from the cache
  // Touch and refresh the hitting slot
  const size_t keys_per_block = (BLOCK_SIZE_ / warp_size) * task_per_warp_tile;
  const size_t grid_size = ((len - 1) / keys_per_block) + 1;
  get_kernel<key_type, ref_counter_type, slabset, set_hasher, slab_hasher, empty_key,
             set_associativity, warp_size><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      d_keys, len, d_values, embedding_vec_size_, d_missing_index, d_missing_keys, d_missing_len,
      global_counter_, slot_counter_, capacity_in_set_, keys_, vals_, set_mutex_,
      task_per_warp_tile);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#endif

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Replace(const key_type* d_keys, const size_t len,
                                     const float* d_values, cudaStream_t stream,
                                     const size_t task_per_warp_tile) {
  // Check if it is a valid replacement
  if (len == 0) {
    return;
  }

  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Try to insert the <k,v> paris into the cache as long as there are unused slot
  // Then replace the <k,v> pairs into the cache
  const size_t keys_per_block = (BLOCK_SIZE_ / warp_size) * task_per_warp_tile;
  const size_t grid_size = ((len - 1) / keys_per_block) + 1;
  insert_replace_kernel<key_type, slabset, ref_counter_type, mutex, atomic_ref_counter_type,
                        set_hasher, slab_hasher, empty_key, set_associativity, warp_size>
      <<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_keys, d_values, embedding_vec_size_, len, keys_,
                                              vals_, slot_counter_, set_mutex_, global_counter_,
                                              capacity_in_set_, task_per_warp_tile);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#else
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Replace(const key_type* d_keys, const size_t len,
                                     const float* d_values, cudaStream_t stream,
                                     const size_t task_per_warp_tile) {
  // Check if it is a valid replacement
  if (len == 0) {
    return;
  }

  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Try to insert the <k,v> paris into the cache as long as there are unused slot
  // Then replace the <k,v> pairs into the cache
  const size_t keys_per_block = (BLOCK_SIZE_ / warp_size) * task_per_warp_tile;
  const size_t grid_size = ((len - 1) / keys_per_block) + 1;
  insert_replace_kernel<key_type, slabset, ref_counter_type, set_hasher, slab_hasher, empty_key,
                        set_associativity, warp_size><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      d_keys, d_values, embedding_vec_size_, len, keys_, vals_, slot_counter_, set_mutex_,
      global_counter_, capacity_in_set_, task_per_warp_tile);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#endif

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Update(const key_type* d_keys, const size_t len, const float* d_values,
                                    cudaStream_t stream, const size_t task_per_warp_tile) {
  // Check if it is a valid update request
  if (len == 0) {
    return;
  }

  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Update the value of input keys that are existed in the cache
  const size_t keys_per_block = (BLOCK_SIZE_ / warp_size) * task_per_warp_tile;
  const size_t grid_size = ((len - 1) / keys_per_block) + 1;
  update_kernel<key_type, slabset, set_hasher, slab_hasher, mutex, empty_key, set_associativity,
                warp_size><<<grid_size, BLOCK_SIZE_, 0, stream>>>(
      d_keys, len, d_values, embedding_vec_size_, capacity_in_set_, keys_, vals_, set_mutex_,
      task_per_warp_tile);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#else
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Update(const key_type* d_keys, const size_t len, const float* d_values,
                                    cudaStream_t stream, const size_t task_per_warp_tile) {
  // Check if it is a valid update request
  if (len == 0) {
    return;
  }

  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Update the value of input keys that are existed in the cache
  const size_t keys_per_block = (BLOCK_SIZE_ / warp_size) * task_per_warp_tile;
  const size_t grid_size = ((len - 1) / keys_per_block) + 1;
  update_kernel<key_type, slabset, set_hasher, slab_hasher, empty_key, set_associativity, warp_size>
      <<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_keys, len, d_values, embedding_vec_size_,
                                              capacity_in_set_, keys_, vals_, set_mutex_,
                                              task_per_warp_tile);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#endif

#ifdef LIBCUDACXX_VERSION
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Dump(key_type* d_keys, size_t* d_dump_counter,
                                  const size_t start_set_index, const size_t end_set_index,
                                  cudaStream_t stream) {
  // Check if it is a valid dump request
  if (start_set_index >= capacity_in_set_) {
    printf("Error: Invalid value for start_set_index. Nothing dumped.\n");
    return;
  }
  if (end_set_index <= start_set_index || end_set_index > capacity_in_set_) {
    printf("Error: Invalid value for end_set_index. Nothing dumped.\n");
    return;
  }

  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Set the global counter to 0 first
  CUDA_CHECK(cudaMemsetAsync(d_dump_counter, 0, sizeof(size_t), stream));

  // Dump keys from the cache
  const size_t grid_size =
      (((end_set_index - start_set_index) - 1) / (BLOCK_SIZE_ / warp_size)) + 1;
  dump_kernel<key_type, slabset, mutex, empty_key, set_associativity, warp_size>
      <<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_keys, d_dump_counter, keys_, set_mutex_,
                                              start_set_index, end_set_index);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#else
template <typename key_type, typename ref_counter_type, key_type empty_key, int set_associativity,
          int warp_size, typename set_hasher, typename slab_hasher>
void gpu_cache<key_type, ref_counter_type, empty_key, set_associativity, warp_size, set_hasher,
               slab_hasher>::Dump(key_type* d_keys, size_t* d_dump_counter,
                                  const size_t start_set_index, const size_t end_set_index,
                                  cudaStream_t stream) {
  // Check if it is a valid dump request
  if (start_set_index >= capacity_in_set_) {
    printf("Error: Invalid value for start_set_index. Nothing dumped.\n");
    return;
  }
  if (end_set_index <= start_set_index || end_set_index > capacity_in_set_) {
    printf("Error: Invalid value for end_set_index. Nothing dumped.\n");
    return;
  }

  // Device Restorer
  nv::CudaDeviceRestorer dev_restorer;
  // Check device
  dev_restorer.check_device(dev_);

  // Set the global counter to 0 first
  CUDA_CHECK(cudaMemsetAsync(d_dump_counter, 0, sizeof(size_t), stream));

  // Dump keys from the cache
  const size_t grid_size =
      (((end_set_index - start_set_index) - 1) / (BLOCK_SIZE_ / warp_size)) + 1;
  dump_kernel<key_type, slabset, empty_key, set_associativity, warp_size>
      <<<grid_size, BLOCK_SIZE_, 0, stream>>>(d_keys, d_dump_counter, keys_, set_mutex_,
                                              start_set_index, end_set_index);

  // Check for GPU error before return
  CUDA_CHECK(cudaGetLastError());
}
#endif

template class gpu_cache<unsigned int, uint64_t, std::numeric_limits<unsigned int>::max(),
                         SET_ASSOCIATIVITY, SLAB_SIZE>;
template class gpu_cache<long long, uint64_t, std::numeric_limits<long long>::max(),
                         SET_ASSOCIATIVITY, SLAB_SIZE>;
}  // namespace gpu_cache
