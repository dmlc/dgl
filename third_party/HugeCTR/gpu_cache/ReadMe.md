# GPU Embedding Cache

This project implements an embedding cache on GPU memory that is designed for CTR inference and training workload.

The cache stores the hot pairs, (embedding id, embedding vectors), on GPU memory.
Storing the data on GPU memory reduces the traffic to the parameter server when performing embedding table lookup.

The cache is designed for CTR inference and training, it has following features and restrictions:

* All the backup memory-side operations are performed by the parameter server.
  These operations include prefetching, latency hiding, and so on.
* This is a single-GPU design.
  Each cache belongs to one GPU.
* The cache is thread-safe: multiple workers, CPU threads, can concurrently call the API of a single cache object with well-defined behavior.
* The cache implements a least recently used (LRU) replacement algorithm so that it caches the most recently queried embeddings.
* The embeddings stored inside the cache are unique: there are no duplicated embedding IDs in the cache.

## Project Structure

This project is a stand-alone module in HugeCTR project.
The root folder of this project is the `gpu_cache` folder under the HugeCTR root directory.

The `include` folder contains the headers for the cache library and the `src` folder contains the implementations and Makefile for the cache library.
The `test` folder contains a test that tests the correctness and performance of the GPU embedding cache.
The test also acts as sample code that shows how to use the cache.

The `nv_gpu_cache.hpp` file contains the definition of the main class, `gpu_cache`, that implements the GPU embedding cache.
The `nv_gpu_cache.cu` file contains the implementation.

As a module of HugeCTR, this project is built with and used by the HugeCTR project.

## Supported Data Types

* The cache supports 32 and 64-bit scalar integer types for the key (embedding ID) type.
  For example, the data type declarations `unsigned int` and `long long` match these integer types.
* The cache supports a vector of floats for the value (embedding vector) type.
* You need to specify an empty key to indicate the empty bucket.
  Do not use an empty key to represent any real key.
* Refer to the instantiation code at the end of the `nv_gpu_cache.cu` file for template parameters.

## Requirements

* NVIDIA GPU >= Volta (SM 70).
* CUDA environment >= 11.0.
* (Optional) libcu++ library >= 1.1.0.
  The CUDA Toolkit 11.0 (Early Access) and above meets the required library version.
  Using the libcu++ library provides better performance and more precisely-defined behavior.
  You can enable libcu++ library by defining the `LIBCUDACXX_VERSION` macro when you compile.
  Otherwise, the libcu++ library is not enabled.
* The default building option for HugeCTR is to disable the libcu++ library.

## Usage Overview

```c++
template<typename key_type,
         typename ref_counter_type,
         key_type empty_key,
         int set_associativity,
         int warp_size,
         typename set_hasher = MurmurHash3_32<key_type>,
         typename slab_hasher = Mod_Hash<key_type, size_t>>
class gpu_cache{
public:
    //Ctor
    gpu_cache(const size_t capacity_in_set, const size_t embedding_vec_size);

    //Dtor
    ~gpu_cache();

    // Query API, i.e. A single read from the cache
    void Query(const key_type* d_keys,
               const size_t len,
               float* d_values,
               uint64_t* d_missing_index,
               key_type* d_missing_keys,
               size_t* d_missing_len,
               cudaStream_t stream,
               const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO);

    // Replace API, i.e. Follow the Query API to update the content of the cache to Most Recent
    void Replace(const key_type* d_keys,
                 const size_t len,
                 const float* d_values,
                 cudaStream_t stream,
                 const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO);

    // Update API, i.e. update the embeddings which exist in the cache
    void Update(const key_type* d_keys,
                const size_t len,
                const float* d_values,
                cudaStream_t stream,
                const size_t task_per_warp_tile = TASK_PER_WARP_TILE_MACRO);

    // Dump API, i.e. dump some slabsets' keys from the cache
    void Dump(key_type* d_keys,
              size_t* d_dump_counter,
              const size_t start_set_index,
              const size_t end_set_index,
              cudaStream_t stream);

};
```

## API

`Constructor`

To create a new embedding cache, you need to provide the following:

* Template parameters:
    + key_type: the data type of embedding ID.
    + ref_counter_type: the data type of the internal counter. This data type should be 64bit unsigned integer(i.e. uint64_t), 32bit integer has the risk of overflow.
    + empty_key: the key value indicate for empty bucket(i.e. The empty key), user should never use empty key value to represent any real keys.
    + set_associativity: the hyper-parameter indicates how many slabs per cache set.(See `Performance hint` session below)
    + warp_size: the hyper-parameter indicates how many [key, value] pairs per slab. Acceptable value includes 1/2/4/8/16/32.(See `Performance hint` session below)
    + For other template parameters just use the default value.
* Parameters:
    + capacity_in_set: # of cache set in the embedding cache. So the total capacity of the embedding cache is `warp_size * set_associativity * capacity_in_set` [key, value] pairs.
    + embedding_vec_size: # of float per a embedding vector.
* The host thread will wait for the GPU kernels to complete before returning from the API, thus this API is synchronous with CPU thread. When returned, the initialization process of the cache is already done.
* The embedding cache will be created on the GPU where user call the constructor. Thus, user should set the host thread to the target CUDA device before creating the embedding cache. All resources(i.e. device-side buffers, CUDA streams) used later for this embedding cache should be allocated on the same CUDA device as the embedding cache.
* The constructor can be called only once, thus is not thread-safe.

`Destructor`

* The destructor clean up the embedding cache. This API should be called only once when user need to delete the embedding cache object, thus is not thread-safe.

`Query`

* Search `len` elements from device-side buffers `d_keys` in the cache and return the result in device-side buffer `d_values` if a key is hit in the cache.
* If a key is missing, the missing key and its index in the `d_keys` buffer will be returned in device-side buffers `d_missing_keys` and `d_missing_index`. The # of missing key will be return in device-side buffer `d_missing_len`. For simplicity, these buffers should have the same length as `d_keys` to avoid out-of-bound access.
* The GPU kernels will be launched in `stream` CUDA stream.
* The host thread will return from the API immediately after the kernels are launched, thus this API is Asynchronous with CPU thread.
* The keys to be queried in the `d_keys` buffer can have duplication. In this case, user will get duplicated returned values or missing information.
* This API is thread-safe and can be called concurrently with other APIs.
* For hyper-parameter `task_per_warp_tile`, see `Performance hint` session below.

`Replace`

* The API will replace `len` [key, value] pairs listed in `d_keys` and `d_values` into the embedding cache using the LRU replacement algorithm.
* The GPU kernels will be launched in `stream` CUDA stream.
* The host thread will return from the API immediately after the kernels are launched, thus this API is Asynchronous with CPU thread.
* The keys to be replaced in the `d_keys` buffer can have duplication and can be already stored inside the cache. In these cases, the cache will detect any possible duplication and maintain the uniqueness of all the [key ,value] pairs stored in the cache.
* This API is thread-safe and can be called concurrently with other APIs.
* This API will first try to insert the [key, value] pairs into the cache if there is any empty slot. If the cache is full, it will do the replacement.
* For hyper-parameter `task_per_warp_tile`, see `Performance hint` session below.

`Update`

* The API will search for `len` keys listed in `d_keys` buffer within the cache. If a key is found in the cache, this API will update the value associated with the key to the corresponding values provided in `d_values` buffer. If a key is not found in the cache, this API will do nothing to this key.
* The GPU kernels will be launched in `stream` CUDA stream.
* The host thread will return from the API immediately after the kernels are launched, thus this API is Asynchronous with CPU thread.
* If the keys to be updated in the `d_keys` buffer have duplication, all values associated with this key in the `d_values` buffer will be updated to the cache atomically. The final result depends on the order of updating the value.
* This API is thread-safe and can be called concurrently with other APIs.
* For hyper-parameter `task_per_warp_tile`, see `Performance hint` session below.

`Dump`

* The API will dump all the keys stored in [`start_set_index`, `end_set_index`) cache sets to `d_keys` buffer as a linear array(the key order is not guaranteed). The total # of keys dumped will be reported in `d_dump_counter` variable.
* The GPU kernels will be launched in `stream` CUDA stream.
* The host thread will return from the API immediately after the kernels are launched, thus this API is Asynchronous with CPU thread.
* This API is thread-safe and can be called concurrently with other APIs.

## More Information

* The detailed introduction of the GPU embedding cache data structure is presented at GTC China 2020: https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cns20626-%e4%bd%bf%e7%94%a8+gpu+embedding+cache+%e5%8a%a0%e9%80%9f+ctr+%e6%8e%a8%e7%90%86%e8%bf%87%e7%a8%8b
* The `test` folder contains a example of using the GPU embedding cache.
* This project is used by `embedding_cache` class in `HugeCTR/include/inference/embedding_cache.hpp` which can be used as an example.

## Performance Hint

* The hyper-parameter `warp_size` should be keep as 32 by default. When the length for Query or Replace operations is small(~1-50k), user can choose smaller warp_size and increase the total # of cache set(while maintaining the same cache size) to increase the parallelism and improve the performance.
* The hyper-parameter `set_associativity` is critical to performance:
    + If set too small, may cause load imbalance between different cache sets(lower down the effective capacity of the cache, lower down the hit rate). To prevent this, the embedding cache uses a very random hash function to hash the keys to different cache set, thus will achieve load balance statistically. However, larger cache set will tends to have better load balance.
    + If set too large, the searching space for a single key will be very large. The performance of the embedding cache API will drop dramatically. Also, each set will be accessed exclusively, thus the more cache sets the higher parallelism can be achieved.
    + Recommend setting `set_associativity` to 2 or 4.
* The runtime hyper-parameter `task_per_warp_tile` is set to 1 as default parameter, thus users don't need to change their code to accommodate this interface change. This hyper-parameter determines how many keys are been queried/replaced/updated by a single warp tile. The acceptable value is between [1, `warp_size`]. For small to medium size operations to the cache, less task per warp tile can increase the total # of warp tiles running concurrently on the GPU chip, thus can bring significant performance improvement. For large size operations to the cache, the increased # of warp tile will not bring any performance improvement(even a little regression on the performance, ~5%). User can choose the value for this parameter based on the value of `len` parameter.
* The GPU is designed for optimizing throughput. Always try to batch up the inference task and try to have larger `query_size`.
* As the APIs of the embedding cache is asynchronous with host threads. Try to optimize the E2E inference pipeline by overlapping asynchronous tasks on GPU or between CPU and GPU. For example, after retrieving the missing values from the parameter server, user can combine the missing values with the hit values and do the rest of inference pipeline at the same time with the `Replace` API. Replacement is not necessarily happens together with Query all the time, user can do query multiple times then do a replacement if the hit rate is acceptable.
* Try different cache capacity and evaluate the hit rate. If the capacity of embedding cache can be larger than actual embedding footprint, the hit rate can be as high as 99%+.
