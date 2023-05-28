/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file shared_memory_utils.h
 * @brief Share memory utilities.
 */
#ifndef GRAPHBOLT_SHM_UTILS_H_
#define GRAPHBOLT_SHM_UTILS_H_

#include <graphbolt/shared_memory.h>
#include <torch/torch.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

/** @brief SharedMemoryTensors includes: (1) two share memory objects holding
 * tensor meta information and data respectively; (2) a vector of tensors on
 * shared memory. */
using SharedMemoryTensors =
    std::tuple<SharedMemoryPtr, SharedMemoryPtr, std::vector<torch::Tensor>>;

/**
 * @brief Copy torch tensors to shared memory.
 *
 * The function has two steps:
 * 1. Copy meta info to shared memory `shared_memory_name + "_meta"`. This is to
 * make sure that other loading processes can get the meta info of tensors.
 * 2. Copy tensors to shared memory `shared_memory_name + "_data"`, which can be
 * loaded by other processes with meta info.
 *
 * @param name The name of shared memory.
 * @param tensors The tensors to copy.
 * @param meta_memory_size The maximum size of meta memory.
 *
 * @return A tuple of tensor meta shared memory, tensor data shared memory and
 * shared tensors.
 */
SharedMemoryTensors CopyTensorsToSharedMemory(
    const std::string& name, const std::vector<torch::Tensor>& tensors,
    int64_t meta_memory_size);

/**
 * @brief Load torch tensors from shared memory.
 *
 * The loading process follows the same logic as `CopyTensorsToSharedMemory`.
 *
 * @param name The name of shared memory.
 * @param meta_memory_size The maximum size of meta memory.
 *
 * @return A tuple of tensor meta shared memory, tensor data shared memory and
 * shared tensors.
 */
SharedMemoryTensors LoadTensorsFromSharedMemory(
    const std::string& name, int64_t meta_memory_size);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHM_UTILS_H_
