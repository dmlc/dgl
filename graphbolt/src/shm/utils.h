/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file shm/utils.h
 * @brief Share memory utilities.
 */
#ifndef GRAPHBOLT_SHM_UTILS_H_
#define GRAPHBOLT_SHM_UTILS_H_

#include <graphbolt/shared_mem.h>
#include <torch/torch.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace graphbolt {
namespace sampling {

/**
 * @brief Copy torch tensors to shared memory.
 *
 * The function has two steps:
 * 1. Copy meta info to shared memory `shared_memory_name + "_meta"`. This is to
 * make sure that the loading process can get the meta info of tensors.
 * 2. Copy tensors to shared memory `shared_memory_name + "_data"`, which can be
 * loaded by other processes with meta info.
 *
 * @param shared_memory_name The name of shared memory.
 * @param tensors The tensors to copy.
 * @param meta_memory_size The maximum size of meta memory.
 *
 * @return A pair of shared memory and tensors.
 */
std::pair<
    std::vector<std::unique_ptr<SharedMemory>>, std::vector<torch::Tensor>>
CopyTensorsToSharedMemory(
    const std::string& shared_memory_name,
    const std::vector<torch::Tensor>& tensors, int64_t meta_memory_size);

/**
 * @brief Load torch tensors from shared memory.
 *
 * The loading process follows the same logic as `CopyTensorsToSharedMemory`.
 *
 * @param shared_memory_name The name of shared memory.
 * @param meta_memory_size The maximum size of meta memory.
 *
 * @return A pair of shared memory and tensors.
 */
std::pair<
    std::vector<std::unique_ptr<SharedMemory>>, std::vector<torch::Tensor>>
LoadTensorsFromSharedMemory(
    const std::string& shared_memory_name, int64_t meta_memory_size);

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHM_UTILS_H_
