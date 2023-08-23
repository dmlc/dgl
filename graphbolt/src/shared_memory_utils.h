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
#include <tuple>
#include <vector>

namespace graphbolt {
namespace sampling {

/**
 * @brief SharedMemoryHelper is a helper class to write/read data structures
 * to/from shared memory.
 *
 * In order to write data structure to shared memory, we need to serialize the
 * data structure to a binary buffer and then write the buffer to the shared
 * memory. However, the size of the binary buffer is not known in advance. To
 * solve this problem, we use two shared memory objects: one for storing the
 * meta data and the other for storing the binary buffer. The meta data includes
 * the meta information of data structures such as size and shape. The size of
 * the meta data is decided by the user via `max_meta_memory_size`. The size of
 * the binary buffer is decided by the size of the data structures.
 *
 * To avoid repeated shared memory memory allocation, this helper class uses
 * lazy data structure writing. The data structures are written to the shared
 * memory only when `Flush` is called. The data structures are written in the
 * order of calling `WriteTorchArchive`, `WriteTorchTensor` and
 * `WriteTorchTensorDict`, and also read in the same order.
 *
 * The usage of this class as a writer is as follows:
 * @code{.cpp}
 * SharedMemoryHelper shm_helper("shm_name", 1024, true);
 * shm_helper.WriteTorchArchive(archive);
 * shm_helper.WriteTorchTensor(tensor);
 * shm_helper.WriteTorchTensorDict(tensor_dict);
 * shm_helper.Flush();
 * @endcode
 *
 * The usage of this class as a reader is as follows:
 * @code{.cpp}
 * SharedMemoryHelper shm_helper("shm_name", 1024, false);
 * auto archive = shm_helper.ReadTorchArchive();
 * auto tensor = shm_helper.ReadTorchTensor();
 * auto tensor_dict = shm_helper.ReadTorchTensorDict();
 * @endcode
 *
 *
 */
class SharedMemoryHelper {
 public:
  /*
   * @brief Constructor of the shared memory helper.
   * @param name The name of the shared memory.
   * @param max_meta_memory_size The maximum size of meta data.
   * @param is_creator Whether the process is the creator of the shared memory.
   */
  SharedMemoryHelper(
      const std::string& name, int64_t max_meta_memory_size, bool is_creator);

  void WriteTorchArchive(torch::serialize::OutputArchive&& archive);
  torch::serialize::InputArchive ReadTorchArchive();

  void WriteTorchTensor(torch::optional<torch::Tensor> tensor);
  torch::optional<torch::Tensor> ReadTorchTensor();

  void WriteTorchTensorDict(
      torch::optional<torch::Dict<std::string, torch::Tensor>> tensor_dict);
  torch::optional<torch::Dict<std::string, torch::Tensor>>
  ReadTorchTensorDict();

  /** @brief Flush the data structures to the shared memory. */
  void Flush();
  /** @brief Release the shared memory and return their left values. */
  std::pair<SharedMemoryPtr, SharedMemoryPtr> ReleaseSharedMemory();

 private:
  /** @brief Write the meta data to the shared memory. This function is
   * called by `Flush`.
   */
  void WriteTorchArchiveInternal(torch::serialize::OutputArchive& archive);
  /** @brief Write the tensor data to the shared memory. This function is
   * called by `Flush`.
   */
  void WriteTorchTensorInternal(torch::optional<torch::Tensor> tensor);

  inline void* GetCurrentMetaPtr() const {
    return static_cast<char*>(meta_shared_memory_->GetMemory()) + meta_offset_;
  }
  inline void* GetCurrentDataPtr() const {
    return static_cast<char*>(data_shared_memory_->GetMemory()) + data_offset_;
  }
  inline void MoveMetaPtr(int64_t offset) {
    TORCH_CHECK(
        meta_offset_ + offset <= max_meta_memory_size_,
        "The size of meta data exceeds the maximum size of shared memory.");
    meta_offset_ += offset;
  }
  inline void MoveDataPtr(int64_t offset) { data_offset_ += offset; }

  std::string name_;
  int64_t max_meta_memory_size_;
  bool is_creator_;
  // The shared memory objects for storing meta data and tensor data.
  SharedMemoryPtr meta_shared_memory_, data_shared_memory_;
  // The read/write offsets of the meta data and tensor data.
  int64_t meta_offset_, data_offset_;
  // The data structures to write to the shared memory. They are written to the
  // shared memory only when `Flush` is called.
  std::vector<torch::serialize::OutputArchive> meta_to_write_;
  std::vector<torch::optional<torch::Tensor>> tensors_to_write_;
};

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHM_UTILS_H_
