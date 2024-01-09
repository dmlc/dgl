/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file graphbolt/shared_memory.h
 * @brief Header file of graphbolt shared memory.
 */
#ifndef GRAPHBOLT_SHARED_MEMORY_H_
#define GRAPHBOLT_SHARED_MEMORY_H_

#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

#include <memory>
#include <string>

namespace graphbolt {
namespace sampling {

/**
 * @brief The SharedMemory is responsible for storing all the necessary
 * parameters of the buffer. Each SharedMemory instance is associated with a
 * shared memory object. The object will be removed when the associated
 * SharedMemory instance is destroyed.
 */
class SharedMemory {
 public:
  /**
   * @brief Constructor of the shared memory.
   * @param name The name of the shared memory.
   */
  explicit SharedMemory(const std::string& name);

  SharedMemory(const SharedMemory&) = delete;
  SharedMemory& operator=(const SharedMemory&) = delete;

  /**
   * @brief The destructor is responsible for unmapping the shared memory and
   * removing the associated shared memory object.
   */
  ~SharedMemory();

  /** @brief Get the name of shared memory. */
  std::string GetName() const { return name_; }

  /** @brief Get the pointer to the shared memory. */
  void* GetMemory() const { return ptr_; }

  /** @brief Get the size of the shared memory. */
  size_t GetSize() const { return size_; }

  /**
   * @brief Creates the shared memory object and map the shared memory.
   *
   * @param size The size of the shared memory.
   * @return The pointer to the shared memory.
   */
  void* Create(size_t size);

  /**
   * @brief Open the created shared memory object and map the shared memory.
   *
   */
  void* Open();

  /**
   * @brief Check if the shared memory exists.
   *
   * @param name The name of the shared memory.
   * @return True if the shared memory exists, otherwise False.
   */
  static bool Exists(const std::string& name);

 private:
  /** @brief The name of the shared memory. */
  std::string name_;

  /** @brief The size of the shared memory. */
  size_t size_;

  /** @brief The pointer of the shared memory. */
  void* ptr_;

#ifdef _WIN32

  /** @brief The handle of the shared memory object. */
  HANDLE handle_;

#else  // _WIN32

  /** @brief The file descriptor of the shared memory object. */
  int file_descriptor_;

  /**
   * @brief Whether the shared memory is created by the instance.
   *
   * The instance that creates the shared memory object is responsible for
   * unlinking the shared memory object.
   */
  bool is_creator_;

#endif  // _WIN32
};

using SharedMemoryPtr = std::unique_ptr<SharedMemory>;

}  // namespace sampling
}  // namespace graphbolt

#endif  // GRAPHBOLT_SHARED_MEMORY_H_
