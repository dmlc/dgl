/**
 *  Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.h
 * @brief Numpy File Fetecher class.
 */

#ifdef HAVE_LIBRARY_LIBURING
#include <liburing.h>
#endif  // HAVE_LIBRARY_LIBURING

#include <torch/script.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace graphbolt {
namespace storage {

template <typename T>
class Future : public torch::CustomClassHolder {
 public:
  Future(c10::intrusive_ptr<c10::ivalue::Future> future, T value)
      : future_(future), value_(value) {}

  Future() = default;

  T Wait() {
    future_->waitAndThrow();
    return value_;
  }

 private:
  c10::intrusive_ptr<c10::ivalue::Future> future_;
  T value_;
};

/**
 * @brief Disk Numpy Fetecher class.
 */
class OnDiskNpyArray : public torch::CustomClassHolder {
 public:
  static constexpr int kGroupSize = 128;

  /** @brief Default constructor. */
  OnDiskNpyArray() = default;

  /**
   * @brief Constructor with given file path and data type.
   * @param path Path to the on disk numpy file.
   * @param dtype Data type of numpy array.
   *
   * @return OnDiskNpyArray
   */
  OnDiskNpyArray(
      std::string filename, torch::ScalarType dtype,
      const std::vector<int64_t>& shape, torch::optional<int64_t> num_threads);

  /** @brief Create a disk feature fetcher from numpy file. */
  static c10::intrusive_ptr<OnDiskNpyArray> Create(
      std::string path, torch::ScalarType dtype,
      const std::vector<int64_t>& shape, torch::optional<int64_t> num_threads);

  /** @brief Deconstructor. */
  ~OnDiskNpyArray();

  /**
   * @brief Parses the header of a numpy file to extract feature information.
   **/
  void ParseNumpyHeader();

  /**
   * @brief Read disk numpy file based on given index and transform to
   * tensor.
   */
  c10::intrusive_ptr<Future<torch::Tensor>> IndexSelect(torch::Tensor index);

#ifdef HAVE_LIBRARY_LIBURING
  /**
   * @brief Index-select operation on an on-disk numpy array using IO Uring for
   * asynchronous I/O.
   *
   * This function performs index-select operation on an on-disk numpy array. It
   * uses IO Uring for asynchronous I/O to efficiently read data from disk. The
   * input tensor 'index' specifies the indices of features to select. The
   * function reads features corresponding to the indices from the disk and
   * returns a new tensor containing the selected features.
   *
   * @param index A 1D tensor containing the indices of features to select.
   * @return A tensor containing the selected features.
   * @throws std::runtime_error If index is out of range.
   */
  c10::intrusive_ptr<Future<torch::Tensor>> IndexSelectIOUring(
      torch::Tensor index);

  void IndexSelectIOUringImpl(torch::Tensor index, torch::Tensor result);

#endif  // HAVE_LIBRARY_LIBURING
 private:
  int64_t ReadBufferSizePerThread() const {
    return (aligned_length_ + block_size_) * kGroupSize * 8;
  }

  const std::string filename_;  // Path to numpy file.
  int file_description_;        // File description.
  int64_t block_size_;          // Block size of the opened file.
  int64_t prefix_len_;          // Length of head data in numpy file.
  const std::vector<int64_t>
      feature_dim_;                // Shape of features, e.g. {N,M,K,L}.
  const torch::ScalarType dtype_;  // Feature data type.
  const int64_t feature_size_;     // Number of bytes of feature size.
  int64_t aligned_length_;         // Aligned feature_size.
  int num_thread_;                 // Default thread number.
  torch::Tensor read_tensor_;      // Provides temporary read buffer.
  char* read_buffer_;              // Aligned pointer to read_tensor.

#ifdef HAVE_LIBRARY_LIBURING
  std::unique_ptr<io_uring[]> io_uring_queue_;  // io_uring queue.
#endif                                          // HAVE_LIBRARY_LIBURING
};

}  // namespace storage
}  // namespace graphbolt
