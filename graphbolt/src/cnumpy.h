/**
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.h
 * @brief Numpy File Fetecher class.
 */

#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <torch/script.h>

#ifdef __linux__
#include <liburing.h>
#include <unistd.h>
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace graphbolt {
namespace storage {

/**
 * @brief Disk Numpy Fetecher class.
 */
class OnDiskNpyArray : public torch::CustomClassHolder {
 public:
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
      std::string filename, torch::ScalarType dtype, torch::Tensor shape);

  /** @brief Create a disk feature fetcher from numpy file. */
  static c10::intrusive_ptr<OnDiskNpyArray> Create(
      std::string path, torch::ScalarType dtype, torch::Tensor shape);

  /** @brief Deconstructor. */
  ~OnDiskNpyArray();

  /**
   * @brief Parses the header of a numpy file to extract feature information.
   **/
  void ParseNumpyHeader(torch::Tensor shape);

  /**
   * @brief Read disk numpy file based on given index and transform to
   * tensor.
   */
  torch::Tensor IndexSelect(torch::Tensor index);

#ifdef __linux__
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
  torch::Tensor IndexSelectIOUring(torch::Tensor index);
#endif  // __linux__

 private:
  std::string filename_;              // Path to numpy file.
  int file_description_;              // File description.
  size_t prefix_len_;                 // Length of head data in numpy file.
  std::vector<int64_t> feature_dim_;  // Shape of features, e.g. {N,M,K,L}.
  torch::ScalarType dtype_;           // Feature data type.
  int64_t feature_size_;              // Number of bytes of feature size.
  int num_thread_;                    // Default thread number.
  int64_t group_size_ = 512;          // Default group size.

#ifdef __linux__
  io_uring* io_uring_queue_;  // io_uring queue.
#endif
};

}  // namespace storage
}  // namespace graphbolt
