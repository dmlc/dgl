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

  /** @brief Constructor with given file path. */
  OnDiskNpyArray(std::string filename, torch::ScalarType dtype);

  /** @brief Deconstructor. */
  ~OnDiskNpyArray();

  /**
   * @brief Parse numpy meta data.
   */
  void ParseNumpyHeader();

  /**
   * @brief Read disk numpy file based on given index and transform to
   * tensor.
   */
  torch::Tensor IndexSelectIOuring(torch::Tensor idx);

 private:
  std::string filename_;           // Path to numpy file.
  torch::ScalarType dtype_;        // Feature data type.
  int feature_fd_;                 // File description.
  std::vector<int64_t> feat_dim_;  // Shape of features, e.g. {N,M,K,L}.
  int num_thd_;                    // Default thread number.
  int64_t group_size_ = 512;       // Default group size.
  int64_t feature_size_;           // Number of bytes of feature size.
  size_t prefix_len_;              // Length of head data in numpy file.

#ifdef __linux__
  io_uring* ring_;  // io_uring queue.
#endif
};

c10::intrusive_ptr<OnDiskNpyArray> CreateDiskFetcher(
    std::string path, torch::ScalarType dtype);

}  // namespace storage
}  // namespace graphbolt
