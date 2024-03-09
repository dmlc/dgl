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
#endif
#include <unistd.h>

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
  void ParseNumpyHeader(FILE* fp);

  /**
   * @brief Read disk numpy file based on given index and transform to
   * tensor.
   */
  torch::Tensor IndexSelectIOuring(torch::Tensor idx);

 private:
  std::string filename;           // Path to numpy file.
  torch::ScalarType dtype;        // Feature data type.
  int feature_fd;                 // File description.
  std::vector<int64_t> feat_dim;  // Shape of features, e.g. {N,M,K,L}.
  int num_thd = 4;                // Default thread number.
  int64_t group_size = 512;       // Default group size.
  int64_t feature_size;           // Number of bytes of feature size.
  size_t prefix_len;              // Length of head data in numpy file.

#ifdef __linux__
  io_uring ring[4];  // io_uring queue.
#endif
};

c10::intrusive_ptr<OnDiskNpyArray> CreateDiskFetcher(
    std::string path, torch::ScalarType dtype);

}  // namespace storage
}  // namespace graphbolt
