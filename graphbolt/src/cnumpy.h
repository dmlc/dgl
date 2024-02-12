/**
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.h
 * @brief Numpy File Fetecher class.
 */

#include <stdint.h>
#include <torch/script.h>
#include <zlib.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <typeinfo>

namespace cnpy {

/**
 * @brief Disk Numpy Fetecher class.
 */
class NpyArray {
 public:
  /** @brief Constructor with empty file path. */
  NpyArray() : shape(0), word_size(0), fortran_order(0) {}

  /** @brief Constructor with given file path. */
  NpyArray(std::string _filename) : filename(_filename) {
    FILE *fp = fopen(_filename.c_str(), "rb");
    if (!fp)
      throw std::runtime_error("npy_load: Unable to open file " + _filename);
    parse_npy_header(fp);
    fclose(fp);
  }

  /** @brief Get the pointer of numpy data loaded in memory. */
  template <typename T>
  T *data() {
    return reinterpret_cast<T *>(data_holder + prefix_len);
  }
  template <typename T>
  const T *data() const {
    return reinterpret_cast<T *>(data_holder + prefix_len);
  }
  /**
   * @brief Parse numpy meta data.
   */
  void parse_npy_header(FILE *fp);

  /**
   * @brief Get the feature shape of numpy data according to meta data.
   */
  torch::Tensor feature_size() { return feature_shape; }

  /**
   * @brief Read disk numpy file based on given index and transform to
   tensor.
   */
  torch::Tensor index_select_iouring(torch::Tensor idx);

 private:
  char *data_holder;
  std::string filename;
  std::vector<int64_t> shape;
  size_t word_size;
  bool fortran_order;
  size_t prefix_len;
  signed long feature_dim;
  torch::Tensor feature_shape;
};

}  // namespace cnpy