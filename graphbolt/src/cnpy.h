#include <stdint.h>
#include <stdlib.h>
#include <torch/script.h>
#include <zlib.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

struct NpyArray {
  NpyArray() : shape(0), word_size(0), fortran_order(0), num_vals(0) {}

  NpyArray(std::string _filename) : filename(_filename) {
    FILE *fp = fopen(_filename.c_str(), "rb");
    if (!fp)
      throw std::runtime_error("npy_load: Unable to open file " + _filename);
    parse_npy_header(fp);
    fclose(fp);
  }

  template <typename T>
  T *data() {
    return reinterpret_cast<T *>(data_holder + 80);
  }

  template <typename T>
  const T *data() const {
    return reinterpret_cast<T *>(data_holder + 80);
  }

  template <typename T>
  std::vector<T> as_vec() const {
    const T *p = data<T>();
    return std::vector<T>(p, p + num_vals);
  }

  void parse_npy_header(FILE *fp);
  void print_npy_header();
  void load_all();
  torch::Tensor feature_size() { return feature_shape; }
  torch::Tensor index_select_all(torch::Tensor idx);
  torch::Tensor index_select_pread(torch::Tensor idx);
  torch::Tensor index_select_aio(torch::Tensor idx);
  torch::Tensor index_select_iouring(torch::Tensor idx);

  size_t num_bytes() const { return num_vals * word_size; }

  char *data_holder;
  std::string filename;
  std::vector<int64_t> shape;
  size_t word_size;
  bool fortran_order;
  size_t prefix_len;
  size_t num_vals;
  signed long feature_dim;
  torch::Tensor feature_shape;
};

void test_pread_seq_full(
    const std::string &file_path, int thread_num, uint32_t block_size);

}  // namespace cnpy