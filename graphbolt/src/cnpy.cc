#include "cnpy.h"

#include <fcntl.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <regex>
#include <stdexcept>

#define ALIGNMENT 4096

uint64_t time_diff(struct timespec *start, struct timespec *end) {
  return (end->tv_sec - start->tv_sec) * 1000000000 +
         (end->tv_nsec - start->tv_nsec);
}
void cnpy::test_pread_seq_full(
    const std::string &file_path, int thread_num, uint32_t block_size) {
  block_size = block_size * 1024;
  std::ifstream is(file_path, std::ifstream::binary | std::ifstream::ate);
  std::size_t file_size = is.tellg();
  is.close();

  int fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);

  int align_size =
      (file_size / block_size + (file_size % block_size != 0)) * block_size;

  char *buf = (char *)aligned_alloc(block_size, align_size);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  if (pread(fd, buf, align_size, 0) == -1) {
    printf("error\n");
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  auto duration = time_diff(&start, &end) / 1000000.0;
  uint64_t bandwidth = (align_size / (1024 * 1024)) / (duration / 1000);
  uint64_t *array = (uint64_t *)buf;
  uint64_t sum = 0;
  for (size_t i = 0; i < file_size / sizeof(uint64_t); ++i) {
    sum ^= array[i];
  }
  printf("%lu, %.3fms, %luMB/s\n", sum, duration, bandwidth);
  free(buf);
  close(fd);
}

void cnpy::NpyArray::load_all() {
  std::ifstream is(filename, std::ifstream::binary | std::ifstream::ate);
  std::size_t file_size = is.tellg();
  is.close();

  printf("file size: %lu\n", file_size);

  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    printf("open error\n");
  }

  int align_size =
      (file_size / ALIGNMENT + (file_size % ALIGNMENT != 0)) * ALIGNMENT;

  data_holder = (char *)aligned_alloc(ALIGNMENT, align_size);

  if (pread(fd, data_holder, align_size, 0) == -1) {
    printf("error\n");
  }
  close(fd);
}

torch::Tensor cnpy::NpyArray::index_select(torch::Tensor idx) {
  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;

  int64_t num_idx = idx.numel();

  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  int64_t upper = 0;
  int64_t in_upper = 0;
  int64_t in_lower = 0;
  std::vector<int64_t> in_offset(num_idx);
  std::vector<int64_t> page_offset{};
  for (int64_t i = 0; i < num_idx; i++) {
    // from true offset to in-offset
    // printf("idx: %u\n", idx_data[i]);
    int64_t offset = idx_data[i] * feature_size + prefix_len;
    // printf("offset: %u\n", offset);
    int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
    // printf("algn: %u\n", aligned_offset);
    int64_t residual = offset - aligned_offset;

    if (offset > upper) {
      in_offset[i] = in_upper + residual;
      upper = aligned_offset + ALIGNMENT;
      page_offset.push_back(aligned_offset);
      // printf("put page: %u\n", aligned_offset);
      in_lower = in_upper;
      in_upper += ALIGNMENT;
    } else {
      in_offset[i] = in_lower + residual;
    }
    if (residual + feature_size > ALIGNMENT) {
      upper += ALIGNMENT;
      in_lower = in_upper;
      in_upper += ALIGNMENT;
      page_offset.push_back(aligned_offset + ALIGNMENT);
      // printf("put page: %u\n", aligned_offset + ALIGNMENT);
    }
  }

  // printf("page offset: ");
  // for (auto i : page_offset) {
  //   printf("%u ", i);
  // }
  // printf("\n");
  // printf("in offset: ");
  // for (auto i : in_offset) {
  //   printf("%u ", i);
  // }
  // printf("\n");

  char *read_buffer =
      (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * page_offset.size());

  for (size_t i = 0; i < page_offset.size(); i++) {
    if (pread(fd, read_buffer + i * ALIGNMENT, ALIGNMENT, page_offset[i]) ==
        -1) {
      fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
  }

  // printf("page view\n");
  // for (uint i = 0; i < 10; i++) {
  //   printf("preview %.3f\n", reinterpret_cast<float *>(read_buffer + 80)[i]);
  // }

  for (uint64_t i = 0; i < in_offset.size(); i++) {
    memcpy(
        result_buffer + feature_size * i, read_buffer + in_offset[i],
        feature_size);
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

  free(read_buffer);
  close(fd);

  return result;
}

void cnpy::NpyArray::print_npy_header() {
  printf("prefix size: %lu\n", prefix_len);
  std::cout << "feature_shape:" << feature_shape << std::endl;
  printf(
      "file: %s\nword_size: %lu \nfeature_dim: %lu\nfortran_order: %u\n",
      filename.c_str(), word_size, feature_dim, fortran_order);
  for (size_t i = 0; i < shape.size(); i++) {
    printf("shape %lu\n", shape[i]);
  }
}

void cnpy::NpyArray::parse_npy_header(FILE *fp) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
  std::string header = fgets(buffer, 256, fp);
  assert(header[header.size() - 1] == '\n');
  prefix_len = 11 + header.size();

  size_t loc1, loc2;

  // fortran order
  loc1 = header.find("fortran_order");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

  // shape
  loc1 = header.find("(");
  loc2 = header.find(")");
  if (loc1 == std::string::npos || loc2 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: '(' or ')'");

  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  shape.clear();

  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  while (std::regex_search(str_shape, sm, num_regex)) {
    shape.emplace_back(std::stoi(sm[0].str()));
    str_shape = sm.suffix().str();
  }
  num_vals = feature_dim = 1;

  for (size_t i = 0; i < shape.size(); i++) {
    num_vals *= shape[i];
  }

  std::vector<int64_t> shape_vec;
  for (size_t i = 1; i < shape.size(); i++) {
    feature_dim *= shape[i];
    shape_vec.push_back(shape[i]);
  }

  at::TensorOptions opts = at::TensorOptions().dtype(at::kInt);
  feature_shape =
      torch::from_blob(shape_vec.data(), {int64_t(shape_vec.size())}, opts)
          .clone();

  loc1 = header.find("descr");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'descr'");
  loc1 += 9;
  bool littleEndian =
      (header[loc1] == '<' || header[loc1] == '|' ? true : false);
  assert(littleEndian);

  std::string str_ws = header.substr(loc1 + 2);
  loc2 = str_ws.find("'");
  word_size = atoi(str_ws.substr(0, loc2).c_str());
}