/**
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.cc
 * @brief Numpy File Fetecher class.
 */

#include "cnumpy.h"

#include <fcntl.h>
#include <liburing.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

#define ALIGNMENT 4096

namespace graphbolt {
namespace storage {

/**
 * @brief Read disk numpy file based on given index and transform to tensor.
 */
torch::Tensor OnDiskNpyArray::index_select_iouring(
    torch::Tensor idx, torch::ScalarType dtype) {
  int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feat_len * word_size;
  // The min page size to contain one feature.
  int64_t align_len = (feature_size + ALIGNMENT) & (long)~(ALIGNMENT - 1);
  int64_t num_idx = idx.numel();

  static int num_thd = 4;           // default thread number.
  static int64_t group_size = 512;  // default group size.
  static struct io_uring ring[4];   // io_uring queue.

  // Init io_uring fetcher.
  for (int64_t t = 0; t < 4; t++) {
    io_uring_queue_init(group_size, &ring[t], 0);
  }

  omp_set_num_threads(num_thd);

  char *read_buffer = (char *)aligned_alloc(
      ALIGNMENT, (align_len + ALIGNMENT) * group_size * num_thd);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  // Record the inside offsets of feteched features.
  int64_t residual[group_size * num_thd];

  // Indicator for index error.
  bool error_flag = false;
#pragma omp parallel for num_threads(num_thd)
  for (int64_t n = 0; n < num_idx; n += group_size) {
    if (!error_flag) {
      auto thd_id = omp_get_thread_num();
      int64_t r = std::min(num_idx, n + group_size);
      // Compute the read block offsets and save them as io_uring requests.
      for (int64_t m = 0; m < r - n; m++) {
        int64_t i = idx_data[n + m];  // feature id.
        if (i > feat_dim[0]) {
          error_flag = true;
          break;
        }
        int64_t offset = i * feature_size + prefix_len;
        int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
        residual[thd_id * group_size + m] = offset - aligned_offset;

        int64_t read_size;
        if (residual[thd_id * group_size + m] + feature_size > ALIGNMENT) {
          read_size = align_len + ALIGNMENT;
        } else {
          read_size = align_len;
        }

        // Putting requests into io_uring queue.
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring[thd_id]);
        io_uring_prep_read(
            sqe, feature_fd,
            read_buffer + ((align_len + ALIGNMENT) * group_size * thd_id) +
                ((align_len + ALIGNMENT) * m),
            read_size, aligned_offset);
      }
      if (error_flag) {
        continue;
      }

      // Submit I/O requests.
      io_uring_submit(&ring[thd_id]);

      // Wait for completion of I/O requests.
      int64_t num_finish = 0;
      // Wait until all the disk blocks are loaded in current group.
      while (num_finish < r - n) {
        struct io_uring_cqe *cqe;
        if (io_uring_wait_cqe(&ring[thd_id], &cqe) < 0) {
          perror("io_uring_wait_cqe");
          abort();
        }
        struct io_uring_cqe *cqes[group_size];
        int cqecount = io_uring_peek_batch_cqe(&ring[thd_id], cqes, group_size);
        if (cqecount == -1) {
          perror("io_uring_peek_batch error\n");
          abort();
        }
        // Move the head pointer of completion queue.
        io_uring_cq_advance(&ring[thd_id], cqecount);
        num_finish += cqecount;
      }

      // Copy the features in the disk blocks to the result buffer.
      for (int64_t m = 0; m < r - n; m++) {
        memcpy(
            result_buffer + feature_size * (n + m),
            read_buffer + ((align_len + ALIGNMENT) * group_size * thd_id) +
                ((align_len + ALIGNMENT) * m +
                 residual[thd_id * group_size + m]),
            feature_size);
      }
    }
  }
  // IO queue exit.
  for (int64_t t = 0; t < num_thd; t++) {
    io_uring_queue_exit(&ring[t]);
  }

  auto result = torch::empty({0});
  if (!error_flag) {
    auto options = torch::TensorOptions()
                       .dtype(dtype)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);

    std::vector<int64_t> shape;
    shape.push_back(num_idx);
    shape.insert(shape.end(), feat_dim.begin() + 1, feat_dim.end());
    result = torch::from_blob(result_buffer, torch::IntArrayRef(shape), options)
                 .clone();
  } else {
    throw std::runtime_error("IndexError: Index out of range.");
  }

  free(read_buffer);
  free(result_buffer);
  close(feature_fd);

  return result;
}

/**
 * @brief Parse numpy meta data.
 */
void OnDiskNpyArray::parse_npy_header(FILE *fp) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
  std::string header = fgets(buffer, 256, fp);
  assert(header[header.size() - 1] == '\n');
  prefix_len = 11 + header.size();

  size_t loc1, loc2;

  // Get shape location.
  loc1 = header.find("(");
  loc2 = header.find(")");
  if (loc1 == std::string::npos || loc2 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: '(' or ')'");

  std::regex num_regex("[0-9][0-9]*");
  std::smatch sm;
  std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
  while (std::regex_search(str_shape, sm, num_regex)) {
    feat_dim.emplace_back(std::stoi(sm[0].str()));
    str_shape = sm.suffix().str();
  }

  feat_len = 1;
  std::vector<int64_t> shape;
  for (size_t i = 1; i < feat_dim.size(); i++) {
    feat_len *= feat_dim[i];
    shape.push_back(feat_dim[i]);
  }

  at::TensorOptions opts = at::TensorOptions().dtype(at::kLong);
  feat_shape =
      torch::from_blob(shape.data(), {int64_t(shape.size())}, opts).clone();

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

}  // namespace storage
}  // namespace graphbolt
