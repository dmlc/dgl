/**
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.cc
 * @brief Numpy File Fetecher class.
 */

#ifdef __linux__
#include "cnumpy.h"

#include <torch/torch.h>

#include <cstring>
#include <regex>
#include <stdexcept>

namespace graphbolt {
namespace storage {

#define ALIGNMENT 4096

c10::intrusive_ptr<OnDiskNpyArray> CreateDiskFetcher(
    std::string path, torch::ScalarType dtype) {
  return c10::make_intrusive<OnDiskNpyArray>(path, dtype);
}

OnDiskNpyArray::OnDiskNpyArray(std::string filename, torch::ScalarType dtype)
    : filename_(filename), dtype_(dtype) {
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp)
    throw std::runtime_error("npy_load: Unable to open file " + filename);
  ParseNumpyHeader(fp);
  fclose(fp);
  feature_fd_ = open(filename.c_str(), O_RDONLY | O_DIRECT);

  // Get system max thread number.
  num_thd_ = torch::get_num_threads();
  ring_ = new io_uring[num_thd_];

  // Init io_uring queue.
  for (int64_t t = 0; t < num_thd_; t++) {
    io_uring_queue_init(group_size_, &ring_[t], 0);
  }
}

OnDiskNpyArray::~OnDiskNpyArray() {
  // IO queue exit.
  for (int64_t t = 0; t < num_thd_; t++) {
    io_uring_queue_exit(&ring_[t]);
  }
  close(feature_fd_);
}

void OnDiskNpyArray::ParseNumpyHeader(FILE *fp) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
  std::string header = fgets(buffer, 256, fp);
  assert(header[header.size() - 1] == '\n');
  prefix_len_ = 11 + header.size();

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
    feat_dim_.emplace_back(std::stoi(sm[0].str()));
    str_shape = sm.suffix().str();
  }

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
  size_t word_size = atoi(str_ws.substr(0, loc2).c_str());

  signed long feat_len = 1;
  for (size_t i = 1; i < feat_dim_.size(); i++) {
    feat_len *= feat_dim_[i];
  }

  feature_size_ = feat_len * word_size;
}

torch::Tensor OnDiskNpyArray::IndexSelectIOuring(torch::Tensor idx) {
  idx = idx.to(torch::kLong);
  // The minimum page size to contain one feature.
  int64_t align_len = (feature_size_ + ALIGNMENT) & (long)~(ALIGNMENT - 1);
  int64_t num_idx = idx.numel();

  char *read_buffer = (char *)aligned_alloc(
      ALIGNMENT, (align_len + ALIGNMENT) * group_size_ * num_thd_);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size_ * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  // Record the inside offsets of feteched features.
  int64_t residual[group_size_ * num_thd_];

  // Indicator for index error.
  bool error_flag = false;
  torch::parallel_for(0, num_idx, group_size_, [&](int64_t begin, int64_t end) {
    auto thd_id = torch::get_thread_num();
    if (!error_flag) {
      for (int64_t n = begin; n < end; n++) {
        int64_t m = n - begin;
        int64_t i = idx_data[n];  // Feature id.
        if (i >= feat_dim_[0]) {
          error_flag = true;
          break;
        }
        int64_t offset = i * feature_size_ + prefix_len_;
        int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
        residual[thd_id * group_size_ + m] = offset - aligned_offset;

        int64_t read_size;
        if (residual[thd_id * group_size_ + m] + feature_size_ > ALIGNMENT) {
          read_size = align_len + ALIGNMENT;
        } else {
          read_size = align_len;
        }

        // Putting requests into io_uring queue.
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_[thd_id]);
        io_uring_prep_read(
            sqe, feature_fd_,
            read_buffer + ((align_len + ALIGNMENT) * group_size_ * thd_id) +
                ((align_len + ALIGNMENT) * m),
            read_size, aligned_offset);
      }
      if (!error_flag) {
        // Submit I/O requests.
        io_uring_submit(&ring_[thd_id]);

        // Wait for completion of I/O requests.
        int64_t num_finish = 0;
        // Wait until all the disk blocks are loaded in current group.
        while (num_finish < end - begin) {
          struct io_uring_cqe *cqe;
          if (io_uring_wait_cqe(&ring_[thd_id], &cqe) < 0) {
            perror("io_uring_wait_cqe");
            abort();
          }
          struct io_uring_cqe *cqes[group_size_];
          int cqecount =
              io_uring_peek_batch_cqe(&ring_[thd_id], cqes, group_size_);
          if (cqecount == -1) {
            perror("io_uring_peek_batch error\n");
            abort();
          }
          // Move the head pointer of completion queue.
          io_uring_cq_advance(&ring_[thd_id], cqecount);
          num_finish += cqecount;
        }

        // Copy the features in the disk blocks to the result buffer.
        for (int64_t m = 0; m < end - begin; m++) {
          memcpy(
              result_buffer + feature_size_ * (begin + m),
              read_buffer + ((align_len + ALIGNMENT) * group_size_ * thd_id) +
                  ((align_len + ALIGNMENT) * m +
                   residual[thd_id * group_size_ + m]),
              feature_size_);
        }
      }
    }
  });

  auto result = torch::empty({0});
  if (!error_flag) {
    auto options = torch::TensorOptions()
                       .dtype(dtype_)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);

    std::vector<int64_t> shape;
    shape.push_back(num_idx);
    shape.insert(shape.end(), feat_dim_.begin() + 1, feat_dim_.end());
    result = torch::from_blob(result_buffer, torch::IntArrayRef(shape), options)
                 .clone();
  } else {
    throw std::runtime_error("IndexError: Index out of range.");
  }

  free(read_buffer);
  free(result_buffer);

  return result;
}

}  // namespace storage
}  // namespace graphbolt
#endif
