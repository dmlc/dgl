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

namespace cnpy {

torch::Tensor cnpy::NpyArray::index_select_iouring(torch::Tensor idx) {
  int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

  int64_t feature_size = feature_dim * word_size;
  int64_t num_idx = idx.numel();

  int numthd = 8;
  omp_set_num_threads(numthd);

  int64_t group_size = 512;

  char *read_buffer =
      (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * group_size * numthd);
  char *result_buffer =
      (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

  auto idx_data = idx.data_ptr<int64_t>();

  struct io_uring ring[numthd];
  for (int64_t t = 0; t < numthd; t++) {
    io_uring_queue_init(512, &ring[t], 0);
  }

  int64_t residual[group_size * numthd];

#pragma omp parallel for num_threads(numthd)
  for (int64_t n = 0; n < num_idx; n += group_size) {
    auto thd_id = omp_get_thread_num();
    int64_t r = std::min(num_idx, n + group_size);
    for (int64_t m = 0; m < r - n; m++) {
      int64_t i = idx_data[n + m];
      int64_t offset = i * feature_size + prefix_len;
      int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
      int64_t read_size;
      residual[thd_id * group_size + m] = offset - aligned_offset;

      if (residual[thd_id * group_size + m] + feature_size > ALIGNMENT) {
        read_size = ALIGNMENT * 2;
      } else {
        read_size = ALIGNMENT;
      }

      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring[thd_id]);
      io_uring_prep_read(
          sqe, feature_fd,
          read_buffer + (ALIGNMENT * 2 * group_size * thd_id) +
              (ALIGNMENT * 2 * m),
          read_size, aligned_offset);
    }
    io_uring_submit(&ring[thd_id]);

    int64_t finish_num = 0;
    while (finish_num < r - n) {
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
      io_uring_cq_advance(&ring[thd_id], cqecount);
      finish_num += cqecount;
    }

    for (int64_t m = 0; m < r - n; m++) {
      memcpy(
          result_buffer + feature_size * (n + m),
          read_buffer + (ALIGNMENT * 2 * group_size * thd_id) +
              (ALIGNMENT * 2 * m + residual[thd_id * group_size + m]),
          feature_size);
    }
  }

  for (int64_t t = 0; t < numthd; t++) {
    io_uring_queue_exit(&ring[t]);
  }

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .layout(torch::kStrided)
                     .device(torch::kCPU)
                     .requires_grad(false);
  auto result =
      torch::from_blob(result_buffer, {num_idx, feature_dim}, options).clone();

  free(read_buffer);
  free(result_buffer);
  close(feature_fd);

  return result;
}

// /**
//  * @brief Read disk numpy file based on given index and transform to tensor.
//  */
// torch::Tensor NpyArray::index_select_iouring(torch::Tensor idx) {
//   int feature_fd = open(filename.c_str(), O_RDONLY | O_DIRECT);

//   int64_t feature_size = feature_dim * word_size;
//   int64_t num_idx = idx.numel();

//   // Multi-thread number.
//   int numthd = 8;
//   omp_set_num_threads(numthd);

//   // Each group size for reader.
//   int64_t group_size = 512;
//   char *read_buffer =
//       (char *)aligned_alloc(ALIGNMENT, ALIGNMENT * 4 * group_size);
//   char *result_buffer =
//       (char *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

//   auto idx_data = idx.data_ptr<int64_t>();

//   // Init io_uring fetcher.
//   struct io_uring ring;
//   io_uring_queue_init(1024, &ring, 0);

//   // Two group fetechers for pipeline overlap.
//   // Record the offset of each feteched feature.
//   int64_t residual[group_size * 2];

//   // Record the index interval of each fetcher.
//   // For example, group 0 fetch the feature of [l[0], r[0]) in index array;
//   // group 1 fetch the feature of [l[1], r[1]) in index array.
//   int64_t l[2] = {0, 0}, r[2] = {0, 0}, group = 0;
//   r[group] = std::min(num_idx, l[group] + group_size);

//   // This function computes the read disk block offsets in multi-thread and
//   // saves them as io_uring requests.
//   auto prepare_func = [&] {
// #pragma omp parallel for num_threads(numthd)
//     for (int64_t m = 0; m < r[group] - l[group]; m++) {
//       int64_t i = idx_data[l[group] + m];
//       int64_t offset = i * feature_size + prefix_len;
//       int64_t aligned_offset = offset & (long)~(ALIGNMENT - 1);
//       // Record the offset of feature in disk block.
//       residual[m + group * group_size] = offset - aligned_offset;

//       int64_t read_size;
//       if (residual[m + group * group_size] + feature_size > ALIGNMENT) {
//         read_size = ALIGNMENT * 2;
//       } else {
//         read_size = ALIGNMENT;
//       }

// // Putting requests into io_uring queue must be sequentially.
// #pragma omp critical
//       {
//         struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
//         io_uring_prep_read(
//             sqe, feature_fd,
//             read_buffer + (ALIGNMENT * 2 * m) +
//                 group * ALIGNMENT * 2 * group_size,
//             read_size, aligned_offset);
//       }
//     }
//     // Switch to the next group fetecher.
//     group ^= 1;
//   };

//   // This function wait for completion of I/O requests.
//   auto wait_func = [&] {
//     int64_t finish_num = 0;
//     // Wait until all the disk blocks are loaded in current group.
//     while (finish_num < r[group] - l[group]) {
//       struct io_uring_cqe *cqe;
//       if (io_uring_wait_cqe(&ring, &cqe) < 0) {
//         perror("io_uring_wait_cqe");
//         abort();
//       }
//       struct io_uring_cqe *cqes[group_size];
//       int cqecount = io_uring_peek_batch_cqe(&ring, cqes, group_size);
//       if (cqecount == -1) {
//         perror("io_uring_peek_batch error\n");
//         abort();
//       }
//       // Move the head pointer of completion queue.
//       io_uring_cq_advance(&ring, cqecount);
//       finish_num += cqecount;
//     }
//   };

//   // This function copy the features in the disk blocks to the result buffer.
//   auto copy_func = [&] {
// #pragma omp parallel for num_threads(numthd)
//     for (int64_t m = 0; m < r[group] - l[group]; m++) {
//       memcpy(
//           result_buffer + feature_size * (l[group] + m),
//           read_buffer + group * ALIGNMENT * 2 * group_size +
//               (ALIGNMENT * 2 * m + residual[m + group * group_size]),
//           feature_size);
//     }
//   };

//   prepare_func();
//   l[group] = l[group ^ 1] + group_size;
//   r[group] = std::min(num_idx, l[group] + group_size);
//   // Submit I/O requests.
//   io_uring_submit(&ring);

//   // Use asynchronization to overlap the I/O and compute.
//   for (; l[group] < num_idx;) {
//     // Prepare for the next group data asynchronously.
//     prepare_func();
//     wait_func();
//     io_uring_submit(&ring);
//     // Copy feature asynchronously.
//     copy_func();
//     l[group] = l[group ^ 1] + group_size;
//     r[group] = std::min(num_idx, l[group] + group_size);
//   }
//   group ^= 1;
//   wait_func();
//   copy_func();

//   io_uring_queue_exit(&ring);

//   auto options = torch::TensorOptions()
//                      .dtype(torch::kFloat16)
//                      .layout(torch::kStrided)
//                      .device(torch::kCPU)
//                      .requires_grad(false);
//   auto result =
//       torch::from_blob(result_buffer, {num_idx, feature_dim},
//       options).clone();

//   free(read_buffer);
//   free(result_buffer);
//   close(feature_fd);

//   return result;
// }

/**
 * @brief Parse numpy meta data.
 */
void NpyArray::parse_npy_header(FILE *fp) {
  char buffer[256];
  size_t res = fread(buffer, sizeof(char), 11, fp);
  if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
  std::string header = fgets(buffer, 256, fp);
  assert(header[header.size() - 1] == '\n');
  prefix_len = 11 + header.size();

  size_t loc1, loc2;

  // Get fortran order.
  loc1 = header.find("fortran_order");
  if (loc1 == std::string::npos)
    throw std::runtime_error(
        "parse_npy_header: failed to find header keyword: 'fortran_order'");
  loc1 += 16;
  fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

  // Get shape location.
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

  feature_dim = 1;
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

}  // namespace cnpy
