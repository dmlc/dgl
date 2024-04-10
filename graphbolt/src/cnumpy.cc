/**
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.cc
 * @brief Numpy File Fetecher class.
 */

#include "cnumpy.h"

#include <torch/torch.h>

#include <cstring>
#include <regex>
#include <stdexcept>

namespace graphbolt {
namespace storage {

static constexpr int kDiskAlignmentSize = 4096;

OnDiskNpyArray::OnDiskNpyArray(
    std::string filename, torch::ScalarType dtype, torch::Tensor shape)
    : filename_(filename), dtype_(dtype) {
#ifdef __linux__
  ParseNumpyHeader(shape);
  file_description_ = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (file_description_ == -1) {
    throw std::runtime_error("npy_load: Unable to open file " + filename);
  }

  // Get system max thread number.
  num_thread_ = torch::get_num_threads();
  io_uring_queue_ = new io_uring[num_thread_];

  // Init io_uring queue.
  for (int64_t t = 0; t < num_thread_; t++) {
    io_uring_queue_init(group_size_, &io_uring_queue_[t], 0);
  }
#endif  // __linux__
}

c10::intrusive_ptr<OnDiskNpyArray> OnDiskNpyArray::Create(
    std::string path, torch::ScalarType dtype, torch::Tensor shape) {
  return c10::make_intrusive<OnDiskNpyArray>(path, dtype, shape);
}

OnDiskNpyArray::~OnDiskNpyArray() {
#ifdef __linux__
  // IO queue exit.
  for (int64_t t = 0; t < num_thread_; t++) {
    io_uring_queue_exit(&io_uring_queue_[t]);
  }
  close(file_description_);
#endif  // __linux__
}

void OnDiskNpyArray::ParseNumpyHeader(torch::Tensor shape) {
#ifdef __linux__
  // Parse numpy file header to get basic info of feature.
  size_t word_size = c10::elementSize(dtype_);
  int64_t num_dim = shape.numel();
  auto shape_ptr = shape.data_ptr<int64_t>();
  for (int64_t d = 0; d < num_dim; d++) {
    feature_dim_.emplace_back(shape_ptr[d]);
  }
  // Compute single feature size.
  signed long feature_length = 1;
  for (size_t i = 1; i < feature_dim_.size(); i++) {
    feature_length *= feature_dim_[i];
  }
  feature_size_ = feature_length * word_size;

  // Get file prefix length.
  std::ifstream file(filename_);
  if (!file.is_open()) {
    throw std::runtime_error(
        "ParseNumpyHeader: Unable to open file " + filename_);
  }
  std::string header;
  std::getline(file, header);
  // Get prefix length for computing feature offset,
  // add one for new-line character.
  prefix_len_ = header.size() + 1;
#endif  // __linux__
}

torch::Tensor OnDiskNpyArray::IndexSelect(torch::Tensor index) {
#ifdef __linux__
  return IndexSelectIOUring(index);
#else
  TORCH_CHECK(false, "OnDiskNpyArray is not supported on non-Linux systems.");
  return torch::empty({0});
#endif  // __linux__
}

#ifdef __linux__
torch::Tensor OnDiskNpyArray::IndexSelectIOUring(torch::Tensor index) {
  index = index.to(torch::kLong);
  // The minimum page size to contain one feature.
  int64_t aligned_length =
      (feature_size_ + kDiskAlignmentSize) & (long)~(kDiskAlignmentSize - 1);
  int64_t num_index = index.numel();

  char *read_buffer = (char *)aligned_alloc(
      kDiskAlignmentSize,
      (aligned_length + kDiskAlignmentSize) * group_size_ * num_thread_);
  char *result_buffer =
      (char *)aligned_alloc(kDiskAlignmentSize, feature_size_ * num_index);

  auto index_data = index.data_ptr<int64_t>();

  // Record the inside offsets of feteched features.
  int64_t residual[group_size_ * num_thread_];

  // Indicator for index error.
  std::atomic<bool> error_flag{};
  TORCH_CHECK(
      num_thread_ >= torch::get_num_threads(),
      "The number of threads can not be changed to larger than the number of "
      "threads when a disk feature fetcher is constructed.");
  torch::parallel_for(
      0, num_index, group_size_, [&](int64_t begin, int64_t end) {
        auto thread_id = torch::get_thread_num();
        if (!error_flag.load()) {
          for (int64_t i = begin; i < end; i++) {
            int64_t group_id = i - begin;
            int64_t feature_id = index_data[i];  // Feature id.
            if (feature_id >= feature_dim_[0]) {
              error_flag.store(true);
              break;
            }
            int64_t offset = feature_id * feature_size_ + prefix_len_;
            int64_t aligned_offset = offset & (long)~(kDiskAlignmentSize - 1);
            residual[thread_id * group_size_ + group_id] =
                offset - aligned_offset;

            int64_t read_size;
            if (residual[thread_id * group_size_ + group_id] + feature_size_ >
                kDiskAlignmentSize) {
              read_size = aligned_length + kDiskAlignmentSize;
            } else {
              read_size = aligned_length;
            }

            // Put requests into io_uring queue.
            struct io_uring_sqe *submit_queue =
                io_uring_get_sqe(&io_uring_queue_[thread_id]);
            io_uring_prep_read(
                submit_queue, file_description_,
                read_buffer +
                    ((aligned_length + kDiskAlignmentSize) * group_size_ *
                     thread_id) +
                    ((aligned_length + kDiskAlignmentSize) * group_id),
                read_size, aligned_offset);
          }
        }
        if (!error_flag.load()) {
          // Submit I/O requests.
          io_uring_submit(&io_uring_queue_[thread_id]);

          // Wait for completion of I/O requests.
          int64_t num_finish = 0;
          // Wait until all the disk blocks are loaded in current group.
          while (num_finish < end - begin) {
            struct io_uring_cqe *complete_queue;
            if (io_uring_wait_cqe(
                    &io_uring_queue_[thread_id], &complete_queue) < 0) {
              perror("io_uring_wait_cqe");
              abort();
            }
            struct io_uring_cqe *complete_queues[group_size_];
            int cqe_count = io_uring_peek_batch_cqe(
                &io_uring_queue_[thread_id], complete_queues, group_size_);
            if (cqe_count == -1) {
              perror("io_uring_peek_batch error\n");
              abort();
            }
            // Move the head pointer of completion queue.
            io_uring_cq_advance(&io_uring_queue_[thread_id], cqe_count);
            num_finish += cqe_count;
          }

          // Copy the features in the disk blocks to the result buffer.
          for (int64_t group_id = 0; group_id < end - begin; group_id++) {
            memcpy(
                result_buffer + feature_size_ * (begin + group_id),
                read_buffer +
                    ((aligned_length + kDiskAlignmentSize) * group_size_ *
                     thread_id) +
                    ((aligned_length + kDiskAlignmentSize) * group_id +
                     residual[thread_id * group_size_ + group_id]),
                feature_size_);
          }
        }
      });

  auto result = torch::empty({0});
  if (!error_flag.load()) {
    auto options = torch::TensorOptions()
                       .dtype(dtype_)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);

    std::vector<int64_t> shape;
    shape.push_back(num_index);
    shape.insert(shape.end(), feature_dim_.begin() + 1, feature_dim_.end());
    result = torch::from_blob(result_buffer, torch::IntArrayRef(shape), options)
                 .clone();
  } else {
    throw std::runtime_error("IndexError: Index out of range.");
  }

  free(read_buffer);
  free(result_buffer);

  return result;
}
#endif  // __linux__

}  // namespace storage
}  // namespace graphbolt
