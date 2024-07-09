/**
 *  Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.cc
 * @brief Numpy File Fetecher class.
 */

#include "cnumpy.h"

#include <ATen/ParallelFuture.h>
#include <torch/torch.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
namespace graphbolt {
namespace storage {

static constexpr int kDiskAlignmentSize = 4096;

OnDiskNpyArray::OnDiskNpyArray(
    std::string filename, torch::ScalarType dtype, torch::Tensor shape,
    torch::optional<int64_t> num_threads)
    : filename_(filename), dtype_(dtype), group_size_(512) {
#ifndef __linux__
  throw std::runtime_error(
      "OnDiskNpyArray is not supported on non-Linux systems.");
#endif
#ifdef HAVE_LIBRARY_LIBURING
  ParseNumpyHeader(shape);
  file_description_ = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (file_description_ == -1) {
    throw std::runtime_error("npy_load: Unable to open file " + filename);
  }

  // Get system max thread number.
  num_thread_ = torch::get_num_threads();
  if (num_threads.has_value() && num_thread_ > *num_threads) {
    num_thread_ = *num_threads;
  }
  TORCH_CHECK(num_thread_ > 0, "A positive # threads is required.");
  io_uring_queue_ = std::make_unique<io_uring[]>(num_thread_);

  // Init io_uring queue.
  for (int64_t t = 0; t < num_thread_; t++) {
    io_uring_queue_init(group_size_, &io_uring_queue_[t], 0);
  }
#else
  throw std::runtime_error("DiskBasedFeature is not available now.");
#endif  // HAVE_LIBRARY_LIBURING
}

c10::intrusive_ptr<OnDiskNpyArray> OnDiskNpyArray::Create(
    std::string path, torch::ScalarType dtype, torch::Tensor shape,
    torch::optional<int64_t> num_threads) {
  return c10::make_intrusive<OnDiskNpyArray>(path, dtype, shape, num_threads);
}

OnDiskNpyArray::~OnDiskNpyArray() {
#ifdef HAVE_LIBRARY_LIBURING
  // IO queue exit.
  for (int64_t t = 0; t < num_thread_; t++) {
    io_uring_queue_exit(&io_uring_queue_[t]);
  }
  close(file_description_);
#endif  // HAVE_LIBRARY_LIBURING
}

void OnDiskNpyArray::ParseNumpyHeader(torch::Tensor shape) {
#ifdef HAVE_LIBRARY_LIBURING
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
#endif  // HAVE_LIBRARY_LIBURING
}

c10::intrusive_ptr<Future<torch::Tensor>> OnDiskNpyArray::IndexSelect(
    torch::Tensor index) {
#ifdef HAVE_LIBRARY_LIBURING
  return IndexSelectIOUring(index);
#else
  TORCH_CHECK(false, "OnDiskNpyArray is not supported on non-Linux systems.");
  return {};
#endif  // HAVE_LIBRARY_LIBURING
}

#ifdef HAVE_LIBRARY_LIBURING
void OnDiskNpyArray::IndexSelectIOUringImpl(
    torch::Tensor index, torch::Tensor result) {
  const int64_t num_index = index.numel();
  // The minimum page size to contain one feature.
  const int64_t aligned_length = (feature_size_ + kDiskAlignmentSize - 1) &
                                 (long)~(kDiskAlignmentSize - 1);

  const size_t read_buffer_intended_size =
      (aligned_length + kDiskAlignmentSize) * group_size_ * num_thread_;
  size_t read_buffer_size = read_buffer_intended_size + kDiskAlignmentSize - 1;
  auto read_tensor = torch::empty(
      read_buffer_size,
      torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
  auto read_buffer_void_ptr = read_tensor.data_ptr();
  auto read_buffer = reinterpret_cast<char *>(std::align(
      kDiskAlignmentSize, read_buffer_intended_size, read_buffer_void_ptr,
      read_buffer_size));
  TORCH_CHECK(read_buffer, "read_buffer allocation failed!");

  auto result_buffer = reinterpret_cast<char *>(result.data_ptr());

  // Record the inside offsets of feteched features.
  auto residual =
      std::unique_ptr<int64_t[]>(new int64_t[group_size_ * num_thread_]);
  // Indicator for index error.
  std::atomic<bool> error_flag{};
  std::atomic<int64_t> work_queue{};
  torch::parallel_for(0, num_thread_, 1, [&](int64_t begin, int64_t end) {
    const auto thread_id = begin;
    while (true) {
      begin = work_queue.fetch_add(group_size_, std::memory_order_relaxed);
      if (begin >= num_index || error_flag.load()) break;
      end = std::min(begin + group_size_, num_index);
      int64_t batch_offset = begin;
      AT_DISPATCH_INDEX_TYPES(
          index.scalar_type(), "IndexSelectIOUring", ([&] {
            auto index_data = index.data_ptr<index_t>();
            for (int64_t i = begin; i < end; ++i) {
              int64_t group_id = i - begin;
              int64_t feature_id = index_data[i];  // Feature id.
              if (feature_id >= feature_dim_[0]) {
                error_flag.store(true);
                break;
              }
              // calculate offset of the feature.
              int64_t offset = feature_id * feature_size_ + prefix_len_;
              int64_t aligned_offset = offset & (long)~(kDiskAlignmentSize - 1);
              // put offset of the feature into array.
              const auto local_residual_offset =
                  thread_id * group_size_ + (group_id % group_size_);
              residual[local_residual_offset] = offset - aligned_offset;
              // If the tail of the feature extends into another block,
              // read an additional block.
              int64_t read_size;
              if (residual[thread_id * group_size_ + (group_id % group_size_)] +
                      feature_size_ >
                  kDiskAlignmentSize) {
                read_size = aligned_length + kDiskAlignmentSize;
              } else {
                read_size = aligned_length;
              }
              const auto local_read_buffer_offset =
                  (aligned_length + kDiskAlignmentSize) * group_size_ *
                  thread_id;
              // Put requests into io_uring queue.
              struct io_uring_sqe *submit_queue =
                  io_uring_get_sqe(&io_uring_queue_[thread_id]);
              io_uring_prep_read(
                  submit_queue, file_description_,
                  read_buffer + local_read_buffer_offset +
                      ((aligned_length + kDiskAlignmentSize) *
                       (group_id % group_size_)),
                  read_size, aligned_offset);

              if (((group_id + 1) % group_size_ == 0) || i == end - 1) {
                if (!error_flag.load()) {
                  io_uring_submit(&io_uring_queue_[thread_id]);
                  // Wait for completion of I/O requests.
                  int64_t num_finish = 0;
                  // Wait until all the disk blocks are loaded in current
                  // group.
                  while (num_finish < (group_id % group_size_ + 1)) {
                    struct io_uring_cqe *complete_queue;
                    if (io_uring_wait_cqe(
                            &io_uring_queue_[thread_id], &complete_queue) < 0) {
                      perror("io_uring_wait_cqe");
                      std::exit(EXIT_FAILURE);
                    }
                    struct io_uring_cqe *complete_queues[group_size_];
                    int cqe_count = io_uring_peek_batch_cqe(
                        &io_uring_queue_[thread_id], complete_queues,
                        group_size_);
                    if (cqe_count == -1) {
                      perror("io_uring_peek_batch error\n");
                      std::exit(EXIT_FAILURE);
                    }
                    // Move the head pointer of completion queue.
                    io_uring_cq_advance(&io_uring_queue_[thread_id], cqe_count);
                    num_finish += cqe_count;
                  }
                  // Copy results into result_buffer.
                  for (int64_t batch_id = batch_offset; batch_id <= i;
                       batch_id++) {
                    const auto local_id = (batch_id - begin) % group_size_;
                    const auto batch_offset =
                        (aligned_length + kDiskAlignmentSize) * local_id;
                    std::memcpy(
                        result_buffer + feature_size_ * batch_id,
                        read_buffer + local_read_buffer_offset +
                            (batch_offset +
                             residual[thread_id * group_size_ + local_id]),
                        feature_size_);
                  }
                  batch_offset += group_size_;
                }
              }
            }
          }));
    }
  });
  if (error_flag.load()) {
    throw std::runtime_error("IndexError: Index out of range.");
  }

  // return result; the input result parameter is the return value of this func.
}

c10::intrusive_ptr<Future<torch::Tensor>> OnDiskNpyArray::IndexSelectIOUring(
    torch::Tensor index) {
  std::vector<int64_t> shape;
  shape.push_back(index.numel());
  shape.insert(shape.end(), feature_dim_.begin() + 1, feature_dim_.end());
  auto result = torch::empty(
      shape, index.options()
                 .dtype(dtype_)
                 .layout(torch::kStrided)
                 .requires_grad(false));

  auto future = at::intraop_launch_future(
      [=]() { IndexSelectIOUringImpl(index, result); });

  return c10::make_intrusive<Future<torch::Tensor>>(future, result);
}

#endif  // HAVE_LIBRARY_LIBURING
}  // namespace storage
}  // namespace graphbolt
