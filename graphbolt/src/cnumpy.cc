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
static constexpr int kGroupSize = 512;

OnDiskNpyArray::OnDiskNpyArray(
    std::string filename, torch::ScalarType dtype, torch::Tensor shape,
    torch::optional<int64_t> num_threads)
    : filename_(filename), dtype_(dtype) {
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
    io_uring_queue_init(kGroupSize, &io_uring_queue_[t], 0);
  }

  // The minimum page size to contain one feature.
  aligned_length_ = (feature_size_ + kDiskAlignmentSize - 1) &
                    (long)~(kDiskAlignmentSize - 1);

  const size_t read_buffer_intended_size =
      (aligned_length_ + kDiskAlignmentSize) * kGroupSize * num_thread_;
  size_t read_buffer_size = read_buffer_intended_size + kDiskAlignmentSize - 1;
  read_tensor_ = torch::empty(
      read_buffer_size,
      torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
  auto read_buffer_void_ptr = read_tensor_.data_ptr();
  read_buffer_ = reinterpret_cast<char *>(std::align(
      kDiskAlignmentSize, read_buffer_intended_size, read_buffer_void_ptr,
      read_buffer_size));
  TORCH_CHECK(read_buffer_, "read_buffer allocation failed!");
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
torch::Tensor OnDiskNpyArray::IndexSelectIOUringImpl(torch::Tensor index) {
  std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  shape.insert(shape.end(), feature_dim_.begin() + 1, feature_dim_.end());
  auto result = torch::empty(
      shape, index.options()
                 .dtype(dtype_)
                 .layout(torch::kStrided)
                 .pinned_memory(index.is_pinned())
                 .requires_grad(false));
  auto result_buffer = reinterpret_cast<char *>(result.data_ptr());

  // Indicator for index error.
  std::atomic<bool> error_flag{};
  std::atomic<int64_t> work_queue{};
  torch::parallel_for(0, num_thread_, 1, [&](int64_t begin, int64_t end) {
    const auto thread_id = begin;
    // Record the inside offsets of fetched features.
    int64_t residual[kGroupSize];
    while (true) {
      begin = work_queue.fetch_add(kGroupSize, std::memory_order_relaxed);
      if (begin >= index.numel() || error_flag.load()) break;
      end = std::min(begin + kGroupSize, index.numel());
      AT_DISPATCH_INDEX_TYPES(
          index.scalar_type(), "IndexSelectIOUring", ([&] {
            auto index_data = index.data_ptr<index_t>();
            for (int64_t i = begin; i < end; ++i) {
              int64_t local_id = i - begin;
              int64_t feature_id = index_data[i];  // Feature id.
              if (feature_id >= feature_dim_[0]) {
                error_flag.store(true);
                break;
              }
              // calculate offset of the feature.
              int64_t offset = feature_id * feature_size_ + prefix_len_;
              int64_t aligned_offset = offset & (long)~(kDiskAlignmentSize - 1);
              // put offset of the feature into array.
              residual[local_id] = offset - aligned_offset;
              // If the tail of the feature extends into another block,
              // read an additional block.
              int64_t read_size;
              if (residual[local_id] + feature_size_ > kDiskAlignmentSize) {
                read_size = aligned_length_ + kDiskAlignmentSize;
              } else {
                read_size = aligned_length_;
              }
              const auto local_read_buffer_offset =
                  (aligned_length_ + kDiskAlignmentSize) * kGroupSize *
                  thread_id;
              // Put requests into io_uring queue.
              struct io_uring_sqe *submit_queue =
                  io_uring_get_sqe(&io_uring_queue_[thread_id]);
              io_uring_prep_read(
                  submit_queue, file_description_,
                  read_buffer_ + local_read_buffer_offset +
                      ((aligned_length_ + kDiskAlignmentSize) * local_id),
                  read_size, aligned_offset);

              if (i + 1 == end && !error_flag.load()) {
                io_uring_submit(&io_uring_queue_[thread_id]);
                // Wait for completion of I/O requests.
                struct io_uring_cqe *complete_queues[kGroupSize];
                // Wait for submitted end - begin reads to finish.
                if (io_uring_wait_cqe_nr(
                        &io_uring_queue_[thread_id], complete_queues,
                        end - begin) < 0) {
                  perror("io_uring_wait_cqe_nr error\n");
                  std::exit(EXIT_FAILURE);
                }
                // Move the head pointer of completion queue.
                io_uring_cq_advance(&io_uring_queue_[thread_id], end - begin);

                // Copy results into result_buffer.
                for (int64_t j = begin; j < end; j++) {
                  const auto local_id = j - begin;
                  const auto batch_offset =
                      (aligned_length_ + kDiskAlignmentSize) * local_id;
                  std::memcpy(
                      result_buffer + feature_size_ * j,
                      read_buffer_ + local_read_buffer_offset +
                          (batch_offset + residual[local_id]),
                      feature_size_);
                }
              }
            }
          }));
    }
  });
  if (error_flag.load()) {
    throw std::runtime_error("IndexError: Index out of range.");
  }

  return result;
}

c10::intrusive_ptr<Future<torch::Tensor>> OnDiskNpyArray::IndexSelectIOUring(
    torch::Tensor index) {
  return async([=] { return IndexSelectIOUringImpl(index); });
}

#endif  // HAVE_LIBRARY_LIBURING
}  // namespace storage
}  // namespace graphbolt
