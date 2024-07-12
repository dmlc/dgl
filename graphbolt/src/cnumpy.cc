/**
 *  Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.cc
 * @brief Numpy File Fetecher class.
 */

#include "./cnumpy.h"

#ifdef HAVE_LIBRARY_LIBURING
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <ATen/ParallelFuture.h>
#include <torch/torch.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "./circular_queue.h"

namespace graphbolt {
namespace storage {

static constexpr int kGroupSize = 512;

OnDiskNpyArray::OnDiskNpyArray(
    std::string filename, torch::ScalarType dtype,
    const std::vector<int64_t> &shape, torch::optional<int64_t> num_threads)
    : filename_(filename),
      feature_dim_(shape),
      dtype_(dtype),
      feature_size_(std::accumulate(
          shape.begin() + 1, shape.end(), c10::elementSize(dtype),
          std::multiplies<int64_t>())) {
#ifndef __linux__
  throw std::runtime_error(
      "OnDiskNpyArray is not supported on non-Linux systems.");
#endif
#ifdef HAVE_LIBRARY_LIBURING
  ParseNumpyHeader();
  file_description_ = ::open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (file_description_ < 0) {
    throw std::runtime_error("npy_load: Unable to open file " + filename);
  }
  struct stat st;
  TORCH_CHECK(::fstat(file_description_, &st) == 0);
  const auto file_size = st.st_size;
  block_size_ = st.st_blksize;
  TORCH_CHECK(file_size - prefix_len_ >= feature_dim_[0] * feature_size_);

  // Get system max thread number.
  num_thread_ = torch::get_num_threads();
  if (num_threads.has_value() && num_thread_ > *num_threads) {
    num_thread_ = *num_threads;
  }
  TORCH_CHECK(num_thread_ > 0, "A positive # threads is required.");
  io_uring_queue_ = std::make_unique<io_uring[]>(num_thread_);

  // Init io_uring queue.
  for (int64_t t = 0; t < num_thread_; t++) {
    TORCH_CHECK(::io_uring_queue_init(kGroupSize, &io_uring_queue_[t], 0) == 0);
  }

  // The minimum page size to contain one feature.
  aligned_length_ = (feature_size_ + block_size_ - 1) & ~(block_size_ - 1);

  const size_t read_buffer_intended_size =
      (aligned_length_ + block_size_) * kGroupSize * num_thread_ * 2;
  size_t read_buffer_size = read_buffer_intended_size + block_size_ - 1;
  read_tensor_ = torch::empty(
      read_buffer_size,
      torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
  auto read_buffer_void_ptr = read_tensor_.data_ptr();
  read_buffer_ = reinterpret_cast<char *>(std::align(
      block_size_, read_buffer_intended_size, read_buffer_void_ptr,
      read_buffer_size));
  TORCH_CHECK(read_buffer_, "read_buffer allocation failed!");
#else
  throw std::runtime_error("DiskBasedFeature is not available now.");
#endif  // HAVE_LIBRARY_LIBURING
}

c10::intrusive_ptr<OnDiskNpyArray> OnDiskNpyArray::Create(
    std::string path, torch::ScalarType dtype,
    const std::vector<int64_t> &shape, torch::optional<int64_t> num_threads) {
  return c10::make_intrusive<OnDiskNpyArray>(path, dtype, shape, num_threads);
}

OnDiskNpyArray::~OnDiskNpyArray() {
#ifdef HAVE_LIBRARY_LIBURING
  // IO queue exit.
  for (int64_t t = 0; t < num_thread_; t++) {
    ::io_uring_queue_exit(&io_uring_queue_[t]);
  }
  TORCH_CHECK(::close(file_description_) == 0);
#endif  // HAVE_LIBRARY_LIBURING
}

void OnDiskNpyArray::ParseNumpyHeader() {
  // Parse numpy file header to get basic info of feature.
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

struct ReadRequest {
  char *destination_;
  int64_t read_len_;
  int64_t offset_;
  int64_t block_size_;
  char *aligned_read_buffer_;

  auto AlignedOffset() const { return offset_ & ~(block_size_ - 1); }

  auto ReadBuffer() const {
    return aligned_read_buffer_ + offset_ - AlignedOffset();
  }

  auto AlignedReadSize() const {
    const int64_t end_offset = offset_ + read_len_;
    const int64_t aligned_end_offset =
        (end_offset + block_size_ - 1) & ~(block_size_ - 1);
    return aligned_end_offset - AlignedOffset();
  }

  auto MinimumReadSize() const { return offset_ + read_len_ - AlignedOffset(); }
};

#ifdef HAVE_LIBRARY_LIBURING
void OnDiskNpyArray::IndexSelectIOUringImpl(
    torch::Tensor index, torch::Tensor result) {
  auto result_buffer = reinterpret_cast<char *>(result.data_ptr());

  // Indicator for index error.
  std::atomic<int> error_flag{};
  std::atomic<int64_t> work_queue{};
  torch::parallel_for(0, num_thread_, 1, [&](int64_t begin, int64_t end) {
    if (begin >= end) return;
    const auto thread_id = begin;
    auto &my_io_uring_queue = io_uring_queue_[thread_id];
    auto my_read_buffer = read_buffer_ + (aligned_length_ + block_size_) *
                                             kGroupSize * thread_id * 2;
    CircularQueue<ReadRequest> read_queue(2 * kGroupSize);
    int64_t num_submitted = 0;
    int64_t num_completed = 0;
    for (int64_t read_buffer_slot = 0; true;) {
      const auto num_requested_items =
          std::max(kGroupSize - read_queue.Size(), static_cast<int64_t>(0));
      begin =
          work_queue.fetch_add(num_requested_items, std::memory_order_relaxed);
      if ((begin >= index.numel() && read_queue.IsEmpty() &&
           num_completed >= num_submitted) ||
          error_flag.load(std::memory_order_relaxed))
        break;
      end = std::min(begin + num_requested_items, index.numel());
      AT_DISPATCH_INDEX_TYPES(
          index.scalar_type(), "IndexSelectIOUring", ([&] {
            auto index_data = index.data_ptr<index_t>();
            for (int64_t i = begin; i < end; ++i) {
              int64_t feature_id = index_data[i];
              if (feature_id < 0) feature_id += feature_dim_[0];
              if (feature_id < 0 || feature_id >= feature_dim_[0]) {
                error_flag.store(1, std::memory_order_relaxed);
                continue;
              }
              // calculate offset of the feature.
              const int64_t offset = feature_id * feature_size_ + prefix_len_;

              ReadRequest req{
                  result_buffer + feature_size_ * i, feature_size_, offset,
                  block_size_,
                  my_read_buffer + (aligned_length_ + block_size_) *
                                       (read_buffer_slot++ % (2 * kGroupSize))};

              // Put requests into io_uring queue.
              struct io_uring_sqe *sqe = io_uring_get_sqe(&my_io_uring_queue);
              TORCH_CHECK(sqe);
              io_uring_sqe_set_data(sqe, read_queue.Push(req));
              io_uring_prep_read(
                  sqe, file_description_, req.aligned_read_buffer_,
                  req.AlignedReadSize(), req.AlignedOffset());
            }
          }));

      if (!error_flag.load(std::memory_order_relaxed)) {
        TORCH_CHECK(read_queue.Size() <= kGroupSize);
        // Submit and wait for the reads.
        for (auto submitting = read_queue.Size(); submitting;) {
          const auto submitted = ::io_uring_submit(&my_io_uring_queue);
          TORCH_CHECK(submitted >= 0);
          submitting -= submitted;
          num_submitted += submitted;
        }
        // Clear the queue as we submitted all entries.
        read_queue.Clear();
        // Wait for the reads.
        struct io_uring_cqe *cqe;
        TORCH_CHECK(num_submitted - num_completed <= 2 * kGroupSize);
        TORCH_CHECK(
            ::io_uring_wait_cqe_nr(
                &my_io_uring_queue, &cqe, num_submitted - num_completed) == 0);
        // Check the reads and abort on failure.
        int64_t i = 0;  // Counts how many were completed in this batch.
        unsigned head;
        io_uring_for_each_cqe(&my_io_uring_queue, head, cqe) {
          const auto &req =
              *reinterpret_cast<ReadRequest *>(io_uring_cqe_get_data(cqe));
          auto actual_read_len = cqe->res;
          if (actual_read_len < 0) {
            error_flag.store(3, std::memory_order_relaxed);
            break;
          }
          const auto remaining_read_len =
              std::max(req.MinimumReadSize() - actual_read_len, int64_t{});
          const auto remaining_useful_read_len =
              std::min(remaining_read_len, req.read_len_);
          const auto useful_read_len =
              req.read_len_ - remaining_useful_read_len;
          if (remaining_read_len) {
            // Remaining portion will be read as part of the next batch.
            ReadRequest rest{
                req.destination_ + useful_read_len, remaining_useful_read_len,
                req.offset_ + useful_read_len, block_size_,
                my_read_buffer + (aligned_length_ + block_size_) *
                                     (read_buffer_slot++ % (2 * kGroupSize))};
            // Put requests into io_uring queue.
            struct io_uring_sqe *sqe = io_uring_get_sqe(&my_io_uring_queue);
            TORCH_CHECK(sqe);
            io_uring_sqe_set_data(sqe, read_queue.Push(rest));
            io_uring_prep_read(
                sqe, file_description_, rest.aligned_read_buffer_,
                rest.AlignedReadSize(), rest.AlignedOffset());
          }
          // Copy results into result_buffer.
          std::memcpy(req.destination_, req.ReadBuffer(), useful_read_len);
          i++;
        }

        // Move the head pointer of completion queue.
        io_uring_cq_advance(&my_io_uring_queue, i);
        num_completed += i;
      }
    }
  });
  // return result; the input result parameter is the return value of this func.
  switch (error_flag.load(std::memory_order_relaxed)) {
    case 0:  // Successful.
      return;
    case 1:
      throw std::out_of_range("IndexError: Index out of range.");
    default:
      throw std::runtime_error("io_uring error!");
  }
}

c10::intrusive_ptr<Future<torch::Tensor>> OnDiskNpyArray::IndexSelectIOUring(
    torch::Tensor index) {
  std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  shape.insert(shape.end(), feature_dim_.begin() + 1, feature_dim_.end());
  auto result = torch::empty(
      shape, index.options()
                 .dtype(dtype_)
                 .layout(torch::kStrided)
                 .pinned_memory(index.is_pinned())
                 .requires_grad(false));

  auto future = at::intraop_launch_future(
      [=]() { IndexSelectIOUringImpl(index, result); });

  return c10::make_intrusive<Future<torch::Tensor>>(future, result);
}

#endif  // HAVE_LIBRARY_LIBURING
}  // namespace storage
}  // namespace graphbolt
