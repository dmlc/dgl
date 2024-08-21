/**
 *  Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.cc
 * @brief Numpy File Fetecher class.
 */

#include "./cnumpy.h"

#include "./io_uring.h"

#ifdef HAVE_LIBRARY_LIBURING
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <graphbolt/async.h>
#include <torch/torch.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "./circular_queue.h"
#include "./utils.h"

namespace graphbolt {
namespace storage {

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

  // The minimum page size to contain one feature.
  aligned_length_ = (feature_size_ + block_size_ - 1) & ~(block_size_ - 1);

  std::call_once(call_once_flag_, [&] {
    // Get system max interop thread count.
    num_queues_ =
        io_uring::num_threads.value_or(torch::get_num_interop_threads());
    TORCH_CHECK(num_queues_ > 0, "A positive # queues is required.");
    io_uring_queue_ = std::unique_ptr<::io_uring[], io_uring_queue_destroyer>(
        new ::io_uring[num_queues_], io_uring_queue_destroyer{num_queues_});
    TORCH_CHECK(num_queues_ <= counting_semaphore_t::max());
    semaphore_.release(num_queues_);
    available_queues_.reserve(num_queues_);
    // Init io_uring queue.
    for (int64_t t = 0; t < num_queues_; t++) {
      available_queues_.push_back(t);
      TORCH_CHECK(
          ::io_uring_queue_init(2 * kGroupSize, &io_uring_queue_[t], 0) == 0);
      // We have allocated 2 * kGroupSize submission queue entries and
      // 4 * kGroupSize completion queue entries after this call.
    }
  });

  num_thread_ = std::min(
      static_cast<int64_t>(num_queues_), num_threads.value_or(num_queues_));
  TORCH_CHECK(num_thread_ > 0, "A positive # threads is required.");

  // We allocate buffers for each existing queue because we might get assigned
  // any queue in range [0, num_queues_).
  read_tensor_ = torch::empty(
      ReadBufferSizePerThread() * num_queues_ + block_size_ - 1,
      torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));
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

class ReadRequest {
 public:
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
torch::Tensor OnDiskNpyArray::IndexSelectIOUringImpl(torch::Tensor index) {
  std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  shape.insert(shape.end(), feature_dim_.begin() + 1, feature_dim_.end());
  auto result = torch::empty(
      shape, index.options()
                 .dtype(dtype_)
                 .layout(torch::kStrided)
                 .pinned_memory(utils::is_pinned(index))
                 .requires_grad(false));
  auto result_buffer = reinterpret_cast<char *>(result.data_ptr());

  // Indicator for index error.
  std::atomic<int> error_flag{};
  std::atomic<int64_t> work_queue{};
  // Construct a QueueAndBufferAcquirer object so that the worker threads can
  // share the available queues and buffers.
  QueueAndBufferAcquirer queue_source(this);
  graphbolt::parallel_for_each_interop(0, num_thread_, 1, [&](int) {
    // The completion queue might contain 4 * kGroupSize while we may submit
    // 4 * kGroupSize more. No harm in overallocation here.
    CircularQueue<ReadRequest> read_queue(8 * kGroupSize);
    int64_t num_submitted = 0;
    int64_t num_completed = 0;
    auto [acquired_queue_handle, read_buffer_source2] = queue_source.get();
    auto &io_uring_queue = acquired_queue_handle.get();
    // Capturing structured binding is available only in C++20, so we rename.
    auto read_buffer_source = read_buffer_source2;
    auto submit_fn = [&](int64_t submission_minimum_batch_size) {
      if (read_queue.Size() < submission_minimum_batch_size) return;
      TORCH_CHECK(  // Check for sqe overflow.
          read_queue.Size() <= 2 * kGroupSize);
      TORCH_CHECK(  // Check for cqe overflow.
          read_queue.Size() + num_submitted - num_completed <= 4 * kGroupSize);
      // Submit and wait for the reads.
      while (!read_queue.IsEmpty()) {
        const auto submitted = ::io_uring_submit(&io_uring_queue);
        TORCH_CHECK(submitted >= 0);
        num_submitted += submitted;
        // Pop the submitted entries from the queue.
        read_queue.PopN(submitted);
      }
    };
    for (int64_t read_buffer_slot = 0; true;) {
      auto request_read_buffer = [&]() {
        return read_buffer_source + (aligned_length_ + block_size_) *
                                        (read_buffer_slot++ % (8 * kGroupSize));
      };
      const auto num_requested_items = std::max(
          std::min(
              // The condition not to overflow the completion queue.
              2 * kGroupSize -
                  (read_queue.Size() + num_submitted - num_completed),
              // The condition not to overflow the submission queue.
              kGroupSize - read_queue.Size()),
          int64_t{});
      const auto begin =
          work_queue.fetch_add(num_requested_items, std::memory_order_relaxed);
      if ((begin >= index.numel() && read_queue.IsEmpty() &&
           num_completed >= num_submitted) ||
          // Even when we encounter out of bounds index (error_flag == 1), we
          // continue. We want to ensure the reads in flight successfully
          // complete to avoid the instability due to incompleted reads.
          error_flag.load(std::memory_order_relaxed) > 1)
        break;
      const auto end = std::min(begin + num_requested_items, index.numel());
      AT_DISPATCH_INDEX_TYPES(
          index.scalar_type(), "IndexSelectIOUring", ([&] {
            auto index_data = index.data_ptr<index_t>();
            for (int64_t i = begin; i < end; ++i) {
              int64_t feature_id = index_data[i];
              if (feature_id < 0) feature_id += feature_dim_[0];
              if (feature_id < 0 || feature_id >= feature_dim_[0]) {
                error_flag.store(1, std::memory_order_relaxed);
                // Simply skip the out of bounds index.
                continue;
              }
              // calculate offset of the feature.
              const int64_t offset = feature_id * feature_size_ + prefix_len_;

              ReadRequest req{
                  result_buffer + feature_size_ * i, feature_size_, offset,
                  block_size_, request_read_buffer()};

              // Put requests into io_uring queue.
              struct io_uring_sqe *sqe = io_uring_get_sqe(&io_uring_queue);
              TORCH_CHECK(sqe);
              io_uring_sqe_set_data(sqe, read_queue.Push(req));
              io_uring_prep_read(
                  sqe, file_description_, req.aligned_read_buffer_,
                  req.AlignedReadSize(), req.AlignedOffset());
              submit_fn(kGroupSize);
            }
          }));

      submit_fn(1);  // Submit all sqes.
      // Wait for the reads; completion queue entries.
      struct io_uring_cqe *cqe;
      TORCH_CHECK(num_submitted - num_completed <= 2 * kGroupSize);
      TORCH_CHECK(
          ::io_uring_wait_cqe_nr(
              &io_uring_queue, &cqe, num_submitted - num_completed) == 0);
      // Check the reads and abort on failure.
      int num_cqes_seen = 0;
      unsigned head;
      io_uring_for_each_cqe(&io_uring_queue, head, cqe) {
        const auto &req =
            *reinterpret_cast<ReadRequest *>(io_uring_cqe_get_data(cqe));
        auto actual_read_len = cqe->res;
        if (actual_read_len < 0) {
          error_flag.store(actual_read_len, std::memory_order_relaxed);
          break;
        }
        const auto remaining_read_len =
            std::max(req.MinimumReadSize() - actual_read_len, int64_t{});
        const auto remaining_useful_read_len =
            std::min(remaining_read_len, req.read_len_);
        const auto useful_read_len = req.read_len_ - remaining_useful_read_len;
        if (remaining_read_len) {
          // Remaining portion will be read as part of the next batch.
          ReadRequest rest{
              req.destination_ + useful_read_len, remaining_useful_read_len,
              req.offset_ + useful_read_len, block_size_,
              request_read_buffer()};
          // Put requests into io_uring queue.
          struct io_uring_sqe *sqe = io_uring_get_sqe(&io_uring_queue);
          TORCH_CHECK(sqe);
          io_uring_sqe_set_data(sqe, read_queue.Push(rest));
          io_uring_prep_read(
              sqe, file_description_, rest.aligned_read_buffer_,
              rest.AlignedReadSize(), rest.AlignedOffset());
          submit_fn(kGroupSize);
        }
        // Copy results into result_buffer.
        std::memcpy(req.destination_, req.ReadBuffer(), useful_read_len);
        num_cqes_seen++;
      }

      // Move the head pointer of completion queue.
      io_uring_cq_advance(&io_uring_queue, num_cqes_seen);
      num_completed += num_cqes_seen;
    }
  });
  const auto ret_val = error_flag.load(std::memory_order_relaxed);
  switch (ret_val) {
    case 0:  // Successful.
      return result;
    case 1:
      throw std::out_of_range("IndexError: Index out of range.");
    default:
      throw std::runtime_error(
          "io_uring error with errno: " + std::to_string(-ret_val));
  }
}

c10::intrusive_ptr<Future<torch::Tensor>> OnDiskNpyArray::IndexSelectIOUring(
    torch::Tensor index) {
  return async([=, this] { return IndexSelectIOUringImpl(index); });
}

#endif  // HAVE_LIBRARY_LIBURING
}  // namespace storage
}  // namespace graphbolt
