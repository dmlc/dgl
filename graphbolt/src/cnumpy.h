/**
 *  Copyright (c) 2024, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)
 *  Copyright (c) 2023 by Contributors
 * @file cnumpy.h
 * @brief Numpy File Fetecher class.
 */

#ifdef HAVE_LIBRARY_LIBURING
#include <liburing.h>
#endif  // HAVE_LIBRARY_LIBURING

#include <graphbolt/async.h>
#include <torch/script.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda/std/semaphore>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace graphbolt {
namespace storage {

namespace {
#ifdef HAVE_LIBRARY_LIBURING
struct io_uring_queue_destroyer {
  int num_thread_;
  void operator()(::io_uring* queues) {
    if (!queues) return;
    for (int t = 0; t < num_thread_; t++) {
      // IO queue exit.
      ::io_uring_queue_exit(&queues[t]);
    }
    delete[] queues;
  }
};
#endif  // HAVE_LIBRARY_LIBURING
}  // namespace

/**
 * @brief Disk Numpy Fetecher class.
 */
class OnDiskNpyArray : public torch::CustomClassHolder {
  // No user will need more than 1024 io_uring queues.
  using counting_semaphore_t = ::cuda::std::counting_semaphore<1024>;

 public:
  static constexpr int kGroupSize = 256;

  /** @brief Default constructor. */
  OnDiskNpyArray() = default;

  /**
   * @brief Constructor with given file path and data type.
   * @param path Path to the on disk numpy file.
   * @param dtype Data type of numpy array.
   *
   * @return OnDiskNpyArray
   */
  OnDiskNpyArray(
      std::string filename, torch::ScalarType dtype,
      const std::vector<int64_t>& shape, torch::optional<int64_t> num_threads);

  /** @brief Create a disk feature fetcher from numpy file. */
  static c10::intrusive_ptr<OnDiskNpyArray> Create(
      std::string path, torch::ScalarType dtype,
      const std::vector<int64_t>& shape, torch::optional<int64_t> num_threads);

  /** @brief Deconstructor. */
  ~OnDiskNpyArray();

  /**
   * @brief Parses the header of a numpy file to extract feature information.
   **/
  void ParseNumpyHeader();

  /**
   * @brief Read disk numpy file based on given index and transform to
   * tensor.
   */
  c10::intrusive_ptr<Future<torch::Tensor>> IndexSelect(torch::Tensor index);

#ifdef HAVE_LIBRARY_LIBURING
  /**
   * @brief Index-select operation on an on-disk numpy array using IO Uring for
   * asynchronous I/O.
   *
   * This function performs index-select operation on an on-disk numpy array. It
   * uses IO Uring for asynchronous I/O to efficiently read data from disk. The
   * input tensor 'index' specifies the indices of features to select. The
   * function reads features corresponding to the indices from the disk and
   * returns a new tensor containing the selected features.
   *
   * @param index A 1D tensor containing the indices of features to select.
   * @return A tensor containing the selected features.
   * @throws std::runtime_error If index is out of range.
   */
  c10::intrusive_ptr<Future<torch::Tensor>> IndexSelectIOUring(
      torch::Tensor index);

  torch::Tensor IndexSelectIOUringImpl(torch::Tensor index);

#endif  // HAVE_LIBRARY_LIBURING
 private:
  int64_t ReadBufferSizePerThread() const {
    return (aligned_length_ + block_size_) * kGroupSize * 8;
  }

  char* ReadBuffer(int thread_id) const {
    auto read_buffer_void_ptr = read_tensor_.data_ptr();
    size_t read_buffer_size = read_tensor_.numel();
    auto read_buffer = reinterpret_cast<char*>(std::align(
        block_size_, ReadBufferSizePerThread() * num_thread_,
        read_buffer_void_ptr, read_buffer_size));
    TORCH_CHECK(read_buffer, "read_buffer allocation failed!");
    return read_buffer + ReadBufferSizePerThread() * thread_id;
  }

  const std::string filename_;  // Path to numpy file.
  int file_description_;        // File description.
  int64_t block_size_;          // Block size of the opened file.
  int64_t prefix_len_;          // Length of head data in numpy file.
  const std::vector<int64_t>
      feature_dim_;                // Shape of features, e.g. {N,M,K,L}.
  const torch::ScalarType dtype_;  // Feature data type.
  const int64_t feature_size_;     // Number of bytes of feature size.
  int64_t aligned_length_;         // Aligned feature_size.
  int num_thread_;                 // Default thread number.
  torch::Tensor read_tensor_;      // Provides temporary read buffer.

#ifdef HAVE_LIBRARY_LIBURING

  static inline std::once_flag
      call_once_flag_;            // Protect initialization of below.
  static inline int num_queues_;  // Number of queues.
  static inline std::unique_ptr<::io_uring[], io_uring_queue_destroyer>
      io_uring_queue_;  // io_uring queue.
  static inline counting_semaphore_t semaphore_{
      0};  // Control access to the io_uring queues.
  static inline std::mutex available_queues_mtx_;  // available_queues_ mutex.
  static inline std::vector<int> available_queues_;

  /**
   * @brief This class is meant to distribute the available read buffers and the
   * statically declared io_uring queues among the worker threads.
   */
  class QueueAndBufferAcquirer {
   public:
    class UniqueQueue {
     public:
      UniqueQueue(int thread_id) : thread_id_(thread_id) {}
      UniqueQueue(const UniqueQueue&) = delete;
      UniqueQueue& operator=(const UniqueQueue&) = delete;

      /**
       * @brief Returns the queue back to the pool.
       */
      ~UniqueQueue() {
        {
          // We give back the slot we used.
          std::lock_guard lock(available_queues_mtx_);
          available_queues_.push_back(thread_id_);
        }
        semaphore_.release();
      }

      /**
       * @brief Returns the raw io_uring queue.
       */
      ::io_uring& get() const { return io_uring_queue_[thread_id_]; }

     private:
      int thread_id_;
    };

    QueueAndBufferAcquirer(OnDiskNpyArray* array) : array_(array) {
      semaphore_.acquire();
    }

    ~QueueAndBufferAcquirer() {
      // If none of the worker threads acquire the semaphore, we make sure to
      // release the ticket taken in the constructor.
      if (!entering_first_.test_and_set(std::memory_order_relaxed)) {
        semaphore_.release();
      }
    }

    /**
     * @brief Returns the secured io_uring queue and the read buffer as a pair.
     * The raw io_uring queue can be accessed by calling `.get()` on the
     * returned UniqueQueue object.
     *
     * @note The returned UniqueQueue object manages the lifetime of the
     * io_uring queue. Its destructor returns the queue back to the pool.
     */
    std::pair<UniqueQueue, char*> get() {
      // We consume a slot from the semaphore to use a queue.
      if (entering_first_.test_and_set(std::memory_order_relaxed)) {
        semaphore_.acquire();
      }
      const auto thread_id = [&] {
        std::lock_guard lock(available_queues_mtx_);
        TORCH_CHECK(!available_queues_.empty());
        const auto thread_id = available_queues_.back();
        available_queues_.pop_back();
        return thread_id;
      }();
      return {
          std::piecewise_construct, std::make_tuple(thread_id),
          std::make_tuple(array_->ReadBuffer(thread_id))};
    }

   private:
    const OnDiskNpyArray* array_;
    std::atomic_flag entering_first_ = ATOMIC_FLAG_INIT;
  };

#endif  // HAVE_LIBRARY_LIBURING
};

}  // namespace storage
}  // namespace graphbolt
