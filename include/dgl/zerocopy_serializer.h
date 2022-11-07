/**
 *  Copyright (c) 2020 by Contributors
 * @file rpc/shared_mem_serializer.h
 * @brief headers for serializer.
 */
#ifndef DGL_ZEROCOPY_SERIALIZER_H_
#define DGL_ZEROCOPY_SERIALIZER_H_

#include <dgl/runtime/ndarray.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/serializer.h>

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dmlc/logging.h"

namespace dgl {

/**
 *
 * StreamWithBuffer is backed up by dmlc::MemoryFixedSizeStream or
 * dmlc::MemoryStringStream. This class supports serializing and deserializing
 * NDArrays stored in shared memory. If the stream is created for
 * sending/recving data through network, the data pointer of the NDArray will be
 * transmitted directly without and copy. Otherwise, the stream is for
 * sending/recving data to another process on the same machine, so if an NDArray
 * is stored in shared memory, it will just record the shared memory name
 * instead of the actual data buffer.
 *
 * For example:
 *
 * std::string blob;
 * // Send to local
 * StreamWithBuffer strm(&blob, false);
 * // Send to remote
 * StreamWithBuffer strm(&blob, true);
 * // Receive from local
 * StreamWithBuffer strm(&blob, false);
 * // Receive from remote
 * std::vector<void*> ptr_list
 * StreamWithBuffer strm(&blob, ptr_list);
 */
class StreamWithBuffer : public dmlc::SeekStream {
 public:
  // Buffer type. Storing NDArray to maintain the reference counting to ensure
  // the liveness of data pointer
  struct Buffer {
    dgl::runtime::NDArray tensor = dgl::runtime::NDArray();
    void* data = nullptr;
    int64_t size = 0;

    Buffer(const dgl::runtime::NDArray& tensor, void* data, int64_t data_size)
        : tensor(tensor), data(data), size(data_size) {}

    explicit Buffer(void* data) : data(data) {}
  };

  /**
   * @brief This constructor is for writing scenario or reading from local
   * machine
   * @param strm The backup stream to write/load from
   * @param send_to_remote Whether this stream will be deserialized at remote
   * machine or the local machine. If true, will record the data pointer into
   * buffer list.
   */
  StreamWithBuffer(std::unique_ptr<dmlc::SeekStream> strm, bool send_to_remote)
      : strm_(std::move(strm)),
        buffer_list_(),
        send_to_remote_(send_to_remote) {}
  /**
   * @brief This constructor is for reading from remote
   * @param strm The stream to write/load from zerocopy write/load
   * @param data_ptr_list list of pointer to reconstruct NDArray
   *
   * For example:
   * std::string blob;
   * std::vector<void*> data_ptr_list;
   * // Read from remote sended pointer list
   * StreamWithBuffer buf_strm(&blob, data_ptr_list)
   */
  StreamWithBuffer(
      std::unique_ptr<dmlc::SeekStream> strm,
      const std::vector<void*>& data_ptr_list)
      : strm_(std::move(strm)), send_to_remote_(true) {
    for (void* data : data_ptr_list) {
      buffer_list_.emplace_back(data);
    }
  }

  /**
   * @brief Construct stream backed up by string
   * @param blob The string to write/load from zerocopy write/load
   * @param send_to_remote Whether this stream will be deserialized at remote
   * machine or the local machine. If true, will record the data pointer into
   * buffer list.
   */
  StreamWithBuffer(std::string* blob, bool send_to_remote)
      : strm_(new dmlc::MemoryStringStream(blob)),
        send_to_remote_(send_to_remote) {}

  /**
   * @brief Construct stream backed up by string
   * @param p_buffer buffer pointer
   * @param size buffer size
   * @param send_to_remote Whether this stream will be deserialized at remote
   * machine or the local machine. If true, will record the data pointer into
   * buffer list.
   */
  StreamWithBuffer(char* p_buffer, size_t size, bool send_to_remote)
      : strm_(new dmlc::MemoryFixedSizeStream(p_buffer, size)),
        send_to_remote_(send_to_remote) {}

  /**
   * @brief Construct stream backed up by string, and reconstruct NDArray
   * from data_ptr_list
   * @param blob The string to write/load from zerocopy write/load
   * @param data_ptr_list pointer list for NDArrays to deconstruct from
   */
  StreamWithBuffer(std::string* blob, const std::vector<void*>& data_ptr_list)
      : strm_(new dmlc::MemoryStringStream(blob)), send_to_remote_(true) {
    for (void* data : data_ptr_list) {
      buffer_list_.emplace_back(data);
    }
  }

  /**
   * @brief Construct stream backed up by string, and reconstruct NDArray
   * from data_ptr_list
   * @param p_buffer buffer pointer
   * @param size buffer size
   * @param data_ptr_list pointer list for NDArrays to deconstruct from
   */
  StreamWithBuffer(
      char* p_buffer, size_t size, const std::vector<void*>& data_ptr_list)
      : strm_(new dmlc::MemoryFixedSizeStream(p_buffer, size)),
        send_to_remote_(true) {
    for (void* data : data_ptr_list) {
      buffer_list_.emplace_back(data);
    }
  }

  // delegate methods to strm_
  virtual size_t Read(void* ptr, size_t size) { return strm_->Read(ptr, size); }
  virtual void Write(const void* ptr, size_t size) { strm_->Write(ptr, size); }
  virtual void Seek(size_t pos) { strm_->Seek(pos); }
  virtual size_t Tell(void) { return strm_->Tell(); }

  using dmlc::Stream::Read;
  using dmlc::Stream::Write;

  /**
   * @brief push NDArray into stream
   * If send_to_remote=true, the NDArray will be saved to the buffer list
   * If send_to_remote=false, the NDArray will be saved to the backedup string
   */
  void PushNDArray(const runtime::NDArray& tensor);

  /**
   * @brief pop NDArray from stream
   * If send_to_remote=true, the NDArray will be reconstructed from buffer list
   * If send_to_remote=false, the NDArray will be reconstructed from shared
   * memory
   */
  dgl::runtime::NDArray PopNDArray();

  /**
   * @brief Get whether this stream is for remote usage
   */
  bool send_to_remote() { return send_to_remote_; }

  /**
   * @brief Get underlying buffer list
   */
  const std::deque<Buffer>& buffer_list() const { return buffer_list_; }

 private:
  std::unique_ptr<dmlc::SeekStream> strm_;
  std::deque<Buffer> buffer_list_;
  bool send_to_remote_;
};  // namespace dgl

}  // namespace dgl

#endif  // DGL_ZEROCOPY_SERIALIZER_H_
