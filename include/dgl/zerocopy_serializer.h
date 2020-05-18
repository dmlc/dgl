/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/shared_mem_serializer.h
 * \brief headers for serializer.
 */
#ifndef DGL_ZEROCOPY_SERIALIZER_H_
#define DGL_ZEROCOPY_SERIALIZER_H_

#include <dgl/runtime/ndarray.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/serializer.h>

#include <deque>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dmlc/logging.h"

namespace dgl {

// StringStreamWithBuffer is backed up by a string. It stores the data pointer
// seperately instead of directly write in to the stream.
class StringStreamWithBuffer : public dmlc::MemoryStringStream {
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

  /*!
   * \brief This constructor is for writing scenario or reading from local
   * machine
   * \param metadata_ptr The string to write/load from zerocopy write/load
   * \param send_to_remote Whether this stream will be deserialized at remote
   * machine or the local machine. If true, will record the data pointer into
   * buffer list.
   *
   * For example:
   * std::string blob;
   * // Write to send to local
   * StringStreamWithBuffer buf_strm(&blob, false)
   * // Write to send to remote
   * StringStreamWithBuffer buf_strm(&blob, true)
   * // Or
   * StringStreamWithBuffer buf_strm(&blob)
   * // Read from local
   * StringStreamWithBuffer buf_strm(&blob, false)
   */
  explicit StringStreamWithBuffer(std::string* metadata_ptr,
                                  bool send_to_remote = true)
      : MemoryStringStream(metadata_ptr),
        buffer_list_(),
        send_to_remote_(send_to_remote) {}
  /*!
   * \brief This constructor is for reading from remote
   * \param metadata_ptr The string to write/load from zerocopy write/load
   * \param data_ptr_list list of pointer to reconstruct NDArray
   *
   * For example:
   * std::string blob;
   * std::vector<void*> data_ptr_list;
   * // Read from remote sended pointer list
   * StringStreamWithBuffer buf_strm(&blob, data_ptr_list)
   */
  StringStreamWithBuffer(std::string* metadata_ptr,
                         const std::vector<void*>& data_ptr_list)
      : MemoryStringStream(metadata_ptr), send_to_remote_(true) {
    for (void* data : data_ptr_list) {
      buffer_list_.emplace_back(data);
    }
  }

  /*!
   * \brief push NDArray into stream
   * If send_to_remote=true, the NDArray will be saved to the buffer list
   * If send_to_remote=false, the NDArray will be saved to the backedup string
   */
  void PushNDArray(const runtime::NDArray& tensor);

  /*!
   * \brief pop NDArray from stream
   * If send_to_remote=true, the NDArray will be reconstructed from buffer list
   * If send_to_remote=false, the NDArray will be reconstructed from shared
   * memory
   */
  dgl::runtime::NDArray PopNDArray();

  /*!
   * \brief Get whether this stream is for remote usage
   */
  bool send_to_remote() { return send_to_remote_; }

  /*!
   * \brief Get underlying buffer list
   */
  const std::deque<Buffer>& buffer_list() const { return buffer_list_; }

 private:
  std::deque<Buffer> buffer_list_;
  bool send_to_remote_;
};  // namespace dgl

}  // namespace dgl

#endif  // DGL_ZEROCOPY_SERIALIZER_H_
