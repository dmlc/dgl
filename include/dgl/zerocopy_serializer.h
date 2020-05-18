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

#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <deque>
#include <vector>

#include "dmlc/logging.h"

namespace dgl {

class StringStreamWithBuffer : public dmlc::MemoryStringStream {
 public:
  struct Buffer {
    dgl::runtime::NDArray tensor;
    void* data;
    int64_t size;

    Buffer(const dgl::runtime::NDArray& tensor, void* data, int64_t data_size)
        : tensor(tensor), data(data), size(data_size) {}

    explicit Buffer(void* data) : data(data) {}
  };

  /*!
   * \brief Constructor of the StringStreamWithBuffer. StringStreamWithBuffer is
   * backed up by a string. It stores the data pointer seperately instead of
   * directly write in to the stream. \param metadata_ptr The string to
   * write/load from \param ptr_list Store the zerocopy pointer information for
   * zerocopy write/load \param send_to_remote Means whether the write/load
   * operation is for the same-machine operation. For write scenario, if it's
   * true, the write process will raise error if the NDArray is not in shared
   * memory. For load scenario, if it's true, the NDArray will load from shared
   * memory. If it's false, the NDArray will load from the pointer in ptr_list.
   */
  explicit StringStreamWithBuffer(std::string* metadata_ptr,
                                  bool send_to_remote = true)
      : MemoryStringStream(metadata_ptr),
        buffer_list_(),
        send_to_remote_(send_to_remote),
        head_(0) {}

  StringStreamWithBuffer(std::string* metadata_ptr,
                         std::vector<void*> data_ptr_list)
      : MemoryStringStream(metadata_ptr), send_to_remote_(true) {
    for (void* data : data_ptr_list) {
      buffer_list_.emplace_back(data);
    }
  }

  /*!
   * \brief push pointer in to the ptr_list
   */
  void push_NDArray(const runtime::NDArray& tensor);

  /*!
   * \brief get pointer in to the ptr_list
   */
  dgl::runtime::NDArray pop_NDArray();

  const bool& send_to_remote() const { return send_to_remote_; }

  const std::deque<Buffer>& buffer_list() const { return buffer_list_; }

 private:
  std::deque<Buffer> buffer_list_;
  bool send_to_remote_;
  int64_t head_;
};  // namespace dgl

}  // namespace dgl

#endif  // DGL_ZEROCOPY_SERIALIZER_H_
