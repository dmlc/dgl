/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/shared_mem_serializer.h
 * \brief headers for serializer.
 */
#ifndef DGL_ZEROCOPY_SERIALIZER_H_
#define DGL_ZEROCOPY_SERIALIZER_H_

// #include <dgl/runtime/ndarray.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/serializer.h>

#include <queue>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

// #include <dgl/array.h>

namespace dgl {

typedef std::pair<void*, int64_t> Ptr_pair;

using dmlc::MemoryStringStream;


class ZeroCopyStream : public MemoryStringStream {
 public:
 
  /*!
   * \brief constructor of the ZeroCopyStream. ZeroCopyStream is backed up by a string.
   * It stores the data pointer seperately instead of directly write in to the stream.
   * \param metadata_ptr The string to write/load from
   * \param ptr_list Store the zerocopy pointer information for zerocopy write/load
   * \param is_local Means whether the write/load operation is for the same-machine operation.
   *  For write scenario, if it's true, the write process will raise error
   *  if the NDArray is not in shared memory. For load scenario, if it's true, the NDArray will
   *  load from shared memory. If it's false, the NDArray will load from the pointer in ptr_list.
   */
  explicit ZeroCopyStream(std::string* metadata_ptr,
                          std::vector<Ptr_pair>* ptr_list, bool is_local = true)
      : MemoryStringStream(metadata_ptr),
        is_local(is_local),
        _ptr_list(ptr_list),
        idx(0) {}
  /*!
   * \brief push pointer in to the ptr_list
   */
  void push_buffer(void* data, int64_t data_byte_size) {
    _ptr_list->emplace_back(data, data_byte_size);
  }

  
  /*!
   * \brief get pointer in to the ptr_list
   */
  Ptr_pair pop_buffer() {
    auto ret = _ptr_list->at(idx);
    idx++;
    return ret;
  }

  bool is_local;

 private:
  std::vector<Ptr_pair>* _ptr_list;
  int64_t idx;
};

}  // namespace dgl

#endif  // DGL_ZEROCOPY_SERIALIZER_H_
