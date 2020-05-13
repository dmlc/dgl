/*!
 *  Copyright (c) 2020 by Contributors
 * \file rpc/shared_mem_serializer.h
 * \brief headers for serializer.
 */
#ifndef DGL_RPC_SHARED_MEM_SERIALIZER_H_
#define DGL_RPC_SHARED_MEM_SERIALIZER_H_

// #include <dgl/runtime/ndarray.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <dmlc/serializer.h>

#include <queue>
#include <string>
#include <tuple>
#include <vector>

// #include <dgl/array.h>

namespace dgl {

typedef std::pair<void*, int64_t> Ptr_pair;

using dmlc::MemoryStringStream;

class ZeroCopyStream : public MemoryStringStream {
 public:
  //   explicit ZeroCopyStream(std::string* metadata_ptr)
  //       : MemoryStringStream(metadata_ptr) {}

  explicit ZeroCopyStream(std::string* metadata_ptr,
                          std::vector<Ptr_pair>* ptr_list, bool is_local = true)
      : MemoryStringStream(metadata_ptr),
        is_local(is_local),
        _ptr_list(ptr_list),
        idx(0) {}

  void push_buffer(void* data, int64_t data_byte_size) {
    _ptr_list->emplace_back(data, data_byte_size);
  }

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

#endif  // DGL_RPC_SHARED_MEM_SERIALIZER_H_
