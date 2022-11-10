/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/shared_mem_manager.cc
 * @brief DGL shared mem manager APIs
 */

#ifndef DGL_GRAPH_SHARED_MEM_MANAGER_H_
#define DGL_GRAPH_SHARED_MEM_MANAGER_H_

#include <dgl/array.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>

namespace dgl {

using dgl::runtime::SharedMemory;

const size_t SHARED_MEM_METAINFO_SIZE_MAX = 1024 * 32;

// Utility class to copy objects to shared memory and record metadatas
class SharedMemManager : public dmlc::Stream {
 public:
  explicit SharedMemManager(std::string graph_name, dmlc::Stream* strm)
      : graph_name_(graph_name), strm_(strm) {}

  template <typename T>
  T CopyToSharedMem(const T& data, std::string name);

  template <typename T>
  bool CreateFromSharedMem(T* out_data, std::string name);

  // delegate methods to strm_
  virtual size_t Read(void* ptr, size_t size) { return strm_->Read(ptr, size); }
  virtual void Write(const void* ptr, size_t size) { strm_->Write(ptr, size); }

  using dmlc::Stream::Read;
  using dmlc::Stream::Write;

 private:
  std::string graph_name_;
  dmlc::Stream* strm_;
};

}  // namespace dgl

#endif  // DGL_GRAPH_SHARED_MEM_MANAGER_H_
