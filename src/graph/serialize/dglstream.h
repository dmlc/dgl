/**
 *  Copyright (c) 2019 by Contributors
 * @file graph/serialize/dglstream.h
 * @brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_DGLSTREAM_H_
#define DGL_GRAPH_SERIALIZE_DGLSTREAM_H_

#include <dgl/aten/spmat.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <memory>

namespace dgl {
namespace serialize {

/**
 * @brief DGLStream counts the bytes that already written into the
 * underlying stream.
 */
class DGLStream : public dmlc::Stream {
 public:
  /** @brief create a new DGLStream instance */
  static DGLStream *Create(
      const char *uri, const char *const flag, bool allow_null,
      dgl_format_code_t formats) {
    return new DGLStream(uri, flag, allow_null, formats);
  }

  size_t Read(void *ptr, size_t size) override {
    return strm_->Read(ptr, size);
  }

  void Write(const void *ptr, size_t size) override {
    count_ += size;
    strm_->Write(ptr, size);
  }

  using dmlc::Stream::Read;
  using dmlc::Stream::Write;

  bool IsValid() { return strm_.get(); }

  uint64_t Count() const { return count_; }

  uint64_t FormatsToSave() const { return formats_to_save_; }

 private:
  DGLStream(
      const char *uri, const char *const flag, bool allow_null,
      dgl_format_code_t formats)
      : strm_(dmlc::Stream::Create(uri, flag, allow_null)),
        formats_to_save_(formats) {}
  // stream for serialization
  std::unique_ptr<dmlc::Stream> strm_;
  // size of already written to stream
  uint64_t count_ = 0;
  // formats to use when saving graph
  const dgl_format_code_t formats_to_save_ = ANY_CODE;
};
}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZE_DGLSTREAM_H_
