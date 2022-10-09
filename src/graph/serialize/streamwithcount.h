/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/streamwithcount.h
 * \brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_STREAMWITHCOUNT_H_
#define DGL_GRAPH_SERIALIZE_STREAMWITHCOUNT_H_

#include <dgl/aten/spmat.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <memory>

namespace dgl {
namespace serialize {

/*!
 * \brief StreamWithCount counts the bytes that already written into the
 * underlying stream.
 */
class StreamWithCount : dmlc::Stream {
 public:
  static StreamWithCount *Create(const char *uri, const char *const flag,
                                  bool allow_null, dgl_format_code_t formats) {
    return new StreamWithCount(uri, flag, allow_null, formats);
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

  uint64_t Formats() const { return formats_; }

 private:
  StreamWithCount(const char *uri, const char *const flag, bool allow_null,
                   dgl_format_code_t formats)
      : strm_(dmlc::Stream::Create(uri, flag, allow_null)), formats_(formats) {
  }
  std::unique_ptr<dmlc::Stream> strm_;
  uint64_t count_ = 0;
  dgl_format_code_t formats_ = NONE_CODE;
};
}  // namespace serialize
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZE_STREAMWITHCOUNT_H_
