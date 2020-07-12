/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/streamwithcount.h
 * \brief Graph serialization header
 */
#ifndef DGL_GRAPH_SERIALIZE_STREAMWITHCOUNT_H_
#define DGL_GRAPH_SERIALIZE_STREAMWITHCOUNT_H_

#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <memory>

/*!
 * \brief StreamWithCount counts the bytes that already written into the
 * underlying stream.
 */
class StreamWithCount : dmlc::Stream {
 public:
  static StreamWithCount *Create(const char *uri, const char *const flag,
                                 bool allow_null = false) {
    return new StreamWithCount(uri, flag, allow_null);
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

 private:
  StreamWithCount(const char *uri, const char *const flag, bool allow_null)
      : strm_(dmlc::Stream::Create(uri, flag, allow_null)) {}
  std::unique_ptr<dmlc::Stream> strm_;
  uint64_t count_ = 0;
};

#endif  // DGL_GRAPH_SERIALIZE_STREAMWITHCOUNT_H_
