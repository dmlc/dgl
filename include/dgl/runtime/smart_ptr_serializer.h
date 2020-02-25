/*!
 *  Copyright (c) 2017 by Contributors
 * \file dgl/runtime/serializer.h
 * \brief Serializer extension to support DGL data types
 *  Include this file to enable serialization of DLDataType, DLContext
 */
#ifndef DGL_RUNTIME_SMART_PTR_SERIALIZER_H_
#define DGL_RUNTIME_SMART_PTR_SERIALIZER_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>

namespace dmlc {
namespace serializer {

//! \cond Doxygen_Suppress
template <typename T>
struct Handler<std::shared_ptr<T>> {
  inline static void Write(Stream *strm, const std::shared_ptr<T> &data) {
    Handler<T>::Write(strm, *data.get());
  }
  inline static bool Read(Stream *strm, std::shared_ptr<T> *data) {
    return Handler<T>::Read(strm, data->get());
  }
};

//! \cond Doxygen_Suppress
template <typename T>
struct Handler<std::vector<std::shared_ptr<T>>> {
  inline static void Write(Stream *strm,
                           const std::vector<std::shared_ptr<T>> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    strm->Write<uint64_t>(sz);
    strm->WriteArray(dmlc::BeginPtr(vec), vec.size());
  }
  inline static bool Read(Stream *strm,
                          std::vector<std::shared_ptr<T>> *out_vec) {
    uint64_t sz;
    if (!strm->Read<uint64_t>(&sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->reserve(size);
    for (size_t i = 0; i < size; i++) {
      out_vec->push_back(std::make_shared<T>());
    }

    return strm->ReadArray(dmlc::BeginPtr(*out_vec), size);
  }
};

template <typename T>
struct Handler<std::unique_ptr<T>> {
  inline static void Write(Stream *strm, const std::unique_ptr<T> &data) {
    Handler<T>::Write(strm, *data.get());
  }
  inline static bool Read(Stream *strm, std::unique_ptr<T> *data) {
    return Handler<T>::Read(strm, data->get());
  }
};

//! \cond Doxygen_Suppress
template <typename T>
struct Handler<std::vector<std::unique_ptr<T>>> {
  inline static void Write(Stream *strm,
                           const std::vector<std::unique_ptr<T>> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    strm->Write<uint64_t>(sz);
    strm->WriteArray(dmlc::BeginPtr(vec), vec.size());
  }
  inline static bool Read(Stream *strm,
                          std::vector<std::unique_ptr<T>> *out_vec) {
    uint64_t sz;
    if (!strm->Read<uint64_t>(&sz)) return false;
    size_t size = static_cast<size_t>(sz);
    out_vec->reserve(size);
    for (size_t i = 0; i < size; i++) {
      out_vec->push_back(std::unique_ptr<T>(new T()));
    }
    return strm->ReadArray(dmlc::BeginPtr(*out_vec), size);
  }
};

}  // namespace serializer
}  // namespace dmlc
#endif  // DGL_RUNTIME_SMART_PTR_SERIALIZER_H_