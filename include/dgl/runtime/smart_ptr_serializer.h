/**
 *  Copyright (c) 2017 by Contributors
 * @file dgl/runtime/serializer.h
 * @brief Serializer extension to support DGL data types
 *  Include this file to enable serialization of DGLDataType, DGLContext
 */
#ifndef DGL_RUNTIME_SMART_PTR_SERIALIZER_H_
#define DGL_RUNTIME_SMART_PTR_SERIALIZER_H_

#include <dgl/graph_serializer.h>
#include <dmlc/io.h>
#include <dmlc/serializer.h>

#include <memory>

namespace dmlc {
namespace serializer {

//! \cond Doxygen_Suppress
template <typename T>
struct Handler<std::shared_ptr<T>> {
  inline static void Write(Stream *strm, const std::shared_ptr<T> &data) {
    Handler<T>::Write(strm, *data.get());
  }
  inline static bool Read(Stream *strm, std::shared_ptr<T> *data) {
    // When read, the default initialization behavior of shared_ptr is
    // shared_ptr<T>(), which is holding a nullptr. Here we need to manually
    // reset to a real object for further loading
    if (!(*data)) {
      data->reset(dgl::Serializer::new_object<T>());
    }
    return Handler<T>::Read(strm, data->get());
  }
};

template <typename T>
struct Handler<std::unique_ptr<T>> {
  inline static void Write(Stream *strm, const std::unique_ptr<T> &data) {
    Handler<T>::Write(strm, *data.get());
  }
  inline static bool Read(Stream *strm, std::unique_ptr<T> *data) {
    // When read, the default initialization behavior of unique_ptr is
    // unique_ptr<T>(), which is holding a nullptr. Here we need to manually
    // reset to a real object for further loading
    if (!(*data)) {
      data->reset(dgl::Serializer::new_object<T>());
    }
    return Handler<T>::Read(strm, data->get());
  }
};

}  // namespace serializer
}  // namespace dmlc
#endif  // DGL_RUNTIME_SMART_PTR_SERIALIZER_H_
