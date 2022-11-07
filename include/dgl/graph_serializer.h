/**
 *  Copyright (c) 2018 by Contributors
 * @file graph/graph_serializer.cc
 * @brief DGL serializer APIs
 */

#ifndef DGL_GRAPH_SERIALIZER_H_
#define DGL_GRAPH_SERIALIZER_H_

#include <memory>
namespace dgl {

// Util class to call the private/public empty constructor, which is needed for
// serialization
class Serializer {
 public:
  template <typename T>
  static T* new_object() {
    return new T();
  }

  template <typename T>
  static std::shared_ptr<T> make_shared() {
    return std::shared_ptr<T>(new T());
  }
};
}  // namespace dgl

#endif  // DGL_GRAPH_SERIALIZER_H_
