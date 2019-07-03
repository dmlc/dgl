/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/arith.h
 * \brief Arithmetic functors
 */

namespace dgl {
namespace aten {
namespace arith {

struct Add {
  template <typename T>
  inline static T Call(const T& t1, const T& t2) {
    return t1 + t2;
  }
};

struct Sub {
  template <typename T>
  inline static T Call(const T& t1, const T& t2) {
    return t1 - t2;
  }
};

struct Mul {
  template <typename T>
  inline static T Call(const T& t1, const T& t2) {
    return t1 * t2;
  }
};

struct Div {
  template <typename T>
  inline static T Call(const T& t1, const T& t2) {
    return t1 / t2;
  }
};

}  // namespace arith
}  // namespace aten
}  // namespace dgl
