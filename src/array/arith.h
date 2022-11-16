/**
 *  Copyright (c) 2019 by Contributors
 * @file array/arith.h
 * @brief Arithmetic functors
 */
#ifndef DGL_ARRAY_ARITH_H_
#define DGL_ARRAY_ARITH_H_

#ifdef __CUDACC__
#define DGLDEVICE __device__
#define DGLINLINE __forceinline__
#else
#define DGLDEVICE
#define DGLINLINE inline
#endif  // __CUDACC__

namespace dgl {
namespace aten {
namespace arith {

struct Add {
  template <typename T>
  static DGLINLINE DGLDEVICE T Call(const T& t1, const T& t2) {
    return t1 + t2;
  }
};

struct Sub {
  template <typename T>
  static DGLINLINE DGLDEVICE T Call(const T& t1, const T& t2) {
    return t1 - t2;
  }
};

struct Mul {
  template <typename T>
  static DGLINLINE DGLDEVICE T Call(const T& t1, const T& t2) {
    return t1 * t2;
  }
};

struct Div {
  template <typename T>
  static DGLINLINE DGLDEVICE T Call(const T& t1, const T& t2) {
    return t1 / t2;
  }
};

struct Mod {
  template <typename T>
  static DGLINLINE DGLDEVICE T Call(const T& t1, const T& t2) {
    return t1 % t2;
  }
};

struct GT {
  template <typename T>
  static DGLINLINE DGLDEVICE bool Call(const T& t1, const T& t2) {
    return t1 > t2;
  }
};

struct LT {
  template <typename T>
  static DGLINLINE DGLDEVICE bool Call(const T& t1, const T& t2) {
    return t1 < t2;
  }
};

struct GE {
  template <typename T>
  static DGLINLINE DGLDEVICE bool Call(const T& t1, const T& t2) {
    return t1 >= t2;
  }
};

struct LE {
  template <typename T>
  static DGLINLINE DGLDEVICE bool Call(const T& t1, const T& t2) {
    return t1 <= t2;
  }
};

struct EQ {
  template <typename T>
  static DGLINLINE DGLDEVICE bool Call(const T& t1, const T& t2) {
    return t1 == t2;
  }
};

struct NE {
  template <typename T>
  static DGLINLINE DGLDEVICE bool Call(const T& t1, const T& t2) {
    return t1 != t2;
  }
};

struct Neg {
  template <typename T>
  static DGLINLINE DGLDEVICE T Call(const T& t1) {
    return -t1;
  }
};

}  // namespace arith
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_ARITH_H_
