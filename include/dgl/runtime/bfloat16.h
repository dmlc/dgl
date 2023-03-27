/**
 *  Copyright (c) 2023 by Contributors
 * @file dgl/runtime/ndarray.h
 * @brief BFloat16 CPU header
 */
#ifndef DGL_RUNTIME_BFLOAT16_H_
#define DGL_RUNTIME_BFLOAT16_H_

#include <cmath>

class BFloat16 {
  uint16_t val;

 public:
  constexpr BFloat16() : val(0) {}
  // Disable lint "explicit" warning, since implicit usage on constructor is
  // expected.
  BFloat16(float f) {  // NOLINT
    if (std::isnan(f)) {
      val = 0x7FC0;
    } else {
      union {
        uint16_t iraw16[2];
        uint32_t iraw32;
        float f32;
      };

      f32 = f;
      const uint32_t rounding_bias = 0x00007FFF + (iraw16[1] & 0x1);
      val = static_cast<uint16_t>((iraw32 + rounding_bias) >> 16);
    }
  }
  static constexpr BFloat16 Min() {
    BFloat16 min;
    min.val = 0xFF80;
    return min;
  }

  static constexpr BFloat16 Max() {
    BFloat16 max;
    max.val = 0x7F80;
    return max;
  }

  BFloat16& operator-=(const float& rhs) {
    float lhs = (*this);
    (*this) = lhs - rhs;
    return *this;
  }

  BFloat16& operator+=(const float& rhs) {
    float lhs = (*this);
    (*this) = lhs + rhs;
    return *this;
  }

  operator float() const {
    union {
      float f;
      uint16_t raw[2];
    };
    raw[0] = 0;
    raw[1] = val;
    return f;
  }
};

#endif  // DGL_RUNTIME_BFLOAT16_H_
