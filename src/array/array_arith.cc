/**
 *  Copyright (c) 2019 by Contributors
 * @file array/array_aritch.cc
 * @brief DGL array arithmetic operations
 */
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>

#include "../c_api_common.h"
#include "./arith.h"
#include "./array_op.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

// Generate operators with both operations being NDArrays.
#define BINARY_ELEMENT_OP(name, op)                                  \
  IdArray name(IdArray lhs, IdArray rhs) {                           \
    IdArray ret;                                                     \
    CHECK_SAME_DTYPE(lhs, rhs);                                      \
    CHECK_SAME_CONTEXT(lhs, rhs);                                    \
    ATEN_XPU_SWITCH_CUDA(lhs->ctx.device_type, XPU, #name, {         \
      ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {                      \
        ret = impl::BinaryElewise<XPU, IdType, arith::op>(lhs, rhs); \
      });                                                            \
    });                                                              \
    return ret;                                                      \
  }

// Generate operators with only lhs being NDArray.
#define BINARY_ELEMENT_OP_L(name, op)                                \
  IdArray name(IdArray lhs, int64_t rhs) {                           \
    IdArray ret;                                                     \
    ATEN_XPU_SWITCH_CUDA(lhs->ctx.device_type, XPU, #name, {         \
      ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {                      \
        ret = impl::BinaryElewise<XPU, IdType, arith::op>(lhs, rhs); \
      });                                                            \
    });                                                              \
    return ret;                                                      \
  }

// Generate operators with only lhs being NDArray.
#define BINARY_ELEMENT_OP_R(name, op)                                \
  IdArray name(int64_t lhs, IdArray rhs) {                           \
    IdArray ret;                                                     \
    ATEN_XPU_SWITCH_CUDA(rhs->ctx.device_type, XPU, #name, {         \
      ATEN_ID_TYPE_SWITCH(rhs->dtype, IdType, {                      \
        ret = impl::BinaryElewise<XPU, IdType, arith::op>(lhs, rhs); \
      });                                                            \
    });                                                              \
    return ret;                                                      \
  }

// Generate operators with only lhs being NDArray.
#define UNARY_ELEMENT_OP(name, op)                             \
  IdArray name(IdArray lhs) {                                  \
    IdArray ret;                                               \
    ATEN_XPU_SWITCH_CUDA(lhs->ctx.device_type, XPU, #name, {   \
      ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {                \
        ret = impl::UnaryElewise<XPU, IdType, arith::op>(lhs); \
      });                                                      \
    });                                                        \
    return ret;                                                \
  }

BINARY_ELEMENT_OP(Add, Add)
BINARY_ELEMENT_OP(Sub, Sub)
BINARY_ELEMENT_OP(Mul, Mul)
BINARY_ELEMENT_OP(Div, Div)
BINARY_ELEMENT_OP(Mod, Mod)
BINARY_ELEMENT_OP(GT, GT)
BINARY_ELEMENT_OP(LT, LT)
BINARY_ELEMENT_OP(GE, GE)
BINARY_ELEMENT_OP(LE, LE)
BINARY_ELEMENT_OP(EQ, EQ)
BINARY_ELEMENT_OP(NE, NE)

BINARY_ELEMENT_OP_L(Add, Add)
BINARY_ELEMENT_OP_L(Sub, Sub)
BINARY_ELEMENT_OP_L(Mul, Mul)
BINARY_ELEMENT_OP_L(Div, Div)
BINARY_ELEMENT_OP_L(Mod, Mod)
BINARY_ELEMENT_OP_L(GT, GT)
BINARY_ELEMENT_OP_L(LT, LT)
BINARY_ELEMENT_OP_L(GE, GE)
BINARY_ELEMENT_OP_L(LE, LE)
BINARY_ELEMENT_OP_L(EQ, EQ)
BINARY_ELEMENT_OP_L(NE, NE)

BINARY_ELEMENT_OP_R(Add, Add)
BINARY_ELEMENT_OP_R(Sub, Sub)
BINARY_ELEMENT_OP_R(Mul, Mul)
BINARY_ELEMENT_OP_R(Div, Div)
BINARY_ELEMENT_OP_R(Mod, Mod)
BINARY_ELEMENT_OP_R(GT, GT)
BINARY_ELEMENT_OP_R(LT, LT)
BINARY_ELEMENT_OP_R(GE, GE)
BINARY_ELEMENT_OP_R(LE, LE)
BINARY_ELEMENT_OP_R(EQ, EQ)
BINARY_ELEMENT_OP_R(NE, NE)

UNARY_ELEMENT_OP(Neg, Neg)

}  // namespace aten
}  // namespace dgl

///////////////// Operator overloading for NDArray /////////////////
NDArray operator+(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::Add(lhs, rhs);
}
NDArray operator-(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::Sub(lhs, rhs);
}
NDArray operator*(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::Mul(lhs, rhs);
}
NDArray operator/(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::Div(lhs, rhs);
}
NDArray operator%(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::Mod(lhs, rhs);
}
NDArray operator+(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::Add(lhs, rhs);
}
NDArray operator-(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::Sub(lhs, rhs);
}
NDArray operator*(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::Mul(lhs, rhs);
}
NDArray operator/(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::Div(lhs, rhs);
}
NDArray operator%(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::Mod(lhs, rhs);
}
NDArray operator+(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::Add(lhs, rhs);
}
NDArray operator-(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::Sub(lhs, rhs);
}
NDArray operator*(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::Mul(lhs, rhs);
}
NDArray operator/(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::Div(lhs, rhs);
}
NDArray operator%(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::Mod(lhs, rhs);
}
NDArray operator-(const NDArray& array) { return dgl::aten::Neg(array); }

NDArray operator>(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::GT(lhs, rhs);
}
NDArray operator<(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::LT(lhs, rhs);
}
NDArray operator>=(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::GE(lhs, rhs);
}
NDArray operator<=(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::LE(lhs, rhs);
}
NDArray operator==(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::EQ(lhs, rhs);
}
NDArray operator!=(const NDArray& lhs, const NDArray& rhs) {
  return dgl::aten::NE(lhs, rhs);
}
NDArray operator>(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::GT(lhs, rhs);
}
NDArray operator<(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::LT(lhs, rhs);
}
NDArray operator>=(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::GE(lhs, rhs);
}
NDArray operator<=(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::LE(lhs, rhs);
}
NDArray operator==(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::EQ(lhs, rhs);
}
NDArray operator!=(const NDArray& lhs, int64_t rhs) {
  return dgl::aten::NE(lhs, rhs);
}
NDArray operator>(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::GT(lhs, rhs);
}
NDArray operator<(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::LT(lhs, rhs);
}
NDArray operator>=(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::GE(lhs, rhs);
}
NDArray operator<=(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::LE(lhs, rhs);
}
NDArray operator==(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::EQ(lhs, rhs);
}
NDArray operator!=(int64_t lhs, const NDArray& rhs) {
  return dgl::aten::NE(lhs, rhs);
}
