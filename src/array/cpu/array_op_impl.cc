/**
 *  Copyright (c) 2019 by Contributors
 * @file array/cpu/array_op_impl.cc
 * @brief Array operator CPU implementation
 */
#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/parallel_for.h>

#include <numeric>

#include "../arith.h"

namespace dgl {
using runtime::NDArray;
using runtime::parallel_for;
namespace aten {
namespace impl {

///////////////////////////// AsNumBits /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits) {
  CHECK(bits == 32 || bits == 64) << "invalid number of integer bits";
  if (sizeof(IdType) * 8 == bits) {
    return arr;
  }
  const int64_t len = arr->shape[0];
  IdArray ret = NewIdArray(len, arr->ctx, bits);
  const IdType* arr_data = static_cast<IdType*>(arr->data);
  if (bits == 32) {
    int32_t* ret_data = static_cast<int32_t*>(ret->data);
    for (int64_t i = 0; i < len; ++i) {
      ret_data[i] = arr_data[i];
    }
  } else {
    int64_t* ret_data = static_cast<int64_t*>(ret->data);
    for (int64_t i = 0; i < len; ++i) {
      ret_data[i] = arr_data[i];
    }
  }
  return ret;
}

template IdArray AsNumBits<kDGLCPU, int32_t>(IdArray arr, uint8_t bits);
template IdArray AsNumBits<kDGLCPU, int64_t>(IdArray arr, uint8_t bits);

///////////////////////////// BinaryElewise /////////////////////////////

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs) {
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  const IdType* rhs_data = static_cast<IdType*>(rhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  // TODO(BarclayII): this usually incurs lots of overhead in thread spawning,
  // scheduling, etc., especially since the workload is very light.  Need to
  // replace with parallel_for.
  for (int64_t i = 0; i < lhs->shape[0]; i++) {
    ret_data[i] = Op::Call(lhs_data[i], rhs_data[i]);
  }
  return ret;
}

template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Add>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Sub>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Mul>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Div>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Mod>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::GT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::LT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::GE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::LE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::EQ>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::NE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Add>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Sub>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Mul>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Div>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Mod>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::GT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::LT>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::GE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::LE>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::EQ>(
    IdArray lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::NE>(
    IdArray lhs, IdArray rhs);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs) {
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  // TODO(BarclayII): this usually incurs lots of overhead in thread spawning,
  // scheduling, etc., especially since the workload is very light.  Need to
  // replace with parallel_for.
  for (int64_t i = 0; i < lhs->shape[0]; i++) {
    ret_data[i] = Op::Call(lhs_data[i], rhs);
  }
  return ret;
}

template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Add>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Sub>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Mul>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Div>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Mod>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::GT>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::LT>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::GE>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::LE>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::EQ>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::NE>(
    IdArray lhs, int32_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Add>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Sub>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Mul>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Div>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Mod>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::GT>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::LT>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::GE>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::LE>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::EQ>(
    IdArray lhs, int64_t rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::NE>(
    IdArray lhs, int64_t rhs);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs) {
  IdArray ret = NewIdArray(rhs->shape[0], rhs->ctx, rhs->dtype.bits);
  const IdType* rhs_data = static_cast<IdType*>(rhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  // TODO(BarclayII): this usually incurs lots of overhead in thread spawning,
  // scheduling, etc., especially since the workload is very light.  Need to
  // replace with parallel_for.
  for (int64_t i = 0; i < rhs->shape[0]; i++) {
    ret_data[i] = Op::Call(lhs, rhs_data[i]);
  }
  return ret;
}

template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Add>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Sub>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Mul>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Div>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::Mod>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::GT>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::LT>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::GE>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::LE>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::EQ>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int32_t, arith::NE>(
    int32_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Add>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Sub>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Mul>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Div>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::Mod>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::GT>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::LT>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::GE>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::LE>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::EQ>(
    int64_t lhs, IdArray rhs);
template IdArray BinaryElewise<kDGLCPU, int64_t, arith::NE>(
    int64_t lhs, IdArray rhs);

template <DGLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray lhs) {
  IdArray ret = NewIdArray(lhs->shape[0], lhs->ctx, lhs->dtype.bits);
  const IdType* lhs_data = static_cast<IdType*>(lhs->data);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  // TODO(BarclayII): this usually incurs lots of overhead in thread spawning,
  // scheduling, etc., especially since the workload is very light.  Need to
  // replace with parallel_for.
  for (int64_t i = 0; i < lhs->shape[0]; i++) {
    ret_data[i] = Op::Call(lhs_data[i]);
  }
  return ret;
}

template IdArray UnaryElewise<kDGLCPU, int32_t, arith::Neg>(IdArray lhs);
template IdArray UnaryElewise<kDGLCPU, int64_t, arith::Neg>(IdArray lhs);

///////////////////////////// Full /////////////////////////////

template <DGLDeviceType XPU, typename DType>
NDArray Full(DType val, int64_t length, DGLContext ctx) {
  NDArray ret = NDArray::Empty({length}, DGLDataTypeTraits<DType>::dtype, ctx);
  DType* ret_data = static_cast<DType*>(ret->data);
  std::fill(ret_data, ret_data + length, val);
  return ret;
}

template NDArray Full<kDGLCPU, int32_t>(
    int32_t val, int64_t length, DGLContext ctx);
template NDArray Full<kDGLCPU, int64_t>(
    int64_t val, int64_t length, DGLContext ctx);
template NDArray Full<kDGLCPU, float>(
    float val, int64_t length, DGLContext ctx);
template NDArray Full<kDGLCPU, double>(
    double val, int64_t length, DGLContext ctx);

///////////////////////////// Range /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DGLContext ctx) {
  CHECK(high >= low) << "high must be bigger than low";
  IdArray ret = NewIdArray(high - low, ctx, sizeof(IdType) * 8);
  IdType* ret_data = static_cast<IdType*>(ret->data);
  std::iota(ret_data, ret_data + high - low, low);
  return ret;
}

template IdArray Range<kDGLCPU, int32_t>(int32_t, int32_t, DGLContext);
template IdArray Range<kDGLCPU, int64_t>(int64_t, int64_t, DGLContext);

///////////////////////////// Relabel_ /////////////////////////////

template <DGLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays) {
  // build map & relabel
  IdType newid = 0;
  std::unordered_map<IdType, IdType> oldv2newv;
  for (IdArray arr : arrays) {
    for (int64_t i = 0; i < arr->shape[0]; ++i) {
      const IdType id = static_cast<IdType*>(arr->data)[i];
      if (!oldv2newv.count(id)) {
        oldv2newv[id] = newid++;
      }
      static_cast<IdType*>(arr->data)[i] = oldv2newv[id];
    }
  }
  // map array
  IdArray maparr =
      NewIdArray(newid, DGLContext{kDGLCPU, 0}, sizeof(IdType) * 8);
  IdType* maparr_data = static_cast<IdType*>(maparr->data);
  for (const auto& kv : oldv2newv) {
    maparr_data[kv.second] = kv.first;
  }
  return maparr;
}

template IdArray Relabel_<kDGLCPU, int32_t>(const std::vector<IdArray>& arrays);
template IdArray Relabel_<kDGLCPU, int64_t>(const std::vector<IdArray>& arrays);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
