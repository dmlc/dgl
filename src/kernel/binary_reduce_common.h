/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/binary_reduce_common.h
 * \brief Common utilities for binary reduce operation.
 */
#ifndef DGL_KERNEL_BINARY_REDUCE_COMMON_H_
#define DGL_KERNEL_BINARY_REDUCE_COMMON_H_

#include <dgl/runtime/ndarray.h>

#include <limits>
#include <string>

#include "./common.h"

namespace dgl {
namespace kernel {
namespace binary_op {

static const char kReduceSum[] = "sum";
static const char kReduceMax[] = "max";
static const char kReduceMin[] = "min";
static const char kReduceMean[] = "mean";
static const char kReduceProd[] = "prod";
static const char kReduceNone[] = "none";

static const char kAdd[] = "add";
static const char kSub[] = "sub";
static const char kMul[] = "mul";
static const char kDiv[] = "div";
static const char kUseLhs[] = "use_lhs";

enum Target {
  kSrc = 0,
  kDst,
  kEdge,
};

enum BackwardMode {
  kGradLhs = 0,
  kGradRhs,
  kGradBoth,
};
}  // namespace binary_op

// Select src
struct SelectSrc {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return src;
  }
};

// Select dst
struct SelectDst {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return dst;
  }
};

// Select edge
struct SelectEdge {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return edge;
  }
};

#define TARGET_SWITCH(v1, v2, Tgt1, Tgt2, ...)                  \
  if (v1 == binary_op::kSrc && v2 == binary_op::kDst) {         \
    typedef SelectSrc Tgt1;                                     \
    typedef SelectDst Tgt2;                                     \
    {__VA_ARGS__}                                               \
  } else if (v1 == binary_op::kSrc && v2 == binary_op::kEdge) { \
    typedef SelectSrc Tgt1;                                     \
    typedef SelectEdge Tgt2;                                     \
    {__VA_ARGS__}                                               \
  } else if (v1 == binary_op::kEdge && v2 == binary_op::kDst) { \
    typedef SelectEdge Tgt1;                                     \
    typedef SelectDst Tgt2;                                     \
    {__VA_ARGS__}                                               \
  } else if (v1 == binary_op::kDst && v2 == binary_op::kEdge) { \
    typedef SelectDst Tgt1;                                     \
    typedef SelectEdge Tgt2;                                     \
    {__VA_ARGS__}                                               \
  } else if (v1 == binary_op::kDst && v2 == binary_op::kSrc) { \
    typedef SelectDst Tgt1;                                     \
    typedef SelectSrc Tgt2;                                     \
    {__VA_ARGS__}                                               \
  } else if (v1 == binary_op::kEdge && v2 == binary_op::kSrc) { \
    typedef SelectEdge Tgt1;                                     \
    typedef SelectSrc Tgt2;                                     \
    {__VA_ARGS__}                                               \
  } else {                                                      \
    LOG(FATAL) << "Invalid operand target: " << v1 << " and " << v2; \
  }

#define GEN_TARGET(GEN, ...)               \
  GEN(__VA_ARGS__, SelectSrc, SelectDst)   \
  GEN(__VA_ARGS__, SelectDst, SelectSrc)   \
  GEN(__VA_ARGS__, SelectSrc, SelectEdge)  \
  GEN(__VA_ARGS__, SelectEdge, SelectSrc)  \
  GEN(__VA_ARGS__, SelectDst, SelectEdge)  \
  GEN(__VA_ARGS__, SelectEdge, SelectDst)

// direct id
template <int XPU, typename IdxType>
struct DirectId {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType id, IdxType* shuffle_ids) {
    return id;
  }
};

// id mapped by another array
template <int XPU, typename IdxType>
struct IndirectId {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType id, IdxType* shuffle_ids);
};

// common binary functors
template <typename DType>
struct BinaryAdd {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs + rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return 1;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return 1;
  }
};

template <typename DType>
struct BinaryMul {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs * rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return lhs;
  }
};

template <typename DType>
struct BinarySub {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs - rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return 1;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return -1;
  }
};

template <typename DType>
struct BinaryDiv {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs / rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return static_cast<DType>(1) / rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return -lhs;
  }
};

template <typename DType>
struct BinaryUseLhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return 1;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return 0;
  }
};

#define BINARY_OP_SWITCH(val, DType, OpType, ...)   \
  if (val == binary_op::kAdd) {                     \
    typedef BinaryAdd<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == binary_op::kSub) {              \
    typedef BinarySub<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == binary_op::kMul) {              \
    typedef BinaryMul<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == binary_op::kDiv) {              \
    typedef BinaryDiv<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == binary_op::kUseLhs) {           \
    typedef BinaryUseLhs<DType> OpType;             \
    {__VA_ARGS__}                                   \
  } else {                                          \
    LOG(FATAL) << "Unsupported binary op: " << val; \
  }

#define EXPAND( x ) x
#define GEN_BINARY_OP(GEN, ...) \
  EXPAND(GEN(__VA_ARGS__, BinaryAdd)) \
  EXPAND(GEN(__VA_ARGS__, BinarySub)) \
  EXPAND(GEN(__VA_ARGS__, BinaryMul)) \
  EXPAND(GEN(__VA_ARGS__, BinaryDiv)) \
  EXPAND(GEN(__VA_ARGS__, BinaryUseLhs))

// functors for reducers
template <int XPU, typename DType>
struct ReduceSum { };

template <int XPU, typename DType>
struct ReduceMax { };

template <int XPU, typename DType>
struct ReduceMin { };

template <int XPU, typename DType>
struct ReduceProd { };

template <int XPU, typename DType>
struct ReduceNone { };

#define REDUCER_SWITCH(val, XPU, DType, RedType, ...)   \
  if (val == binary_op::kReduceSum                 \
      || val == binary_op::kReduceMean) {          \
    typedef ReduceSum<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMax) {       \
    typedef ReduceMax<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMin) {       \
    typedef ReduceMin<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceProd) {      \
    typedef ReduceProd<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceNone) {      \
    typedef ReduceNone<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else {                                         \
    LOG(FATAL) << "Unsupported reducer: " << val;  \
  }

// functors for zero elements of reducers
template <typename Reducer>
struct Zero { };

template <int XPU, typename DType>
struct Zero<ReduceSum<XPU, DType>> {
  static constexpr DType value = 0;
};

template <int XPU, typename DType>
struct Zero<ReduceMax<XPU, DType>> {
  static constexpr DType value = std::numeric_limits<DType>::lowest();
};

template <int XPU, typename DType>
struct Zero<ReduceMin<XPU, DType>> {
  static constexpr DType value = std::numeric_limits<DType>::max();
};

template <int XPU, typename DType>
struct Zero<ReduceProd<XPU, DType>> {
  static constexpr DType value = 1;
};

template <int XPU, typename DType>
struct Zero<ReduceNone<XPU, DType>> {
  static constexpr DType value = 0;
};

template <int XPU, typename DType>
constexpr DType Zero<ReduceSum<XPU, DType>>::value;

template <int XPU, typename DType>
constexpr DType Zero<ReduceMax<XPU, DType>>::value;

template <int XPU, typename DType>
constexpr DType Zero<ReduceMin<XPU, DType>>::value;

template <int XPU, typename DType>
constexpr DType Zero<ReduceProd<XPU, DType>>::value;

template <int XPU, typename DType>
constexpr DType Zero<ReduceNone<XPU, DType>>::value;

// Selecting output target based on reducer type
template <typename Reducer>
struct OutSelector {
  typedef SelectDst Type;
};

template <int XPU, typename DType>
struct OutSelector<ReduceNone<XPU, DType>> {
  typedef SelectEdge Type;
};

template <typename Reducer>
struct GradOutSelector {
  typedef SelectSrc Type;
};

template <int XPU, typename DType>
struct GradOutSelector<ReduceNone<XPU, DType>> {
  typedef SelectEdge Type;
};

// macro for broadcasting
#define BCAST_NDIM_SWITCH(ndim, NDim, ...) \
  if (ndim <= 2) {                         \
    constexpr int NDim = 2;                \
    {__VA_ARGS__}                          \
  } else if (ndim <= 4) {                  \
    constexpr int NDim = 4;                \
    {__VA_ARGS__}                          \
  } else if (ndim <= 8) {                  \
    constexpr int NDim = 8;                \
    {__VA_ARGS__}                          \
  } else {                                 \
    LOG(FATAL) << "Too many broadcasting dimensions."; \
  }

#define GEN_NDIM(GEN, ...) \
  GEN(__VA_ARGS__, 2) \
  GEN(__VA_ARGS__, 4) \
  GEN(__VA_ARGS__, 8)

// macro for backward mode
#define BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, ...) \
  if (req_lhs && req_rhs) {                               \
    constexpr int Mode = binary_op::kGradBoth;            \
    {__VA_ARGS__}                                         \
  } else if (req_lhs) {                                   \
    constexpr int Mode = binary_op::kGradLhs;             \
    {__VA_ARGS__}                                         \
  } else {                                                \
    constexpr int Mode = binary_op::kGradRhs;             \
    {__VA_ARGS__}                                         \
  }

#define GEN_BACKWARD_MODE(GEN, ...)        \
  GEN(__VA_ARGS__, binary_op::kGradLhs)    \
  GEN(__VA_ARGS__, binary_op::kGradRhs)    \
  GEN(__VA_ARGS__, binary_op::kGradBoth)

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_COMMON_H_
