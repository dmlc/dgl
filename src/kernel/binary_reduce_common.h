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
  kNone,
};

enum BackwardMode {
  kGradLhs = 0,
  kGradRhs,
  kGradBoth,
};
}  // namespace binary_op

// Select src
struct SelectSrc {
  // Target value
  static constexpr binary_op::Target target = binary_op::kSrc;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return src;
  }
};

// Select dst
struct SelectDst {
  // Target value
  static constexpr binary_op::Target target = binary_op::kDst;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return dst;
  }
};

// Select edge
struct SelectEdge {
  // Target value
  static constexpr binary_op::Target target = binary_op::kEdge;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return edge;
  }
};

// Select none
struct SelectNone {
  // Target value
  static constexpr binary_op::Target target = binary_op::kNone;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return 0;
  }
};

// Change SelectSrc to SelectDst and vice versa
// SelectEdge will remain the same.
template <typename Selector>
struct SwitchSrcDst {
  typedef Selector Type;
};

template <>
struct SwitchSrcDst<SelectSrc> {
  typedef SelectDst Type;
};

template <>
struct SwitchSrcDst<SelectDst> {
  typedef SelectSrc Type;
};

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
    return -lhs / (rhs * rhs);
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

#define OP_TARGET_SWITCH(op, lhs, rhs, DType, OpType, LeftType, RightType, ...)   \
  {                                                            \
  using namespace binary_op;                                   \
  if (op == kAdd && lhs == kSrc && rhs == kDst) {              \
    typedef BinaryAdd<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kAdd && lhs == kSrc && rhs == kEdge) {      \
    typedef BinaryAdd<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kAdd && lhs == kDst && rhs == kEdge) {      \
    typedef BinaryAdd<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kMul && lhs == kSrc && rhs == kDst) {       \
    typedef BinaryMul<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kMul && lhs == kSrc && rhs == kEdge) {      \
    typedef BinaryMul<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kMul && lhs == kDst && rhs == kEdge) {      \
    typedef BinaryMul<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kSub && lhs == kSrc && rhs == kDst) {       \
    typedef BinarySub<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kSub && lhs == kDst && rhs == kSrc) {       \
    typedef BinarySub<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectSrc RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kSub && lhs == kSrc && rhs == kEdge) {      \
    typedef BinarySub<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kSub && lhs == kEdge && rhs == kSrc) {      \
    typedef BinarySub<DType> OpType;                           \
    typedef SelectEdge LeftType;                               \
    typedef SelectSrc RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kSub && lhs == kDst && rhs == kEdge) {      \
    typedef BinarySub<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kSub && lhs == kEdge && rhs == kDst) {      \
    typedef BinarySub<DType> OpType;                           \
    typedef SelectEdge LeftType;                               \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDiv && lhs == kSrc && rhs == kDst) {       \
    typedef BinaryDiv<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDiv && lhs == kDst && rhs == kSrc) {       \
    typedef BinaryDiv<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectSrc RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDiv && lhs == kSrc && rhs == kEdge) {      \
    typedef BinaryDiv<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kDiv && lhs == kEdge && rhs == kSrc) {      \
    typedef BinaryDiv<DType> OpType;                           \
    typedef SelectEdge LeftType;                               \
    typedef SelectSrc RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDiv && lhs == kDst && rhs == kEdge) {      \
    typedef BinaryDiv<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kDiv && lhs == kEdge && rhs == kDst) {      \
    typedef BinaryDiv<DType> OpType;                           \
    typedef SelectEdge LeftType;                               \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kUseLhs && lhs == kSrc) {                   \
    typedef BinaryUseLhs<DType> OpType;                        \
    typedef SelectSrc LeftType;                                \
    typedef SelectNone RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kUseLhs && lhs == kEdge) {                  \
    typedef BinaryUseLhs<DType> OpType;                        \
    typedef SelectEdge LeftType;                               \
    typedef SelectNone RightType;                              \
    {__VA_ARGS__}                                              \
  } else {                                                     \
    LOG(FATAL) << "Unsupported operation: op=" << op           \
      << " lhs=" << lhs << " rhs=" << rhs;                     \
  }                                                            \
  }

#define GEN_OP_TARGET(GEN, ...) \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectDst, BinaryAdd))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectEdge, BinaryAdd))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectEdge, BinaryAdd))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectDst, BinaryMul))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectEdge, BinaryMul))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectEdge, BinaryMul))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectDst, BinarySub))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectSrc, BinarySub))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectEdge, BinarySub))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectSrc, BinarySub))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectEdge, BinarySub))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectDst, BinarySub))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectDst, BinaryDiv))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectSrc, BinaryDiv))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectEdge, BinaryDiv))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectSrc, BinaryDiv))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectEdge, BinaryDiv))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectDst, BinaryDiv))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectNone, BinaryUseLhs))  \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectNone, BinaryUseLhs))

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
  MSVC_EXPAND(GEN(__VA_ARGS__, 2)) \
  MSVC_EXPAND(GEN(__VA_ARGS__, 4)) \
  MSVC_EXPAND(GEN(__VA_ARGS__, 8))

// macro for backward mode
#define BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, ...) \
  CHECK(!(req_lhs && req_rhs));                           \
  if (req_lhs) {                                          \
    constexpr int Mode = binary_op::kGradLhs;             \
    {__VA_ARGS__}                                         \
  } else {                                                \
    constexpr int Mode = binary_op::kGradRhs;             \
    {__VA_ARGS__}                                         \
  }

#define GEN_BACKWARD_MODE(GEN, ...)        \
  MSVC_EXPAND(GEN(__VA_ARGS__, binary_op::kGradLhs))    \
  MSVC_EXPAND(GEN(__VA_ARGS__, binary_op::kGradRhs))    \
  MSVC_EXPAND(GEN(__VA_ARGS__, binary_op::kGradBoth))

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_COMMON_H_
