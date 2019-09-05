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
/*! \brief Reducer names. */
static const char kReduceSum[] = "sum";
static const char kReduceMax[] = "max";
static const char kReduceMin[] = "min";
static const char kReduceMean[] = "mean";
static const char kReduceProd[] = "prod";
static const char kReduceNone[] = "none";

/*! \brief Binary op names. */
static const char kAdd[] = "add";
static const char kSub[] = "sub";
static const char kMul[] = "mul";
static const char kDiv[] = "div";
static const char kDot[] = "dot";
static const char kUseLhs[] = "use_lhs";

/*!
 * \brief Enum code for operand targets.
 * \seealso BinaryOpReduce in binary_reduce_common.h
 */
enum Target {
  kSrc = 0,  // select src node
  kDst,      // select dst node
  kEdge,     // select edge
  kNone,     // select none
};

/*! \brief Enum code for backward operator mode. */
enum BackwardMode {
  kGradLhs = 0,  // compute lhs gradient
  kGradRhs,      // compute rhs gradient
  kGradBoth,     // compute both gradients
};
}  // namespace binary_op

//////////////////////////////////////////////////////////////////////////
// Defines operand target category. Each category is a structure with
// two static members:
//  - target: The enum code of this category.
//  - Call: The call functor that returns the selected target.
//////////////////////////////////////////////////////////////////////////

/*! \brief Select src category. */
struct SelectSrc {
  // Target value
  static constexpr binary_op::Target target = binary_op::kSrc;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return src;
  }
};

/*! \brief Select dst category. */
struct SelectDst {
  // Target value
  static constexpr binary_op::Target target = binary_op::kDst;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return dst;
  }
};

/*! \brief Select edge category. */
struct SelectEdge {
  // Target value
  static constexpr binary_op::Target target = binary_op::kEdge;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return edge;
  }
};

/*! \brief Select none category. */
struct SelectNone {
  // Target value
  static constexpr binary_op::Target target = binary_op::kNone;
  // Call functor.
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return 0;
  }
};

/*! \brief Type functor to switch SelectSrc and SelectDst category.
 * SelectEdge and SelectNone will remain the same. */
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

//////////////////////////////////////////////////////////////////////////
// Defines binary op category. Each category is a structure with
// three static members:
//  - Call: The forward computation given two operand.
//  - BackwardLhs: Compute lhs gradient.
//  - BackwardRhs: Compute rhs gradient.
//////////////////////////////////////////////////////////////////////////

// common binary functors
template <typename DType>
struct BinaryAdd {
  static DGLDEVICE DGLINLINE DType Call(DType *lhs, DType *rhs, int64_t len) {
    return lhs[0] + rhs[0];
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
  static DGLDEVICE DGLINLINE DType Call(DType *lhs, DType *rhs, int64_t len) {
    return lhs[0] * rhs[0];
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
  static DGLDEVICE DGLINLINE DType Call(DType *lhs, DType *rhs, int64_t len) {
    return lhs[0] - rhs[0];
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
  static DGLDEVICE DGLINLINE DType Call(DType *lhs, DType *rhs, int64_t len) {
    return lhs[0] / rhs[0];
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
  static DGLDEVICE DGLINLINE DType Call(DType *lhs, DType *rhs, int64_t len) {
    return lhs[0];
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return 1;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return 0;
  }
};

template <typename DType>
struct BinaryDot {
  static DGLDEVICE DGLINLINE DType Call(DType *lhs, DType *rhs, int64_t len) {
    DType out = 0;
    // simple vector dot vector
#pragma unroll
    for (int i = 0; i < len; i ++)
      out += lhs[i] * rhs[i];

    return out;
  }
  static DGLDEVICE DGLINLINE DType BackwardLhs(DType lhs, DType rhs, DType out) {
    return rhs;
  }
  static DGLDEVICE DGLINLINE DType BackwardRhs(DType lhs, DType rhs, DType out) {
    return lhs;
  }
};

// Macro for dispatching op enum code and target code into template arguments.
// The macro dispatches following combinations:
//  - Add(Src, Dst), Add(Src, Edge), Add(Dst, Edge)
//  - Mul(Src, Dst), Mul(Src, Edge), Mul(Dst, Edge)
//  - Sub(Src, Dst), Sub(Src, Edge), Sub(Dst, Edge)
//    Sub(Dst, Src), Sub(Edge, Src), Sub(Edge, Dst)
//  - Div(Src, Dst), Div(Src, Edge), Div(Dst, Edge)
//    Div(Dst, Src), Div(Edge, Src), Div(Edge, Dst)
//  - UseLhs(Src, None), UseLhs(Edge, None)
//  - Dot(Src, Dst), Dot(Src, Edge), Dot(Dst, Edge)
//  - Dot(Dst, Src), Dot(Edge, Src), Dot(Edge, Dst)
// Note that for commutative operators (e.g. Add and Mul), we only generate
// kernels for lhs code smaller than rhs code.
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
  } else if (op == kDot && lhs == kSrc && rhs == kDst) {       \
    typedef BinaryDot<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDot && lhs == kSrc && rhs == kEdge) {      \
    typedef BinaryDot<DType> OpType;                           \
    typedef SelectSrc LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kDot && lhs == kDst && rhs == kEdge) {      \
    typedef BinaryDot<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectEdge RightType;                              \
    {__VA_ARGS__}                                              \
  } else if (op == kDot && lhs == kDst && rhs == kSrc) {       \
    typedef BinaryDot<DType> OpType;                           \
    typedef SelectDst LeftType;                                \
    typedef SelectSrc RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDot && lhs == kEdge && rhs == kSrc) {      \
    typedef BinaryDot<DType> OpType;                           \
    typedef SelectEdge LeftType;                               \
    typedef SelectSrc RightType;                               \
    {__VA_ARGS__}                                              \
  } else if (op == kDot && lhs == kEdge && rhs == kDst) {      \
    typedef BinaryDot<DType> OpType;                           \
    typedef SelectEdge LeftType;                               \
    typedef SelectDst RightType;                               \
    {__VA_ARGS__}                                              \
  } else {                                                     \
    LOG(FATAL) << "Unsupported operation: op=" << op           \
      << " lhs=" << lhs << " rhs=" << rhs;                     \
  }                                                            \
  }

// Macro for unrolling with various template argument combinations
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
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectNone, BinaryUseLhs)) \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectDst, BinaryDot))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectSrc, SelectEdge, BinaryDot))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectEdge, BinaryDot))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectDst, SelectSrc, BinaryDot))      \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectSrc, BinaryDot))     \
  MSVC_EXPAND(GEN(__VA_ARGS__, SelectEdge, SelectDst, BinaryDot))

//////////////////////////////////////////////////////////////////////////
// Defines reducer category. Each category is an empty structure.
// The call functor is device dependent, so should be specialized
// in the each device's implementation.
// See Also:
//  - kernel/cpu/functor.h
//  - kernel/cuda/functor.cuh
//////////////////////////////////////////////////////////////////////////

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

// Macro for dispatching reducer names to Reducer op structure
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

// Type trait for getting zero value of the given reducer type.
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

// Type functor for selecting output target based on reducer type.
/*! \brief For all the reducer types except ReduceNone, select dst as the output target. */
template <typename Reducer>
struct OutSelector {
  typedef SelectDst Type;
};

/*! \brief For ReduceNone, select edge as the output target. */
template <int XPU, typename DType>
struct OutSelector<ReduceNone<XPU, DType>> {
  typedef SelectEdge Type;
};

// macro for dispatching number of broadcasting dimensions to template argument
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

// macro for unrolling different broadcasting dimensions
#define GEN_NDIM(GEN, ...) \
  MSVC_EXPAND(GEN(__VA_ARGS__, 2)) \
  MSVC_EXPAND(GEN(__VA_ARGS__, 4)) \
  MSVC_EXPAND(GEN(__VA_ARGS__, 8))

// macro for dispatching backward mode enum to template argument
#define BACKWARD_MODE_SWITCH(req_lhs, req_rhs, Mode, ...) \
  CHECK(!(req_lhs && req_rhs));                           \
  if (req_lhs) {                                          \
    constexpr int Mode = binary_op::kGradLhs;             \
    {__VA_ARGS__}                                         \
  } else {                                                \
    constexpr int Mode = binary_op::kGradRhs;             \
    {__VA_ARGS__}                                         \
  }

// macro for unrolling different backward mode
#define GEN_BACKWARD_MODE(GEN, ...)        \
  MSVC_EXPAND(GEN(__VA_ARGS__, binary_op::kGradLhs))    \
  MSVC_EXPAND(GEN(__VA_ARGS__, binary_op::kGradRhs))    \
  MSVC_EXPAND(GEN(__VA_ARGS__, binary_op::kGradBoth))

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_COMMON_H_
