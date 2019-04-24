#ifndef DGL_KERNEL_BINARY_REDUCE_COMMON_H_
#define DGL_KERNEL_BINARY_REDUCE_COMMON_H_

#include <limits>
#include <string>
#include "./common.h"

namespace dgl {
namespace kernel {
namespace binary_op {

static const std::string kReduceSum = "sum";
static const std::string kReduceMax = "max";
static const std::string kReduceMin = "min";
static const std::string kReduceMean = "mean";
static const std::string kReduceProd = "prod";
static const std::string kReduceNone = "none";

enum Target {
  kSrc = 0,
  kDst,
  kEdge,
};
}  // namespace binary_op


// functor for no-op
template <typename Ret, typename ... Args>
struct Nop {
  static DGLDEVICE DGLINLINE Ret Call(Args ... args) {
    return 0;
  }
};

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

#define MAPPING_SWITCH(val, XPU, MapType, ...) \
  if (val->ndim == 0) {                        \
    typedef DirectId<int64_t> MapType;         \
    {__VA_ARGS__}                              \
  } else {                                     \
    typedef IndirectId<XPU, int64_t> MapType;  \
    {__VA_ARGS__}                              \
  }

// common binary functors
template <typename DType>
struct BinaryAdd {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs + rhs;
  }
};

template <typename DType>
struct BinaryMul {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs * rhs;
  }
};

template <typename DType>
struct BinarySub {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs - rhs;
  }
};

template <typename DType>
struct BinaryDiv {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs / rhs;
  }
};

template <typename DType>
struct BinaryUseLhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs;
  }
};

template <typename DType>
struct BinaryUseRhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return rhs;
  }
};

#define BINARY_OP_SWITCH(val, DType, OpType, ...)   \
  if (val == "add") {                               \
    typedef BinaryAdd<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "sub") {                        \
    typedef BinarySub<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "mul") {                        \
    typedef BinaryMul<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "div") {                        \
    typedef BinaryDiv<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else {                                          \
    LOG(FATAL) << "Unsupported binary op: " << val; \
  }

#define GEN_BINARY_OP(GEN, ...) \
  GEN(__VA_ARGS__, BinaryAdd) \
  GEN(__VA_ARGS__, BinarySub) \
  GEN(__VA_ARGS__, BinaryMul) \
  GEN(__VA_ARGS__, BinaryDiv)

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

// functors for selecting output target
template <typename Reducer>
struct OutSelector {
  typedef SelectDst Type;
};

template <int XPU, typename DType>
struct OutSelector<ReduceNone<XPU, DType>> {
  typedef SelectEdge Type;
};

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_COMMON_H_
