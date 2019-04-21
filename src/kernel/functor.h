#ifndef DGL_KERNEL_FUNCTOR_H_
#define DGL_KERNEL_FUNCTOR_H_

#include "./common.h"

namespace dgl {
namespace kernel {

// functor for no-op
template <typename Ret, typename ... Args>
struct Nop {
  static DGLDEVICE DGLINLINE Ret Call(Args ... args) {
    return 0;
  }
};

// Read src node data
template <typename DType, typename ReadOp>
struct ReadSrc {
  static DGLDEVICE DGLINLINE DType Call(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return ReadOp::Call(src_data);
  }
};

// Read edge data
template <typename DType, typename ReadOp>
struct ReadEdge {
  static DGLDEVICE DGLINLINE DType Call(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return ReadOp::Call(edge_data);
  }
};

// Read dst node data
template <typename DType, typename ReadOp>
struct ReadDst {
  static DGLDEVICE DGLINLINE DType Call(
      DType* src_data, DType* edge_data, DType* dst_data) {
    return ReadOp::Call(dst_data);
  }
};

// Select src node id
template <typename IdxType>
struct SelectSrc {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType src, IdxType eid, IdxType dst) {
    return src;
  }
};

// Select dst node id
template <typename IdxType>
struct SelectDst {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType src, IdxType eid, IdxType dst) {
    return dst;
  }
};

// Select dst node id
template <typename IdxType>
struct SelectEdge {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType src, IdxType eid, IdxType dst) {
    return eid;
  }
};

// direct id
template <typename IdxType>
struct DirectId {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType id, IdxType* shuffle_ids) {
    return id;
  }
};

// id mapped by another array
template <typename IdxType, typename ReadOp>
struct IndirectId {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType id, IdxType* shuffle_ids) {
    return ReadOp::Call(shuffle_ids + id);
  }
};


// functors for binary operation
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

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_FUNCTOR_H_
