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

// direct id
template <typename IdxType>
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


}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_FUNCTOR_H_
