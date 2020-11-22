/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/spmm.h
 * \brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SEGMENT_REDUCE_H_
#define DGL_ARRAY_CPU_SEGMENT_REDUCE_H_

#include <dgl/array.h>

namespace dgl {
namespace aten {
namespace cpu {

template <typename IdType, typename DType>
void SegmentSum(NDArray feat, NDArray offsets, NDArray out) {
  int n = out->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (IdType j = offsets_data[i]; j < offsets_data[i + 1]; ++j) {
      for (int k = 0; k < dim; ++k) {
        out_data[i * dim + k] += feat_data[j * dim + k];
      }
    }
  }
}

template <typename IdType, typename DType, typename Cmp>
void SegmentCmp(NDArray feat, NDArray offsets,
                NDArray out, NDArray arg) {
  int n = out->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
  IdType *arg_data = arg.Ptr<IdType>();
  std::fill(out_data, out_data + out.NumElements(), Cmp::zero);
  std::fill(arg_data, arg_data + arg.NumElements(), -1);
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (IdType j = offsets_data[i]; j < offsets_data[i + 1]; ++j) {
      for (int k = 0; k < dim; ++k) {
        const DType val = feat_data[j * dim + k];
        if (Cmp::Call(out_data[i * dim + k], val)) {
          out_data[i * dim + k] = val;
          arg_data[i * dim + k] = j;
        }
      }
    }
  }
}

template <typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  int n = feat->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* arg_data = arg.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < dim; ++k) {
      int write_row = arg_data[i * dim + k];
      if (write_row >= 0)
        out_data[write_row * dim + k] = feat_data[i * dim + k];
    }
  }
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SEGMENT_REDUCE_H_
