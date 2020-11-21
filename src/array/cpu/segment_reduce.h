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
void SegmentSum(NDArray feat, NDArray offsets, NDArray out,
                int n, int dim) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (IdType j = offsets_data[i]; j < offsets_data[i + 1]; ++j) {
      for (int k = 0; k < dim; ++k) {
        out_data[i * dim + k] += feat[j * dim + k];
      }
    }
  }
}

template <typename IdType, typename DType, typename Cmp>
void SegmentCmp(NDArray feat, NDArray offsets,
                NDArray out, NDArray arg,
                int n, int dim) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
  std::fill(out_data, out_data + out.NumElements(), Cmp::zero), 
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (IdType j = offsets_data[i]; j < offsets_data[i + 1]; ++j) {
      for (int k = 0; k < dim; ++k) {
        const DType val = feat[i * dim + k];
        if (Cmp::Call(out_data[j * dim + k], val)) {
          out_data[i * dim + k] = feat[j * dim + k];
          arg[i * dim + k] = j;
        }
      }
    }
  }
}


}  // namespace cpu
}  // namespace aten
}  // namespace dgl

