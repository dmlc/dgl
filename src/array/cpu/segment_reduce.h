/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/spmm.h
 * \brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SEGMENT_REDUCE_H_
#define DGL_ARRAY_CPU_SEGMENT_REDUCE_H_

#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/base_heterograph.h>

namespace dgl {
namespace aten {
namespace cpu {

/*!
 * \brief CPU kernel of segment sum.
 * \param feat The input tensor.
 * \param offsets The offset tensor storing the ranges of segments.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void SegmentSum(NDArray feat, NDArray offsets, NDArray out) {
  int n = out->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
  runtime::parallel_for(0, n, [=](int b, int e) {
    for (auto i = b; i < e; ++i) {
      for (IdType j = offsets_data[i]; j < offsets_data[i + 1]; ++j) {
        for (int k = 0; k < dim; ++k) {
          out_data[i * dim + k] += feat_data[j * dim + k];
        }
      }
    }
  });
}

/*!
 * \brief CPU kernel of segment min/max.
 * \param feat The input tensor.
 * \param offsets The offset tensor storing the ranges of segments.
 * \param out The output tensor.
 * \param arg An auxiliary tensor storing the argmin/max information
 *        used in backward phase.
 */
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
  runtime::parallel_for(0, n, [=](int b, int e) {
    for (auto i = b; i < e; ++i) {
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
  });
}

/*!
 * \brief CPU kernel of Scatter Add (on first dimension) operator.
 * \note math equation: out[idx[i], *] += feat[i, *]
 * \param feat The input tensor.
 * \param idx The indices tensor.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out) {
  int n = feat->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* idx_data = idx.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    const int write_row = idx_data[i];
    for (int k = 0; k < dim; ++k) {
#pragma omp atomic
      out_data[write_row * dim + k] += feat_data[i * dim + k];
    }
  }
}

/*!
 * \brief CPU kernel of Scatter Add (on first dimension) operator for heterogeneous graph.
 * \note math equation: out[idx[i], *] += feat[i, *]
 * \param feat The input tensor.
 * \param idx The indices tensor.
 * \param out The output tensor.
 */

template <typename IdType, typename DType>
void ScatterAdd_hetero(HeteroGraphPtr graph,
                       std::vector<NDArray> list_feat,
                       std::vector<NDArray> list_idx,
                       std::vector<NDArray> list_idx_ntypes,
                       std::vector<NDArray> list_out) {

// #pragma omp parallel for
  std::vector<bool> updated(graph->NumVertexTypes(), false);

  for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
    auto pair = graph->meta_graph()->FindEdge(etype);
    const dgl_id_t src_id = pair.first; // graph is reveresed
    const dgl_id_t dst_id = pair.second;
    if( src_id == 2) continue; // TODO (Israt): delete hard code
    if(updated[dst_id]) continue;
    updated[dst_id] = true;
    const DType* feat_data = list_feat[src_id].Ptr<DType>();
    const IdType* idx_data = list_idx[src_id].Ptr<IdType>();
    const IdType* idx_ntype_data = list_idx_ntypes[src_id].Ptr<IdType>();
    DType* out_data = list_out[dst_id].Ptr<DType>();

    int dim = 1;
    for (int i = 1; i < list_out[dst_id]->ndim; ++i)
      dim *= list_out[dst_id]->shape[i];
    int n = list_feat[src_id]->shape[0];

    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < dim; ++k) {
        if (dst_id == idx_ntype_data[i * dim + k]) {
          const int write_row = idx_data[i * dim + k];
          // #pragma omp atomic
          out_data[write_row * dim + k] += feat_data[i * dim + k]; // feat = dZ
        }
      }
    }
  }
}

/*!
 * \brief CPU kernel of backward phase of segment min/max.
 * \note math equation: out[arg[i, k], k] = feat[i, k]
 * \param feat The input tensor.
 * \param arg The argmin/argmax tensor.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  int n = feat->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* arg_data = arg.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
  runtime::parallel_for(0, n, [=](int b, int e) {
    for (auto i = b; i < e; ++i) {
      for (int k = 0; k < dim; ++k) {
        int write_row = arg_data[i * dim + k];
        if (write_row >= 0)
          out_data[write_row * dim + k] = feat_data[i * dim + k];
      }
    }
  });
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SEGMENT_REDUCE_H_
