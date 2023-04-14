/**
 *  Copyright (c) 2020 by Contributors
 * @file array/cpu/spmm.h
 * @brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SEGMENT_REDUCE_H_
#define DGL_ARRAY_CPU_SEGMENT_REDUCE_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/parallel_for.h>

#include <string>
#include <vector>

namespace dgl {
namespace aten {
namespace cpu {

/**
 * @brief CPU kernel of segment sum.
 * @param feat The input tensor.
 * @param offsets The offset tensor storing the ranges of segments.
 * @param out The output tensor.
 */
template <typename IdType, typename DType>
void SegmentSum(NDArray feat, NDArray offsets, NDArray out) {
  if (std::is_same<DType, BFloat16>::value)
    LOG(FATAL) << "Unsupported CPU kernel for SegmentSum for BF16.";
  int n = out->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
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

/**
 * @brief CPU kernel of segment min/max.
 * @param feat The input tensor.
 * @param offsets The offset tensor storing the ranges of segments.
 * @param out The output tensor.
 * @param arg An auxiliary tensor storing the argmin/max information
 *        used in backward phase.
 */
template <typename IdType, typename DType, typename Cmp>
void SegmentCmp(NDArray feat, NDArray offsets, NDArray out, NDArray arg) {
  int n = out->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
  IdType* arg_data = arg.Ptr<IdType>();
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

/**
 * @brief CPU kernel of Scatter Add (on first dimension) operator.
 * @note math equation: out[idx[i], *] += feat[i, *]
 * @param feat The input tensor.
 * @param idx The indices tensor.
 * @param out The output tensor.
 */
template <typename IdType, typename DType>
void ScatterAdd(NDArray feat, NDArray idx, NDArray out) {
  int n = feat->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];
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

/**
 * @brief CPU kernel to update gradients for reduce op max/min
 * @param graph The input heterogeneous graph.
 * @param op The binary operator, could be `copy_u`, `copy_e'.
 * @param list_feat List of the input tensors.
 * @param list_idx  List of the indices tensors.
 * @param list_idx_etype List of the node- or edge-type tensors.
 * @param list_out List of the output tensors.
 */
template <typename IdType, typename DType>
void UpdateGradMinMax_hetero(
    HeteroGraphPtr graph, const std::string& op,
    const std::vector<NDArray>& list_feat, const std::vector<NDArray>& list_idx,
    const std::vector<NDArray>& list_idx_types,
    std::vector<NDArray>* list_out) {
  if (op == "copy_lhs" || op == "copy_rhs") {
    std::vector<std::vector<dgl_id_t>> src_dst_ntypes(
        graph->NumVertexTypes(), std::vector<dgl_id_t>());

    for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
      auto pair = graph->meta_graph()->FindEdge(etype);
      const dgl_id_t dst_ntype = pair.first;  // graph is reversed
      const dgl_id_t src_ntype = pair.second;
      auto same_src_dst_ntype = std::find(
          std::begin(src_dst_ntypes[dst_ntype]),
          std::end(src_dst_ntypes[dst_ntype]), src_ntype);
      // if op is "copy_lhs", relation type with same src and dst node type will
      // be updated once
      if (op == "copy_lhs" &&
          same_src_dst_ntype != std::end(src_dst_ntypes[dst_ntype]))
        continue;
      src_dst_ntypes[dst_ntype].push_back(src_ntype);
      const DType* feat_data = list_feat[dst_ntype].Ptr<DType>();
      const IdType* idx_data = list_idx[dst_ntype].Ptr<IdType>();
      const IdType* idx_type_data = list_idx_types[dst_ntype].Ptr<IdType>();
      int type = (op == "copy_lhs") ? src_ntype : etype;
      DType* out_data = (*list_out)[type].Ptr<DType>();
      int dim = 1;
      for (int i = 1; i < (*list_out)[type]->ndim; ++i)
        dim *= (*list_out)[type]->shape[i];
      int n = list_feat[dst_ntype]->shape[0];
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < dim; ++k) {
          if (type == idx_type_data[i * dim + k]) {
            const int write_row = idx_data[i * dim + k];
#pragma omp atomic
            out_data[write_row * dim + k] +=
                feat_data[i * dim + k];  // feat = dZ
          }
        }
      }
    }
  } else {
    LOG(FATAL) << "Unsupported binary operator: " << op;
  }
}

/**
 * @brief CPU kernel of backward phase of segment min/max.
 * @note math equation: out[arg[i, k], k] = feat[i, k]
 * @param feat The input tensor.
 * @param arg The argmin/argmax tensor.
 * @param out The output tensor.
 */
template <typename IdType, typename DType>
void BackwardSegmentCmp(NDArray feat, NDArray arg, NDArray out) {
  int n = feat->shape[0];
  int dim = 1;
  for (int i = 1; i < out->ndim; ++i) dim *= out->shape[i];
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
