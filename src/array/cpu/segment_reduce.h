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
#include <vector>
#include <string>

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
 * \param graph The input heterogeneous graph.
 * \param op The binary operator, could be `copy_u`, `copy_e'.
 * \param list_feat List of the input tensors.
 * \param list_idx  List of the indices tensors.
 * \param list_idx_etype List of the node- or edge-type tensors.
 * \param list_out List of the output tensors.
 */
template <typename IdType, typename DType>
void UpdateGradMinMax_hetero(HeteroGraphPtr graph,
                       const std::string& op,
                       const std::vector<NDArray>& list_feat,
                       const std::vector<NDArray>& list_idx,
                       const std::vector<NDArray>& list_idx_ntypes,
                       std::vector<NDArray>* list_out) {
  if (op == "copy_lhs") {
    std::vector<std::vector<dgl_id_t>> dst_src_ntids(graph->NumVertexTypes(),
    std::vector<dgl_id_t>());
    for (dgl_type_t etype = 0; etype < graph->NumEdgeTypes(); ++etype) {
      auto pair = graph->meta_graph()->FindEdge(etype);
      const dgl_id_t dst_id = pair.first;  // graph is reversed
      const dgl_id_t src_id = pair.second;
      dst_src_ntids[dst_id].push_back(src_id);  // can have duplicates. Use Hashtable to optimize.
    }
    std::vector<bool> updated(graph->NumVertexTypes());
    for (int dst_id = 0; dst_id < dst_src_ntids.size(); ++dst_id) {
      std::fill(updated.begin(), updated.end(), false);
      for (int j = 0; j < dst_src_ntids[dst_id].size(); ++j) {
        int src_id = dst_src_ntids[dst_id][j];
        if (updated[src_id]) continue;
        const DType* feat_data = list_feat[dst_id].Ptr<DType>();
        const IdType* idx_data = list_idx[dst_id].Ptr<IdType>();
        const IdType* idx_ntype_data = list_idx_ntypes[dst_id].Ptr<IdType>();
        DType* out_data = (*list_out)[src_id].Ptr<DType>();
        int dim = 1;
        for (int i = 1; i < (*list_out)[src_id]->ndim; ++i)
          dim *= (*list_out)[src_id]->shape[i];
        int n = list_feat[dst_id]->shape[0];
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
          for (int k = 0; k < dim; ++k) {
            if (src_id == idx_ntype_data[i * dim + k]) {
              const int write_row = idx_data[i * dim + k];
#pragma omp atomic
              out_data[write_row * dim + k] += feat_data[i * dim + k];  // feat = dZ
            }
          }
        }
        updated[src_id] = true;
      }
    }
  } else if (op == "copy_rhs") {
    for (dgl_type_t etid = 0; etid < graph->NumEdgeTypes(); ++etid) {
      auto pair = graph->meta_graph()->FindEdge(etid);
      const dgl_id_t dst_id = pair.first;  // graph is reversed
      const dgl_id_t src_id = pair.second;
      const DType* feat_data = list_feat[dst_id].Ptr<DType>();
      const IdType* idx_data = list_idx[dst_id].Ptr<IdType>();
      const IdType* idx_ntype_data = list_idx_ntypes[dst_id].Ptr<IdType>();
      DType* out_data = (*list_out)[etid].Ptr<DType>();
      int dim = 1;
      for (int i = 1; i < (*list_out)[etid]->ndim; ++i)
        dim *= (*list_out)[etid]->shape[i];
      int n = list_feat[dst_id]->shape[0];
#pragma omp parallel for
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < dim; ++k) {
          if (etid == idx_ntype_data[i * dim + k]) {
            const int write_row = idx_data[i * dim + k];
#pragma omp atomic
            out_data[write_row * dim + k] += feat_data[i * dim + k];  // feat = dZ
          }
        }
      }
    }
  } else {
    LOG(FATAL) << "Unsupported binary operator: " << op;
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
