/*!
 *  Copyright (c) 2018 by Contributors
 * \file kernel/utils.h
 * \brief Kernel utilities
 */
#ifndef DGL_KERNEL_UTILS_H_
#define DGL_KERNEL_UTILS_H_

#include <minigun/spmat.h>
#include <minigun/minigun.h>
#include <dlpack/dlpack.h>
#include <dgl/array.h>
#include <dgl/runtime/ndarray.h>

#include <cstdlib>
#include <vector>

namespace dgl {
namespace kernel {
namespace utils {

/*
 * !\brief Find number of threads is smaller than dim and max_nthrs
 * and is also the power of two.
 */
int FindNumThreads(int dim, int max_nthrs);

/*
 * !\brief Compute the total number of feature elements.
 */
int64_t ComputeXLength(runtime::NDArray feat_array);

/*
 * !\brief Compute the total number of elements in the array.
 */
int64_t NElements(const runtime::NDArray& array);

/*
 * !\brief Compute the product of the given vector.
 */
int64_t Prod(const std::vector<int64_t>& vec);

/*
 * !\brief Compute the edge mapping given mapping and edge index tensor.
 */
template <typename Idx>
inline void ComputeEdgeMapping(Idx **cur_mapping, runtime::NDArray cur, runtime::NDArray eids) {
  if (*cur_mapping == nullptr) {
    if (!aten::IsNullArray(eids))
      *cur_mapping = static_cast<Idx*>(eids->data);
  } else {
    runtime::NDArray out_map = aten::MergeIDMapping(eids, cur);
    *cur_mapping = static_cast<Idx*>(out_map->data);
  }
}

template void ComputeEdgeMapping<int>(int **cur_mapping, runtime::NDArray cur, runtime::NDArray eids);
template void ComputeEdgeMapping<long long>(long long **cur_mapping, runtime::NDArray cur, runtime::NDArray eids);

/*
 * !\brief Fill the array with constant value.
 */
template <int XPU, typename DType>
void Fill(const DLContext& ctx, DType* ptr, size_t length, DType val);

/*
 * !\brief Create minigun CSR from two ndarrays.
 */
template <typename Idx>
minigun::Csr<Idx> CreateCsr(runtime::NDArray indptr, runtime::NDArray indices) {
  minigun::Csr<Idx> csr;
  csr.row_offsets.data = static_cast<Idx*>(indptr->data);
  csr.row_offsets.length = indptr->shape[0];
  csr.column_indices.data = static_cast<Idx*>(indices->data);
  csr.column_indices.length = indices->shape[0];
  return csr;
}

/*
 * !\brief Create minigun COO from two ndarrays.
 */
template <typename Idx>
minigun::Coo<Idx> CreateCoo(runtime::NDArray row, runtime::NDArray col) {
  minigun::Coo<Idx> coo;
  coo.row.data = static_cast<Idx*>(row->data);
  coo.row.length = row->shape[0];
  coo.column.data = static_cast<Idx*>(col->data);
  coo.column.length = col->shape[0];
  return coo;
}

typedef minigun::advance::Config<minigun::advance::kSrc> AdvanceSrcConfig;
typedef minigun::advance::Config<minigun::advance::kEdge> AdvanceEdgeConfig;
typedef minigun::advance::Config<minigun::advance::kDst> AdvanceDstConfig;

#define CREATE_IN_CSR(spmat, eid_data) do {                                         \
  auto incsr = graph.GetInCSRMatrix();                                              \
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(incsr.indptr, incsr.indices);       \
  (spmat).in_csr = &csr;                                                            \
  (eid_data) = &(incsr.data);                                                       \
} while(0)

#define CREATE_OUT_CSR(spmat, eid_data) do {                                        \
  auto outcsr = graph.GetOutCSRMatrix();                                            \
  minigun::Csr<Idx> csr = utils::CreateCsr<Idx>(outcsr.indptr, outcsr.indices);     \
  (spmat).out_csr = &csr;                                                           \
  (eid_data) = &(outcsr.data);                                                      \
} while(0)

#define CREATE_COO(spmat, eid_data) do {                                            \
  LOG(INFO) << "STAGE 0";                                                           \
  auto coo_matrix = graph.GetCOOMatrix();                                           \
  minigun::Coo<Idx> coo = utils::CreateCoo<Idx>(coo_matrix.row, coo_matrix.col);    \
  (spmat).coo = &coo;                                                               \
  (eid_data) = &(coo_matrix.data);                                                  \
} while(0)

#define ADVANCE_DISPATCH(graph, AtomicUDF, NonAtomicUDF, out_target, GDataType) do {\
  SparseFormat fmt = (graph).GetRestrictFormat();                                   \
  minigun::SpMat<Idx> spmat = {nullptr, nullptr, nullptr};                          \
  bool atomic = false;                                                              \
  minigun::advance::ParallelMode parallel_mode;                                     \
  IdArray *eid_data = nullptr;                                                      \
  if (IsCscAvailable(fmt) && (out_target) == binary_op::kDst) {                     \
    CREATE_IN_CSR(spmat, eid_data);                                                 \
    parallel_mode = minigun::advance::ParallelMode::kDst;                           \
  } else if (IsCsrAvailable(fmt) && (out_target) == binary_op::kSrc) {              \
    CREATE_OUT_CSR(spmat, eid_data);                                                \
    parallel_mode = minigun::advance::ParallelMode::kSrc;                           \
  } else {                                                                          \
    atomic = true;                                                                  \
    parallel_mode = minigun::advance::ParallelMode::kEdge;                          \
    if (IsCooAvailable(fmt)) {                                                      \
      if ((out_target) == binary_op::kEdge)                                         \
        atomic = false;                                                             \
      CREATE_COO(spmat, eid_data);                                                  \
    } else if (IsCscAvailable(fmt)) {                                               \
      CREATE_IN_CSR(spmat, eid_data);                                               \
    } else if (IsCsrAvailable(fmt)) {                                               \
      CREATE_OUT_CSR(spmat, eid_data);                                              \
    }                                                                               \
  }                                                                                 \
  LOG(INFO) << "STAGE 1";                                              \
  if (LeftSelector::target == binary_op::kEdge)                                     \
    utils::ComputeEdgeMapping<Idx>(&(gdata->lhs_mapping), gdata->lhs, *eid_data);   \
  if (RightSelector::target == binary_op::kEdge)                                    \
    utils::ComputeEdgeMapping<Idx>(&(gdata->rhs_mapping), gdata->rhs, *eid_data);   \
  if (OutSelector<Reducer>::Type::target == binary_op::kEdge)                       \
    utils::ComputeEdgeMapping<Idx>(&(gdata->out_mapping), gdata->out, *eid_data);   \
  LOG(INFO) << "STAGE 2";                                             \
  if (atomic) {                                                                     \
    switch (parallel_mode) {                                                        \
      case minigun::advance::ParallelMode::kEdge:                                   \
        minigun::advance::Advance<XPU, Idx, DType, utils::AdvanceEdgeConfig,        \
          GDataType, AtomicUDF>(rtcfg, spmat, gdata);                               \
        break;                                                                      \
      case minigun::advance::ParallelMode::kDst:                                    \
        minigun::advance::Advance<XPU, Idx, DType, utils::AdvanceDstConfig,         \
          GDataType, AtomicUDF>(rtcfg, spmat, gdata);                               \
        break;                                                                      \
      case minigun::advance::ParallelMode::kSrc:                                    \
        minigun::advance::Advance<XPU, Idx, DType, utils::AdvanceSrcConfig,         \
          GDataType, AtomicUDF>(rtcfg, spmat, gdata);                               \
        break;                                                                      \
    }                                                                               \
  } else {                                                                          \
    switch (parallel_mode) {                                                        \
      case minigun::advance::ParallelMode::kEdge:                                   \
        minigun::advance::Advance<XPU, Idx, DType, utils::AdvanceEdgeConfig,        \
          GDataType, NonAtomicUDF>(rtcfg, spmat, gdata);                            \
        break;                                                                      \
      case minigun::advance::ParallelMode::kDst:                                    \
        minigun::advance::Advance<XPU, Idx, DType, utils::AdvanceDstConfig,         \
          GDataType, NonAtomicUDF>(rtcfg, spmat, gdata);                            \
        break;                                                                      \
      case minigun::advance::ParallelMode::kSrc:                                    \
        minigun::advance::Advance<XPU, Idx, DType, utils::AdvanceSrcConfig,         \
          GDataType, NonAtomicUDF>(rtcfg, spmat, gdata);                            \
        break;                                                                      \
    }                                                                               \
  }                                                                                 \
} while(0)

}  // namespace utils
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_UTILS_H_
