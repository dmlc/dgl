/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampler/randomwalks_cpu.cc
 * \brief DGL sampler - CPU implementation of random walks with OpenMP
 */

#include <dgl/packed_func_ext.h>
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <dmlc/omp.h>
#include <utility>
#include "randomwalks.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

template<DLDeviceType XPU, typename IdxType>
TypeArray GetNodeTypesFromMetapath(
    const HeteroGraphPtr hg,
    const TypeArray metapath) {
  uint64_t num_etypes = metapath->shape[0];
  TypeArray result = TypeArray::Empty(
      {metapath->shape[0] + 1}, metapath->dtype, metapath->ctx);

  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);
  IdxType *result_data = static_cast<IdxType *>(result->data);

  dgl_type_t etype = metapath_data[0];
  dgl_type_t srctype = hg->GetEndpointTypes(etype).first;
  dgl_type_t curr_type = srctype;
  result_data[0] = curr_type;

  for (uint64_t i = 0; i < num_etypes; ++i) {
    etype = metapath_data[i];
    auto src_dst_type = hg->GetEndpointTypes(etype);
    dgl_type_t srctype = src_dst_type.first;
    dgl_type_t dsttype = src_dst_type.second;

    if (srctype != curr_type) {
      LOG(FATAL) << "source of edge type #" << i <<
        " does not match destination of edge type #" << i - 1;
      return result;
    }
    curr_type = dsttype;
    result_data[i + 1] = dsttype;
  }
  return result;
}

template
TypeArray GetNodeTypesFromMetapath<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const TypeArray metapath);
template
TypeArray GetNodeTypesFromMetapath<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const TypeArray metapath);

template<DLDeviceType XPU, typename IdxType>
IdArray RandomWalk(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob) {
  int64_t trace_length = metapath->shape[0] + 1;
  int64_t num_seeds = seeds->shape[0];
  IdArray traces = IdArray::Empty({num_seeds, trace_length}, seeds->dtype, seeds->ctx);

  const IdxType *metapath_data = static_cast<IdxType *>(metapath->data);
  const IdxType *seed_data = static_cast<IdxType *>(seeds->data);
  IdxType *traces_data = static_cast<IdxType *>(traces->data);

#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t t;
    for (t = 0; t < metapath->shape[0]; ++t) {
      dgl_type_t etype = metapath_data[t];
      const CSRMatrix &csr = hg->GetCSRAdj(etype, true);
      const IdxType *offsets = static_cast<IdxType *>(csr.indptr->data);
      const IdxType *all_succ = static_cast<IdxType *>(csr.indices->data);
      const IdxType *all_eids = static_cast<IdxType *>(csr.data->data);

#pragma omp critical
      {
        int i;
        std::cout << "---" << omp_get_thread_num() << "---" << std::endl;
        for (i = 0; i < csr.num_rows; ++i)
          std::cout << offsets[i] << ' ';
        std::cout << std::endl;
        for (i = 0; i < csr.indices->shape[0]; ++i)
          std::cout << all_succ[i] << ':' << all_eids[i] << ' ';
        std::cout << std::endl;
      }
    }
  }

#if 0
#pragma omp parallel for
  for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
    int64_t i;
    dgl_id_t curr = seed_data[seed_id];
    traces_data[seed_id * trace_length] = curr;

    for (i = 0; i < metapath->shape[0]; ++i) {
      dgl_type_t etype = metapath_data[i];

      // Get CSR of edge type.
      //
      // Note that the selection of successors is very lightweight (especially in the uniform
      // case), we want to reduce the overheads (even from object copies or object construction)
      // as much as possible.
      //
      // Using GetAdj() slows down by 1x due to returning std::vector with copies of NDArrays.
      // Using Successors() slows down by 2x.
      // Using OutEdges() slows down by 10x.
      const CSRMatrix &csr = hg->GetCSRAdj(etype, true);
      const IdxType *offsets = static_cast<IdxType *>(csr.indptr->data);
      const IdxType *all_succ = static_cast<IdxType *>(csr.indices->data);
      const IdxType *succ = all_succ + offsets[curr];

      int64_t size = offsets[curr + 1] - offsets[curr];
#pragma omp critical
      {
        std::cout << omp_get_thread_num() << ' ' << size << std::endl;
        for (int i = 0; i < csr.num_rows; ++i)
          std::cout << offsets[i] << ' ';
        std::cout << std::endl;
      }
      if (size == 0)
        // no successor, stop
        break;

      FloatArray prob_etype = prob[etype];
      if (prob_etype->shape[0] == 0) {
        // empty probability array; assume uniform
        IdxType idx = RandomEngine::ThreadLocal()->RandInt(size);
#pragma omp critical
        {
          std::cout << "A" << omp_get_thread_num() << ' ' << idx << std::endl;
        }
        curr = succ[idx];
      } else {
        const IdxType *all_eids = static_cast<IdxType *>(csr.data->data);
        const IdxType *eids = all_eids + offsets[curr];

        // non-uniform random walk
        ATEN_FLOAT_TYPE_SWITCH(prob_etype->dtype, DType, "probability", {
          const DType *prob_etype_data = static_cast<DType *>(prob_etype->data);
          std::vector<DType> prob_selected;
          for (int64_t j = 0; j < size; ++j)
            prob_selected.push_back(prob_etype_data[eids[j]]);

          curr = succ[RandomEngine::ThreadLocal()->Choice<int64_t>(prob_selected)];
        });
      }

      traces_data[seed_id * trace_length + i + 1] = curr;

      // restart probability
      if (restart_prob > 0 && RandomEngine::ThreadLocal()->Uniform<double>() < restart_prob)
        break;
    }

    // pad
    for (; i < metapath->shape[0]; ++i)
      traces_data[seed_id * trace_length + i + 1] = -1;
  }
#endif

  return traces;
}

template
IdArray RandomWalk<kDLCPU, int32_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);
template
IdArray RandomWalk<kDLCPU, int64_t>(
    const HeteroGraphPtr hg,
    const IdArray seeds,
    const TypeArray metapath,
    const std::vector<FloatArray> &prob,
    double restart_prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
