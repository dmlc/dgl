/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cpu/negative_sampling.cc
 * @brief Uniform negative sampling on CSR.
 */

#include <dgl/array.h>
#include <dgl/array_iterator.h>
#include <dgl/random.h>
#include <dgl/runtime/parallel_for.h>

#include <algorithm>
#include <utility>

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace impl {

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling(
    const CSRMatrix& csr, int64_t num_samples, int num_trials,
    bool exclude_self_loops, bool replace, double redundancy) {
  const int64_t num_row = csr.num_rows;
  const int64_t num_col = csr.num_cols;
  const int64_t num_actual_samples =
      static_cast<int64_t>(num_samples * (1 + redundancy));
  IdArray row = Full<IdType>(-1, num_actual_samples, csr.indptr->ctx);
  IdArray col = Full<IdType>(-1, num_actual_samples, csr.indptr->ctx);
  IdType* row_data = row.Ptr<IdType>();
  IdType* col_data = col.Ptr<IdType>();

  parallel_for(0, num_actual_samples, 1, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      for (int trial = 0; trial < num_trials; ++trial) {
        IdType u = RandomEngine::ThreadLocal()->RandInt(num_row);
        IdType v = RandomEngine::ThreadLocal()->RandInt(num_col);
        if (!(exclude_self_loops && (u == v)) && !CSRIsNonZero(csr, u, v)) {
          row_data[i] = u;
          col_data[i] = v;
          break;
        }
      }
    }
  });

  PairIterator<IdType> begin(row_data, col_data);
  PairIterator<IdType> end = std::remove_if(
      begin, begin + num_actual_samples,
      [](const std::pair<IdType, IdType>& val) { return val.first == -1; });
  if (!replace) {
    std::sort(
        begin, end,
        [](const std::pair<IdType, IdType>& a,
           const std::pair<IdType, IdType>& b) {
          return a.first < b.first ||
                 (a.first == b.first && a.second < b.second);
        });
    end = std::unique(begin, end);
  }
  int64_t num_sampled =
      std::min(static_cast<int64_t>(end - begin), num_samples);
  return {
      row.CreateView({num_sampled}, row->dtype),
      col.CreateView({num_sampled}, col->dtype)};
}

template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<
    kDGLCPU, int32_t>(const CSRMatrix&, int64_t, int, bool, bool, double);
template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<
    kDGLCPU, int64_t>(const CSRMatrix&, int64_t, int, bool, bool, double);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
