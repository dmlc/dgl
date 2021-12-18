/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cpu/negative_sampling.cc
 * \brief Uniform negative sampling on CSR.
 */

#include <dgl/array.h>
#include <dgl/runtime/parallel_for.h>
#include <dgl/random.h>
#include <utility>
#include <algorithm>

using namespace dgl::runtime;

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling(
    const CSRMatrix &csr,
    int64_t num_samples,
    int num_trials,
    bool exclude_self_loops) {
  const int64_t num_row = csr.num_rows;
  const int64_t num_col = csr.num_cols;
  IdArray row = Full<IdType>(-1, num_samples, csr.indptr->ctx);
  IdArray col = Full<IdType>(-1, num_samples, csr.indptr->ctx);
  IdType* row_data = row.Ptr<IdType>();
  IdType* col_data = col.Ptr<IdType>();
  parallel_for(0, num_samples, 1, [&](int64_t b, int64_t e) {
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

  IdType* row_data_end = std::remove_if(row_data, row_data + num_samples, [](IdType x) {
    return x == -1;
  });
  IdType* col_data_end = std::remove_if(col_data, col_data + num_samples, [](IdType x) {
    return x == -1;
  });
  IdArray row_filtered = row.CreateView({row_data_end - row_data}, row->dtype);
  IdArray col_filtered = col.CreateView({col_data_end - col_data}, col->dtype);
  return {row_filtered, col_filtered};
}

template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<kDLCPU, int32_t>(
    const CSRMatrix&, int64_t, int, bool);
template std::pair<IdArray, IdArray> CSRGlobalUniformNegativeSampling<kDLCPU, int64_t>(
    const CSRMatrix&, int64_t, int, bool);

};  // namespace impl
};  // namespace aten
};  // namespace dgl
