/**
 *  Copyright (c) 2023 by Contributors
 * @file cpu/matrix_ops_impl_cpu.cc
 * @brief DGL C++ implementation of matrix operators.
 */
#include "../matrix_ops_impl.h"
#include "../utils.h"

namespace dgl {
namespace sparse {

template <c10::DeviceType XPU, typename IdType>
std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor>
COOIntersectionImpl(
    const std::shared_ptr<COO> &lhs, const std::shared_ptr<COO> &rhs) {
  const int64_t lhs_len = lhs->indices.size(1);
  const int64_t rhs_len = rhs->indices.size(1);

  torch::Tensor lhs_row = lhs->indices.index({0}).contiguous();
  const IdType *lhs_row_ptr = static_cast<IdType *>(lhs_row.data_ptr());
  torch::Tensor lhs_col = lhs->indices.index({1}).contiguous();
  const IdType *lhs_col_ptr = static_cast<IdType *>(lhs_col.data_ptr());
  torch::Tensor rhs_row = rhs->indices.index({0}).contiguous();
  const IdType *rhs_row_ptr = static_cast<IdType *>(rhs_row.data_ptr());
  torch::Tensor rhs_col = rhs->indices.index({1}).contiguous();
  const IdType *rhs_col_ptr = static_cast<IdType *>(rhs_col.data_ptr());

  std::vector<IdType> ret_rows, ret_cols;
  std::vector<IdType> ret_lhs_indices, ret_rhs_indices;

  auto pair_hash_fn = [](const std::pair<IdType, IdType> &pair) {
    return std::hash<IdType>()(pair.first) ^ std::hash<IdType>()(pair.second);
  };
  std::unordered_map<std::pair<IdType, IdType>, IdType, decltype(pair_hash_fn)>
      pair_map(lhs_len, pair_hash_fn);
  for (int64_t k = 0; k < lhs_len; ++k)
    pair_map.emplace(std::make_pair(lhs_row_ptr[k], lhs_col_ptr[k]), k);

  for (int64_t i = 0; i < rhs_len; ++i) {
    const IdType row_id = rhs_row_ptr[i], col_id = rhs_col_ptr[i];
    auto it = pair_map.find({row_id, col_id});
    if (it != pair_map.end()) {
      ret_rows.push_back(row_id);
      ret_cols.push_back(col_id);
      ret_lhs_indices.push_back(it->second);
      ret_rhs_indices.push_back(i);
    }
  }
  auto ret_indices = torch::stack(
      {VectorToTorchTensor(ret_rows), VectorToTorchTensor(ret_cols)});
  auto ret_coo = std::make_shared<COO>(
      COO{lhs->num_rows, lhs->num_cols, ret_indices, rhs->row_sorted,
          rhs->col_sorted});
  return {
      ret_coo, VectorToTorchTensor(ret_lhs_indices),
      VectorToTorchTensor(ret_rhs_indices)};
}

template std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor>
COOIntersectionImpl<c10::DeviceType::CPU, int32_t>(
    const std::shared_ptr<COO> &lhs, const std::shared_ptr<COO> &rhs);
template std::tuple<std::shared_ptr<COO>, torch::Tensor, torch::Tensor>
COOIntersectionImpl<c10::DeviceType::CPU, int64_t>(
    const std::shared_ptr<COO> &lhs, const std::shared_ptr<COO> &rhs);

}  // namespace sparse
}  // namespace dgl
