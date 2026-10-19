/**
 *  Copyright (c) 2024 by Contributors
 * @file partition_op.cc
 * @brief Operations on partition implemented for Ascend NPU.
 * 
 * This implementation uses CPU fallback with data transfer between NPU and CPU.
 * For optimal performance, use AscendNDArrayPartitionWrapper from hccl.py.
 */

#ifdef DGL_USE_ASCEND

#include <dgl/runtime/device_api.h>
#include <dgl/runtime/c_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../../runtime/workspace.h"
#include "../partition_op.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>

#define ASCEND_CALL(func)                                                \
  {                                                                      \
    aclError e = (func);                                                 \
    CHECK(e == ACL_SUCCESS) << "Ascend Error: " << aclGetRecentErrMsg(); \
  }

namespace dgl {
namespace partition {
namespace impl {

namespace {

template<typename IdType>
void MapProcByRemainderCpu(
    const IdType* global, const int64_t num_elements,
    const int64_t num_parts, IdType* part_id) {
  for (int64_t i = 0; i < num_elements; ++i) {
    part_id[i] = global[i] % num_parts;
  }
}

template<typename IdType>
void MapLocalIndexByRemainderCpu(
    const IdType* global, const int64_t num_elements, const int num_parts,
    IdType* local) {
  for (int64_t i = 0; i < num_elements; ++i) {
    local[i] = global[i] / num_parts;
  }
}

template<typename IdType>
void MapGlobalIndexByRemainderCpu(
    const IdType* local, const int part_id, const int64_t num_elements,
    const int num_parts, IdType* global) {
  for (int64_t i = 0; i < num_elements; ++i) {
    global[i] = (local[i] * num_parts) + part_id;
  }
}

template<typename RangeType>
int SearchRangeCpu(
    const RangeType* range, const int num_parts, const RangeType target) {
  int start = 0;
  int end = num_parts;
  int cur = (end + start) / 2;

  while (start + 1 < end) {
    if (target < range[cur]) {
      end = cur;
    } else {
      start = cur;
    }
    cur = (start + end) / 2;
  }

  return cur;
}

template<typename IdType, typename RangeType>
void MapProcByRangeCpu(
    const RangeType* range, const IdType* global,
    const int64_t num_elements, const int64_t num_parts,
    IdType* part_id) {
  for (int64_t i = 0; i < num_elements; ++i) {
    part_id[i] = static_cast<IdType>(SearchRangeCpu(
        range, static_cast<int>(num_parts),
        static_cast<RangeType>(global[i])));
  }
}

template<typename IdType, typename RangeType>
void MapLocalIndexByRangeCpu(
    const RangeType* range, const IdType* global,
    const int64_t num_elements, const int num_parts, IdType* local) {
  for (int64_t i = 0; i < num_elements; ++i) {
    const int proc = SearchRangeCpu(
        range, static_cast<int>(num_parts),
        static_cast<RangeType>(global[i]));
    local[i] = global[i] - range[proc];
  }
}

template<typename IdType, typename RangeType>
void MapGlobalIndexByRangeCpu(
    const RangeType* range, const IdType* local, const int part_id,
    const int64_t num_elements, const int num_parts, IdType* global) {
  for (int64_t i = 0; i < num_elements; ++i) {
    global[i] = local[i] + range[part_id];
  }
}

}  // namespace

template <DGLDeviceType XPU, typename IdType>
std::pair<IdArray, NDArray> GeneratePermutationFromRemainder(
    int64_t array_size, int num_parts, IdArray in_idx) {
  std::pair<IdArray, NDArray> result;

  const auto& ctx = in_idx->ctx;
  const int64_t num_in = in_idx->shape[0];

  CHECK_GE(num_parts, 1) << "The number of partitions (" << num_parts
                         << ") must be at least 1.";
  if (num_parts == 1) {
    result.first = aten::Range(0, num_in, sizeof(IdType) * 8, ctx);
    result.second = aten::Full(num_in, num_parts, sizeof(int64_t) * 8, ctx);
    return result;
  }

  result.first = aten::NewIdArray(num_in, ctx, sizeof(IdType) * 8);
  result.second = aten::Full(0, num_parts, sizeof(int64_t) * 8, ctx);
  int64_t* out_counts = static_cast<int64_t*>(result.second->data);
  if (num_in == 0) {
    return result;
  }

  DGLContext cpu_ctx{kDGLCPU, 0};
  IdArray in_idx_cpu = in_idx.CopyTo(cpu_ctx);
  IdArray proc_id_cpu = aten::NewIdArray(num_in, cpu_ctx, sizeof(IdType) * 8);
  IdArray perm_out_cpu = aten::NewIdArray(num_in, cpu_ctx, sizeof(IdType) * 8);
  IdArray perm_in_cpu = aten::Range(0, num_in, sizeof(IdType) * 8, cpu_ctx);

  const IdType* in_idx_data = static_cast<const IdType*>(in_idx_cpu->data);
  IdType* proc_id_data = static_cast<IdType*>(proc_id_cpu->data);
  IdType* perm_out_data = static_cast<IdType*>(perm_out_cpu->data);
  IdType* perm_in_data = static_cast<IdType*>(perm_in_cpu->data);

  MapProcByRemainderCpu(in_idx_data, num_in, num_parts, proc_id_data);

  for (int64_t i = 0; i < num_in; ++i) {
    perm_in_data[i] = i;
  }

  std::vector<std::pair<IdType, IdType>> pairs(num_in);
  for (int64_t i = 0; i < num_in; ++i) {
    pairs[i] = {proc_id_data[i], perm_in_data[i]};
  }
  std::stable_sort(pairs.begin(), pairs.end(), 
    [](const auto& a, const auto& b) { return a.first < b.first; });
  for (int64_t i = 0; i < num_in; ++i) {
    proc_id_data[i] = pairs[i].first;
    perm_out_data[i] = pairs[i].second;
  }

  for (int p = 0; p < num_parts; ++p) {
    out_counts[p] = 0;
  }
  for (int64_t i = 0; i < num_in; ++i) {
    out_counts[proc_id_data[i]]++;
  }

  result.first = perm_out_cpu.CopyTo(ctx);
  result.second = result.second.CopyTo(ctx);

  return result;
}

template std::pair<IdArray, IdArray> GeneratePermutationFromRemainder<
    kDGLAscend, int32_t>(int64_t array_size, int num_parts, IdArray in_idx);
template std::pair<IdArray, IdArray> GeneratePermutationFromRemainder<
    kDGLAscend, int64_t>(int64_t array_size, int num_parts, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType>
IdArray MapToLocalFromRemainder(int num_parts, IdArray global_idx) {
  const auto& ctx = global_idx->ctx;

  if (num_parts > 1) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    IdArray global_idx_cpu = global_idx.CopyTo(cpu_ctx);
    IdArray local_idx_cpu = aten::NewIdArray(global_idx->shape[0], cpu_ctx, sizeof(IdType) * 8);

    const IdType* global_data = static_cast<const IdType*>(global_idx_cpu->data);
    IdType* local_data = static_cast<IdType*>(local_idx_cpu->data);

    MapLocalIndexByRemainderCpu(global_data, global_idx->shape[0], num_parts, local_data);

    return local_idx_cpu.CopyTo(ctx);
  } else {
    return global_idx;
  }
}

template IdArray MapToLocalFromRemainder<kDGLAscend, int32_t>(
    int num_parts, IdArray in_idx);
template IdArray MapToLocalFromRemainder<kDGLAscend, int64_t>(
    int num_parts, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType>
IdArray MapToGlobalFromRemainder(
    int num_parts, IdArray local_idx, const int part_id) {
  CHECK_LT(part_id, num_parts)
      << "Invalid partition id " << part_id << "/" << num_parts;
  CHECK_GE(part_id, 0) << "Invalid partition id " << part_id << "/"
                       << num_parts;

  const auto& ctx = local_idx->ctx;

  if (num_parts > 1) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    IdArray local_idx_cpu = local_idx.CopyTo(cpu_ctx);
    IdArray global_idx_cpu = aten::NewIdArray(local_idx->shape[0], cpu_ctx, sizeof(IdType) * 8);

    const IdType* local_data = static_cast<const IdType*>(local_idx_cpu->data);
    IdType* global_data = static_cast<IdType*>(global_idx_cpu->data);

    MapGlobalIndexByRemainderCpu(local_data, part_id, local_idx->shape[0], num_parts, global_data);

    return global_idx_cpu.CopyTo(ctx);
  } else {
    return local_idx;
  }
}

template IdArray MapToGlobalFromRemainder<kDGLAscend, int32_t>(
    int num_parts, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRemainder<kDGLAscend, int64_t>(
    int num_parts, IdArray in_idx, int part_id);

template <DGLDeviceType XPU, typename IdType, typename RangeType>
std::pair<IdArray, NDArray> GeneratePermutationFromRange(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx) {
  std::pair<IdArray, NDArray> result;

  const auto& ctx = in_idx->ctx;
  const int64_t num_in = in_idx->shape[0];

  CHECK_GE(num_parts, 1) << "The number of partitions (" << num_parts
                         << ") must be at least 1.";
  if (num_parts == 1) {
    result.first = aten::Range(0, num_in, sizeof(IdType) * 8, ctx);
    result.second = aten::Full(num_in, num_parts, sizeof(int64_t) * 8, ctx);
    return result;
  }

  result.first = aten::NewIdArray(num_in, ctx, sizeof(IdType) * 8);
  result.second = aten::Full(0, num_parts, sizeof(int64_t) * 8, ctx);
  int64_t* out_counts = static_cast<int64_t*>(result.second->data);
  if (num_in == 0) {
    return result;
  }

  DGLContext cpu_ctx{kDGLCPU, 0};
  IdArray range_cpu = range.CopyTo(cpu_ctx);
  IdArray in_idx_cpu = in_idx.CopyTo(cpu_ctx);
  IdArray proc_id_cpu = aten::NewIdArray(num_in, cpu_ctx, sizeof(IdType) * 8);
  IdArray perm_out_cpu = aten::NewIdArray(num_in, cpu_ctx, sizeof(IdType) * 8);
  IdArray perm_in_cpu = aten::Range(0, num_in, sizeof(IdType) * 8, cpu_ctx);

  const RangeType* range_data = static_cast<const RangeType*>(range_cpu->data);
  const IdType* in_idx_data = static_cast<const IdType*>(in_idx_cpu->data);
  IdType* proc_id_data = static_cast<IdType*>(proc_id_cpu->data);
  IdType* perm_out_data = static_cast<IdType*>(perm_out_cpu->data);
  IdType* perm_in_data = static_cast<IdType*>(perm_in_cpu->data);

  MapProcByRangeCpu(range_data, in_idx_data, num_in, num_parts, proc_id_data);

  for (int64_t i = 0; i < num_in; ++i) {
    perm_in_data[i] = i;
  }

  std::vector<std::pair<IdType, IdType>> pairs(num_in);
  for (int64_t i = 0; i < num_in; ++i) {
    pairs[i] = {proc_id_data[i], perm_in_data[i]};
  }
  std::stable_sort(pairs.begin(), pairs.end(), 
    [](const auto& a, const auto& b) { return a.first < b.first; });
  for (int64_t i = 0; i < num_in; ++i) {
    proc_id_data[i] = pairs[i].first;
    perm_out_data[i] = pairs[i].second;
  }

  for (int p = 0; p < num_parts; ++p) {
    out_counts[p] = 0;
  }
  for (int64_t i = 0; i < num_in; ++i) {
    out_counts[proc_id_data[i]]++;
  }

  result.first = perm_out_cpu.CopyTo(ctx);
  result.second = result.second.CopyTo(ctx);

  return result;
}

template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLAscend, int32_t, int32_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLAscend, int64_t, int32_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLAscend, int32_t, int64_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);
template std::pair<IdArray, IdArray>
GeneratePermutationFromRange<kDGLAscend, int64_t, int64_t>(
    int64_t array_size, int num_parts, IdArray range, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType, typename RangeType>
IdArray MapToLocalFromRange(
    const int num_parts, IdArray range, IdArray global_idx) {
  const auto& ctx = global_idx->ctx;

  if (num_parts > 1 && global_idx->shape[0] > 0) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    IdArray range_cpu = range.CopyTo(cpu_ctx);
    IdArray global_idx_cpu = global_idx.CopyTo(cpu_ctx);
    IdArray local_idx_cpu = aten::NewIdArray(global_idx->shape[0], cpu_ctx, sizeof(IdType) * 8);

    const RangeType* range_data = static_cast<const RangeType*>(range_cpu->data);
    const IdType* global_data = static_cast<const IdType*>(global_idx_cpu->data);
    IdType* local_data = static_cast<IdType*>(local_idx_cpu->data);

    MapLocalIndexByRangeCpu(range_data, global_data, global_idx->shape[0], num_parts, local_data);

    return local_idx_cpu.CopyTo(ctx);
  } else {
    return global_idx;
  }
}

template IdArray MapToLocalFromRange<kDGLAscend, int32_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx);
template IdArray MapToLocalFromRange<kDGLAscend, int64_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx);
template IdArray MapToLocalFromRange<kDGLAscend, int32_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx);
template IdArray MapToLocalFromRange<kDGLAscend, int64_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx);

template <DGLDeviceType XPU, typename IdType, typename RangeType>
IdArray MapToGlobalFromRange(
    const int num_parts, IdArray range, IdArray local_idx, const int part_id) {
  CHECK_LT(part_id, num_parts)
      << "Invalid partition id " << part_id << "/" << num_parts;
  CHECK_GE(part_id, 0) << "Invalid partition id " << part_id << "/"
                       << num_parts;

  const auto& ctx = local_idx->ctx;

  if (num_parts > 1 && local_idx->shape[0] > 0) {
    DGLContext cpu_ctx{kDGLCPU, 0};
    IdArray range_cpu = range.CopyTo(cpu_ctx);
    IdArray local_idx_cpu = local_idx.CopyTo(cpu_ctx);
    IdArray global_idx_cpu = aten::NewIdArray(local_idx->shape[0], cpu_ctx, sizeof(IdType) * 8);

    const RangeType* range_data = static_cast<const RangeType*>(range_cpu->data);
    const IdType* local_data = static_cast<const IdType*>(local_idx_cpu->data);
    IdType* global_data = static_cast<IdType*>(global_idx_cpu->data);

    MapGlobalIndexByRangeCpu(range_data, local_data, part_id, local_idx->shape[0], num_parts, global_data);

    return global_idx_cpu.CopyTo(ctx);
  } else {
    return local_idx;
  }
}

template IdArray MapToGlobalFromRange<kDGLAscend, int32_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRange<kDGLAscend, int64_t, int32_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRange<kDGLAscend, int32_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);
template IdArray MapToGlobalFromRange<kDGLAscend, int64_t, int64_t>(
    int num_parts, IdArray range, IdArray in_idx, int part_id);

}  // namespace impl
}  // namespace partition
}  // namespace dgl

#endif  // DGL_USE_ASCEND
