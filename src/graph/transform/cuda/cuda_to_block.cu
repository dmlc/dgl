/**
 *  Copyright 2020-2021 Contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * @file graph/transform/cuda/cuda_to_block.cu
 * @brief Functions to convert a set of edges into a graph block with local
 * ids.
 *
 * Tested via python wrapper: python/dgl/path/to/to_block.py
 */

#include <cuda_runtime.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/device_api.h>
#include <dgl/runtime/tensordispatch.h>

#include <algorithm>
#include <memory>
#include <utility>

#include "../../../runtime/cuda/cuda_common.h"
#include "../../heterograph.h"
#include "../to_block.h"
#include "cuda_map_edges.cuh"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;
using namespace dgl::transform::cuda;
using TensorDispatcher = dgl::runtime::TensorDispatcher;

namespace dgl {
namespace transform {

namespace {

template <typename IdType>
class DeviceNodeMapMaker {
 public:
  explicit DeviceNodeMapMaker(const std::vector<int64_t>& maxNodesPerType)
      : max_num_nodes_(0) {
    max_num_nodes_ =
        *std::max_element(maxNodesPerType.begin(), maxNodesPerType.end());
  }

  /**
   * @brief This function builds node maps for each node type, preserving the
   * order of the input nodes. Here it is assumed the lhs_nodes are not unique,
   * and thus a unique list is generated.
   *
   * @param lhs_nodes The set of source input nodes.
   * @param rhs_nodes The set of destination input nodes.
   * @param node_maps The node maps to be constructed.
   * @param count_lhs_device The number of unique source nodes (on the GPU).
   * @param lhs_device The unique source nodes (on the GPU).
   * @param stream The stream to operate on.
   */
  void Make(
      const std::vector<IdArray>& lhs_nodes,
      const std::vector<IdArray>& rhs_nodes,
      DeviceNodeMap<IdType>* const node_maps, int64_t* const count_lhs_device,
      std::vector<IdArray>* const lhs_device, cudaStream_t stream) {
    const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

    CUDA_CALL(cudaMemsetAsync(
        count_lhs_device, 0, num_ntypes * sizeof(*count_lhs_device), stream));

    // possibly dublicate lhs nodes
    const int64_t lhs_num_ntypes = static_cast<int64_t>(lhs_nodes.size());
    for (int64_t ntype = 0; ntype < lhs_num_ntypes; ++ntype) {
      const IdArray& nodes = lhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDGLCUDA);
        node_maps->LhsHashTable(ntype).FillWithDuplicates(
            nodes.Ptr<IdType>(), nodes->shape[0],
            (*lhs_device)[ntype].Ptr<IdType>(), count_lhs_device + ntype,
            stream);
      }
    }

    // unique rhs nodes
    const int64_t rhs_num_ntypes = static_cast<int64_t>(rhs_nodes.size());
    for (int64_t ntype = 0; ntype < rhs_num_ntypes; ++ntype) {
      const IdArray& nodes = rhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        node_maps->RhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(), nodes->shape[0], stream);
      }
    }
  }

  /**
   * @brief This function builds node maps for each node type, preserving the
   * order of the input nodes. Here it is assumed both lhs_nodes and rhs_nodes
   * are unique.
   *
   * @param lhs_nodes The set of source input nodes.
   * @param rhs_nodes The set of destination input nodes.
   * @param node_maps The node maps to be constructed.
   * @param stream The stream to operate on.
   */
  void Make(
      const std::vector<IdArray>& lhs_nodes,
      const std::vector<IdArray>& rhs_nodes,
      DeviceNodeMap<IdType>* const node_maps, cudaStream_t stream) {
    const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

    // unique lhs nodes
    const int64_t lhs_num_ntypes = static_cast<int64_t>(lhs_nodes.size());
    for (int64_t ntype = 0; ntype < lhs_num_ntypes; ++ntype) {
      const IdArray& nodes = lhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDGLCUDA);
        node_maps->LhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(), nodes->shape[0], stream);
      }
    }

    // unique rhs nodes
    const int64_t rhs_num_ntypes = static_cast<int64_t>(rhs_nodes.size());
    for (int64_t ntype = 0; ntype < rhs_num_ntypes; ++ntype) {
      const IdArray& nodes = rhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        node_maps->RhsHashTable(ntype).FillWithUnique(
            nodes.Ptr<IdType>(), nodes->shape[0], stream);
      }
    }
  }

 private:
  IdType max_num_nodes_;
};

template <typename IdType>
struct CUDAIdsMapper {
  std::tuple<std::vector<IdArray>, std::vector<IdArray>> operator()(
      const HeteroGraphPtr& graph, bool include_rhs_in_lhs, int64_t num_ntypes,
      const DGLContext& ctx, const std::vector<int64_t>& maxNodesPerType,
      const std::vector<EdgeArray>& edge_arrays,
      const std::vector<IdArray>& src_nodes,
      const std::vector<IdArray>& rhs_nodes,
      std::vector<IdArray>* const lhs_nodes_ptr,
      std::vector<int64_t>* const num_nodes_per_type_ptr) {
    std::vector<IdArray>& lhs_nodes = *lhs_nodes_ptr;
    std::vector<int64_t>& num_nodes_per_type = *num_nodes_per_type_ptr;
    const bool generate_lhs_nodes = lhs_nodes.empty();
    auto device = runtime::DeviceAPI::Get(ctx);
    cudaStream_t stream = runtime::getCurrentCUDAStream();

    // Allocate space for map creation process.
    DeviceNodeMapMaker<IdType> maker(maxNodesPerType);
    DeviceNodeMap<IdType> node_maps(maxNodesPerType, num_ntypes, ctx, stream);
    if (generate_lhs_nodes) {
      lhs_nodes.reserve(num_ntypes);
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        lhs_nodes.emplace_back(
            NewIdArray(maxNodesPerType[ntype], ctx, sizeof(IdType) * 8));
      }
    }

    cudaEvent_t copyEvent;
    NDArray new_len_tensor;
    // Populate the mappings.
    if (generate_lhs_nodes) {
      int64_t* count_lhs_device = static_cast<int64_t*>(
          device->AllocWorkspace(ctx, sizeof(int64_t) * num_ntypes * 2));

      maker.Make(
          src_nodes, rhs_nodes, &node_maps, count_lhs_device, &lhs_nodes,
          stream);

      CUDA_CALL(cudaEventCreate(&copyEvent));
      if (TensorDispatcher::Global()->IsAvailable()) {
        new_len_tensor = NDArray::PinnedEmpty(
            {num_ntypes}, DGLDataTypeTraits<int64_t>::dtype,
            DGLContext{kDGLCPU, 0});
      } else {
        // use pageable memory, it will unecessarily block but be functional
        new_len_tensor = NDArray::Empty(
            {num_ntypes}, DGLDataTypeTraits<int64_t>::dtype,
            DGLContext{kDGLCPU, 0});
      }
      CUDA_CALL(cudaMemcpyAsync(
          new_len_tensor->data, count_lhs_device,
          sizeof(*num_nodes_per_type.data()) * num_ntypes,
          cudaMemcpyDeviceToHost, stream));
      CUDA_CALL(cudaEventRecord(copyEvent, stream));

      device->FreeWorkspace(ctx, count_lhs_device);
    } else {
      maker.Make(lhs_nodes, rhs_nodes, &node_maps, stream);

      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        num_nodes_per_type[ntype] = lhs_nodes[ntype]->shape[0];
      }
    }
    // Map node numberings from global to local, and build pointer for CSR.
    auto ret = MapEdges(graph, edge_arrays, node_maps, stream);

    if (generate_lhs_nodes) {
      // wait for the previous copy
      CUDA_CALL(cudaEventSynchronize(copyEvent));
      CUDA_CALL(cudaEventDestroy(copyEvent));

      // Resize lhs nodes.
      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        num_nodes_per_type[ntype] =
            static_cast<int64_t*>(new_len_tensor->data)[ntype];
        lhs_nodes[ntype]->shape[0] = num_nodes_per_type[ntype];
      }
    }

    return ret;
  }
};

template <typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>> ToBlockGPU(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* const lhs_nodes_ptr) {
  return dgl::transform::ProcessToBlock<IdType>(
      graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes_ptr,
      CUDAIdsMapper<IdType>());
}

}  // namespace

// Use explicit names to get around MSVC's broken mangling that thinks the
// following two functions are the same. Using template<> fails to export the
// symbols.
std::tuple<HeteroGraphPtr, std::vector<IdArray>>
// ToBlock<kDGLCUDA, int32_t>
ToBlockGPU32(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* const lhs_nodes) {
  return ToBlockGPU<int32_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

std::tuple<HeteroGraphPtr, std::vector<IdArray>>
// ToBlock<kDGLCUDA, int64_t>
ToBlockGPU64(
    HeteroGraphPtr graph, const std::vector<IdArray>& rhs_nodes,
    bool include_rhs_in_lhs, std::vector<IdArray>* const lhs_nodes) {
  return ToBlockGPU<int64_t>(graph, rhs_nodes, include_rhs_in_lhs, lhs_nodes);
}

}  // namespace transform
}  // namespace dgl
