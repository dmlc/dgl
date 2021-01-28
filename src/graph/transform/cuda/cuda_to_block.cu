/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

#include "cuda_to_block.h"

#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>
#include <algorithm>

#include "../../../runtime/cuda/cuda_common.h"
#include "../../../runtime/cuda/cuda_hashtable.cuh"
#include "../../heterograph.h"

using namespace dgl::aten;
using namespace dgl::runtime::cuda;

namespace dgl {
namespace transform {
namespace cuda {

namespace {


template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__device__ void map_vertex_ids(
    const IdType * const global,
    IdType * const new_global,
    const IdType num_vertices,
    const HashTable<IdType>& table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Mapping = typename HashTable<IdType>::Mapping;

  const IdType tile_start = TILE_SIZE*blockIdx.x;
  const IdType tile_end = min(TILE_SIZE*(blockIdx.x+1), num_vertices);

  for (IdType idx = threadIdx.x+tile_start; idx < tile_end; idx+=BLOCK_SIZE) {
    const Mapping& mapping = *search_hashmap(global[idx], table);
    new_global[idx] = mapping.local;
  }
}


/**
* @brief This generates a hash map where the keys are the global node numbers,
* and the values are indexes, and inputs may have duplciates.
*
* @tparam IdType The type of of id.
* @tparam BLOCK_SIZE The size of the thread block.
* @tparam TILE_SIZE The number of entries each thread block will process.
* @param nodes The nodes to insert.
* @param num_nodes The number of nodes to insert.
* @param table The hash table.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_duplicates(
    const IdType * const nodes,
    const int64_t num_nodes,
    HashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      insert_hashmap(nodes[index], index, table);
    }
  }
}


/**
* @brief This generates a hash map where the keys are the global node numbers,
* and the values are indexes, and all inputs are unique.
*
* @tparam IdType The type of of id.
* @tparam BLOCK_SIZE The size of the thread block.
* @tparam TILE_SIZE The number of entries each thread block will process.
* @param nodes The unique nodes to insert.
* @param num_nodes The number of nodes to insert.
* @param table The hash table.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(
    const IdType * const nodes,
    const int64_t num_nodes,
    HashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const size_t pos = insert_hashmap(nodes[index], index, table);

      // since we are only inserting unique nodes, we know their local id
      // will be equal to their index
      table[pos].local = static_cast<IdType>(index);
    }
  }
}


/**
* @brief This counts the number of nodes inserted per thread block.
*
* @tparam IdType The type of of id.
* @tparam BLOCK_SIZE The size of the thread block.
* @tparam TILE_SIZE The number of entries each thread block will process.
* @param nodes The nodes ot insert.
* @param num_nodes The number of nodes to insert.
* @param hash_size The size of the hash table.
* @param pairs The hash table.
* @param num_unique_nodes The number of nodes inserted into the hash table per thread
* block.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(
    const IdType * nodes,
    const size_t num_nodes,
    const HashTable<IdType> table,
    IdType * const num_unique_nodes) {
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using Mapping = typename HashTable<IdType>::Mapping;

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  IdType count = 0;

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const Mapping& mapping = *search_hashmap(
          nodes[index], table);
      if (mapping.index == index) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique_nodes[blockIdx.x] = count;
    if (blockIdx.x == 0) {
      num_unique_nodes[gridDim.x] = 0;
    }
  }
}

template<typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(
      const IdType running_total) :
    running_total_(running_total) {
  }

  __device__ IdType operator()(const IdType block_aggregate) {
      const IdType old_prefix = running_total_;
      running_total_ += block_aggregate;
      return old_prefix;
  }
};



/**
* @brief Update the local numbering of elements in the hashmap.
*
* @tparam IdType The type of id.
* @tparam BLOCK_SIZE The size of the thread blocks.
* @tparam TILE_SIZE The number of elements each thread block works on.
* @param nodes The set of non-unique nodes to update from.
* @param num_nodes The number of non-unique nodes.
* @param hash_size The size of the hash table.
* @param mappings The hash table.
* @param num_nodes_prefix The number of unique nodes preceding each thread
* block.
* @param global_nodes The set of unique nodes (output).
* @param global_node_count The number of unique nodes (output).
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(
    const IdType * const nodes,
    const size_t num_nodes,
    HashTable<IdType> table,
    const IdType * const num_nodes_prefix,
    IdType * const global_nodes,
    int64_t * const global_node_count) {
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename HashTable<IdType>::Mapping;

  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

  const IdType offset = num_nodes_prefix[blockIdx.x];

  BlockPrefixCallbackOp<FlagType> prefix_op(0);

  // count successful placements
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i*BLOCK_SIZE + blockIdx.x*TILE_SIZE;

    FlagType flag;
    Mapping * kv;
    if (index < num_nodes) {
      kv = search_hashmap(nodes[index], table);
      flag = kv->index == index;
    } else {
      flag = 0;
    }

    if (!flag) {
      kv = nullptr;
    }

    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = offset+flag;
      kv->local = pos;
      global_nodes[pos] = nodes[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *global_node_count = num_nodes_prefix[gridDim.x];
  }
}

/**
* @brief Generate mapped edge endpoint ids.
*
* @tparam IdType The type of id.
* @tparam BLOCK_SIZE The size of each thread block.
* @tparam TILE_SIZE The number of edges to process per thread block.
* @param global_srcs_device The source ids to map.
* @param new_global_srcs_device The mapped source ids (output).
* @param global_dsts_device The destination ids to map.
* @param new_global_dsts_device The mapped destination ids (output).
* @param num_edges The number of edges to map.
* @param src_mapping The mapping of sources ids.
* @param src_hash_size The the size of source id hash table/mapping.
* @param dst_mapping The mapping of destination ids.
* @param dst_hash_size The the size of destination id hash table/mapping.
*/
template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    const IdType * const global_srcs_device,
    IdType * const new_global_srcs_device,
    const IdType * const global_dsts_device,
    IdType * const new_global_dsts_device,
    const IdType num_edges,
    const HashTable<IdType> src_mapping,
    const HashTable<IdType> dst_mapping) {
  assert(BLOCK_SIZE == blockDim.x);
  assert(2 == gridDim.y);

  if (blockIdx.y == 0) {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_srcs_device,
        new_global_srcs_device,
        num_edges,
        src_mapping);
  } else {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_dsts_device,
        new_global_dsts_device,
        num_edges,
        dst_mapping);
  }
}

// CPU Code ///////////////////////////////////////////////////////////////////

template<typename IdType>
inline size_t RoundUpDiv(
    const IdType num,
    const size_t divisor) {
  return static_cast<IdType>(num/divisor) + (num % divisor == 0 ? 0 : 1);
}


template<typename IdType>
inline IdType RoundUp(
    const IdType num,
    const size_t unit) {
  return RoundUpDiv(num, unit)*unit;
}


template<typename IdType>
class DeviceNodeMap {
 public:
  constexpr static const int HASH_SCALING = 2;

  using Mapping = typename HashTable<IdType>::Mapping;

  DeviceNodeMap(
      const std::vector<int64_t>& num_nodes,
      DGLContext ctx,
      cudaStream_t stream) :
    num_types_(num_nodes.size()),
    rhs_offset_(num_types_/2),
    workspaces_(),
    hash_tables_(),
    ctx_(ctx) {
    auto device = runtime::DeviceAPI::Get(ctx);

    hash_tables_.reserve(num_types_);
    workspaces_.reserve(num_types_);
    for (int64_t i = 0; i < num_types_; ++i) {
      const size_t hash_size = GetHashSize(num_nodes[i]);
      size_t workspace_size = HashTable<IdType>::RequiredWorkspace(hash_size);
      void * workspace = device->AllocWorkspace(ctx, workspace_size);
      workspaces_.emplace_back(workspace);
      hash_tables_.emplace_back(
          HashTable<IdType>(
            hash_size,
            workspace,
            workspace_size,
            stream));
    }
  }

  ~DeviceNodeMap() {
    auto device = runtime::DeviceAPI::Get(ctx_);
    for (void * ptr : workspaces_) {
      device->FreeWorkspace(ctx_, ptr);
    }
  }


  HashTable<IdType>& LhsHashTable(
      const size_t index) {
    return HashData(index);
  }

  HashTable<IdType>& RhsHashTable(
      const size_t index) {
    return HashData(index+rhs_offset_);
  }

  const HashTable<IdType>& LhsHashTable(
      const size_t index) const {
    return HashData(index);
  }

  const HashTable<IdType>& RhsHashTable(
      const size_t index) const {
    return HashData(index+rhs_offset_);
  }

  IdType LhsHashSize(
      const size_t index) const {
    return HashSize(index);
  }

  IdType RhsHashSize(
      const size_t index) const {
    return HashSize(rhs_offset_+index);
  }

  size_t Size() const {
    return hash_tables_.size();
  }

 private:
  size_t num_types_;
  size_t rhs_offset_;
  std::vector<void*> workspaces_;
  std::vector<HashTable<IdType>> hash_tables_;
  DGLContext ctx_;

  static size_t GetHashSize(const size_t num_nodes) {
    const size_t num_bits = static_cast<size_t>(
        std::ceil(std::log2(static_cast<double>(num_nodes))));
    const size_t next_pow2 = 1ull << num_bits;

    // make sure how hash size is not a power of 2
    const size_t hash_size = (next_pow2 << HASH_SCALING)-1;

    return hash_size;
  }

  inline HashTable<IdType>& HashData(
      const size_t index) {
    CHECK_LT(index, hash_tables_.size());
    return hash_tables_[index];
  }

  inline const HashTable<IdType>& HashData(
      const size_t index) const {
    CHECK_LT(index, hash_tables_.size());
    return hash_tables_[index];
  }

  inline IdType HashSize(
      const size_t index) const {
    return HashData(index).size();
  }
};

template<typename IdType>
class DeviceNodeMapMaker {
 public:
  constexpr static const int BLOCK_SIZE = 256;
  constexpr static const size_t TILE_SIZE = 1024;

  static size_t RequiredPrefixSumBytes(
      const size_t max_num) {
    size_t prefix_sum_bytes;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(
          nullptr,
          prefix_sum_bytes,
          static_cast<IdType*>(nullptr),
          static_cast<IdType*>(nullptr),
          RoundUp(max_num, TILE_SIZE)+1));
    return RoundUp(prefix_sum_bytes, 8);
  }

  DeviceNodeMapMaker(
      const std::vector<int64_t> maxNodesPerType) :
      max_num_nodes_(0),
      workspace_size_bytes_(0) {
    max_num_nodes_ = *std::max_element(maxNodesPerType.begin(),
        maxNodesPerType.end());

    workspace_size_bytes_ = RequiredPrefixSumBytes(max_num_nodes_) +
        sizeof(IdType)*(max_num_nodes_+1);
  }

  size_t RequiredWorkspaceBytes() const {
    return workspace_size_bytes_;
  }

  /**
  * @brief This function builds node maps for each node type, preserving the
  * order of the input nodes.
  *
  * @param nodes The set of input nodes.
  * @param nodes_maps The constructed node maps.
  * @param workspace The scratch space to use.
  * @param workspaceSize The size of the scratch space.
  * @param stream The stream to operate on.
  */
  void Make(
      const std::vector<IdArray>& lhs_nodes,
      const std::vector<IdArray>& rhs_nodes,
      DeviceNodeMap<IdType> * const node_maps,
      int64_t * const count_lhs_device,
      std::vector<IdArray>* const lhs_device,
      void * const workspace,
      const size_t workspaceSize,
      cudaStream_t stream) {
    if (workspaceSize < RequiredWorkspaceBytes()) {
      throw std::runtime_error("Not enough space for sorting: " +
          std::to_string(workspaceSize) + " / " +
          std::to_string(RequiredWorkspaceBytes()));
    }

    void * const prefix_sum_space = workspace;
    size_t prefix_sum_bytes = RequiredPrefixSumBytes(
        max_num_nodes_);

    IdType * const node_prefix = reinterpret_cast<IdType*>(
        static_cast<uint8_t*>(prefix_sum_space) + prefix_sum_bytes);

    const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

    CUDA_CALL(cudaMemsetAsync(
      count_lhs_device,
      0,
      num_ntypes*sizeof(*count_lhs_device),
      stream));

    // possibly dublicate lhs nodes
    for (int64_t ntype = 0; ntype < lhs_nodes.size(); ++ntype) {
      const IdArray& nodes = lhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);

        const dim3 grid(RoundUpDiv(nodes->shape[0], TILE_SIZE));
        const dim3 block(BLOCK_SIZE);

        generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            node_maps->LhsHashTable(ntype));
        CUDA_CALL(cudaGetLastError());

        count_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            node_maps->LhsHashTable(ntype),
            node_prefix);
        CUDA_CALL(cudaGetLastError());

        CUDA_CALL(cub::DeviceScan::ExclusiveSum(
            prefix_sum_space,
            prefix_sum_bytes,
            node_prefix,
            node_prefix,
            grid.x+1, stream));

        compact_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            node_maps->LhsHashTable(ntype),
            node_prefix,
            (*lhs_device)[ntype].Ptr<IdType>(),
            count_lhs_device+ntype);
        CUDA_CALL(cudaGetLastError());
      }
    }

    // unique rhs nodes
    for (int64_t ntype = 0; ntype < rhs_nodes.size(); ++ntype) {
      const IdArray& nodes = rhs_nodes[ntype];
      if (nodes->shape[0] > 0) {
        CHECK_EQ(nodes->ctx.device_type, kDLGPU);

        const dim3 grid(RoundUpDiv(nodes->shape[0], TILE_SIZE));
        const dim3 block(BLOCK_SIZE);

        generate_hashmap_unique<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
            nodes.Ptr<IdType>(),
            nodes->shape[0],
            node_maps->RhsHashTable(ntype));
        CUDA_CALL(cudaGetLastError());
      }
    }
  }

 private:
  IdType max_num_nodes_;
  size_t workspace_size_bytes_;
};

template<typename IdType>
std::tuple<std::vector<IdArray>, std::vector<IdArray>>
MapEdges(
    HeteroGraphPtr graph,
    const std::vector<EdgeArray>& edge_sets,
    const DeviceNodeMap<IdType>& node_map,
    cudaStream_t stream) {
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;

  const auto& ctx = graph->Context();

  std::vector<IdArray> new_lhs;
  new_lhs.reserve(edge_sets.size());
  std::vector<IdArray> new_rhs;
  new_rhs.reserve(edge_sets.size());

  // The next peformance optimization here, is to perform mapping of all edge
  // types in a single kernel launch.
  for (int64_t etype = 0; etype < edge_sets.size(); ++etype) {
    const EdgeArray& edges = edge_sets[etype];
    if (edges.id.defined()) {
      const int64_t num_edges = edges.src->shape[0];

      new_lhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType)*8));
      new_rhs.emplace_back(NewIdArray(num_edges, ctx, sizeof(IdType)*8));

      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const int src_type = src_dst_types.first;
      const int dst_type = src_dst_types.second;

      const dim3 grid(RoundUpDiv(num_edges, TILE_SIZE), 2);
      const dim3 block(BLOCK_SIZE);

      // map the srcs
      map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE><<<
        grid,
        block,
        0,
        stream>>>(
        edges.src.Ptr<IdType>(),
        new_lhs.back().Ptr<IdType>(),
        edges.dst.Ptr<IdType>(),
        new_rhs.back().Ptr<IdType>(),
        num_edges,
        node_map.LhsHashTable(src_type),
        node_map.RhsHashTable(dst_type));
      CUDA_CALL(cudaGetLastError());
    } else {
      new_lhs.emplace_back(aten::NullArray());
      new_rhs.emplace_back(aten::NullArray());
    }
  }

  return std::tuple<std::vector<IdArray>, std::vector<IdArray>>(
      std::move(new_lhs), std::move(new_rhs));
}


template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlockInternal(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    const bool include_rhs_in_lhs) {
  cudaStream_t stream = 0;
  const auto& ctx = graph->Context();
  auto device = runtime::DeviceAPI::Get(ctx);

  CHECK_EQ(ctx.device_type, kDLGPU);

  // Since DST nodes are included in SRC nodes, a common requirement is to fetch
  // the DST node features from the SRC nodes features. To avoid expensive sparse lookup,
  // the function assures that the DST nodes in both SRC and DST sets have the same ids.
  // As a result, given the node feature tensor ``X`` of type ``utype``,
  // the following code finds the corresponding DST node features of type ``vtype``:

  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();

  CHECK(rhs_nodes.size() == static_cast<size_t>(num_ntypes))
    << "rhs_nodes not given for every node type";

  std::vector<EdgeArray> edge_arrays(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t dsttype = src_dst_types.second;
    if (!aten::IsNullArray(rhs_nodes[dsttype])) {
      edge_arrays[etype] = graph->Edges(etype);
    }
  }

  // count lhs and rhs nodes
  std::vector<int64_t> maxNodesPerType(num_ntypes*2, 0);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    maxNodesPerType[ntype+num_ntypes] += rhs_nodes[ntype]->shape[0];

    if (include_rhs_in_lhs) {
      maxNodesPerType[ntype] += rhs_nodes[ntype]->shape[0];
    }
  }
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    if (edge_arrays[etype].src.defined()) {
      maxNodesPerType[srctype] += edge_arrays[etype].src->shape[0];
    }
  }

  // gather lhs_nodes
  std::vector<int64_t> src_node_offsets(num_ntypes, 0);
  std::vector<IdArray> src_nodes(num_ntypes);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    src_nodes[ntype] = NewIdArray(maxNodesPerType[ntype], ctx,
        sizeof(IdType)*8);
    if (include_rhs_in_lhs) {
      // place rhs nodes first
      device->CopyDataFromTo(rhs_nodes[ntype].Ptr<IdType>(), 0,
          src_nodes[ntype].Ptr<IdType>(), src_node_offsets[ntype],
          sizeof(IdType)*rhs_nodes[ntype]->shape[0],
          rhs_nodes[ntype]->ctx, src_nodes[ntype]->ctx,
          rhs_nodes[ntype]->dtype,
          stream);
      src_node_offsets[ntype] += sizeof(IdType)*rhs_nodes[ntype]->shape[0];
    }
  }
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    if (edge_arrays[etype].src.defined()) {
      device->CopyDataFromTo(
          edge_arrays[etype].src.Ptr<IdType>(), 0,
          src_nodes[srctype].Ptr<IdType>(),
          src_node_offsets[srctype],
          sizeof(IdType)*edge_arrays[etype].src->shape[0],
          rhs_nodes[srctype]->ctx,
          src_nodes[srctype]->ctx,
          rhs_nodes[srctype]->dtype,
          stream);

      src_node_offsets[srctype] += sizeof(IdType)*edge_arrays[etype].src->shape[0];
    }
  }

  // allocate space for map creation process
  DeviceNodeMapMaker<IdType> maker(maxNodesPerType);
  const size_t scratch_size = maker.RequiredWorkspaceBytes();
  void * const scratch_device = device->AllocWorkspace(ctx, scratch_size);

  DeviceNodeMap<IdType> node_maps(maxNodesPerType, ctx, stream);

  int64_t total_lhs = 0;
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    total_lhs += maxNodesPerType[ntype];
  }

  std::vector<IdArray> lhs_nodes;
  lhs_nodes.reserve(num_ntypes);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    lhs_nodes.emplace_back(NewIdArray(
        maxNodesPerType[ntype], ctx, sizeof(IdType)*8));
  }

  // populate the mappings
  int64_t * count_lhs_device = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*num_ntypes*2));
  maker.Make(
      src_nodes,
      rhs_nodes,
      &node_maps,
      count_lhs_device,
      &lhs_nodes,
      scratch_device,
      scratch_size,
      stream);
  device->FreeWorkspace(ctx, scratch_device);

  std::vector<IdArray> induced_edges;
  induced_edges.reserve(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].id.defined()) {
      induced_edges.push_back(edge_arrays[etype].id);
    } else {
      induced_edges.push_back(
            aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx));
    }
  }

  // build metagraph -- small enough to be done on CPU
  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);

  // allocate vector for graph relations while GPU is busy
  std::vector<HeteroGraphPtr> rel_graphs;
  rel_graphs.reserve(num_etypes);

  std::vector<int64_t> num_nodes_per_type(num_ntypes*2);
  // populate RHS nodes from what we already know
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    num_nodes_per_type[num_ntypes+ntype] = rhs_nodes[ntype]->shape[0];
  }
  device->CopyDataFromTo(
      count_lhs_device, 0,
      num_nodes_per_type.data(), 0,
      sizeof(*num_nodes_per_type.data())*num_ntypes,
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, 64, 1},
      stream);
  device->StreamSync(ctx, stream);

  // wait for the node counts to finish transferring
  device->FreeWorkspace(ctx, count_lhs_device);

  // map node numberings from global to local, and build pointer for CSR
  std::vector<IdArray> new_lhs;
  std::vector<IdArray> new_rhs;
  std::tie(new_lhs, new_rhs) = MapEdges(graph, edge_arrays, node_maps, stream);

  // resize lhs nodes
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    lhs_nodes[ntype]->shape[0] = num_nodes_per_type[ntype];
  }

  // build the heterograph
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;

    if (rhs_nodes[dsttype]->shape[0] == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_nodes[srctype]->shape[0], rhs_nodes[dsttype]->shape[0],
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx),
          aten::NullArray(DLDataType{kDLInt, sizeof(IdType)*8, 1}, ctx)));
    } else {
      rel_graphs.push_back(CreateFromCOO(
          2,
          lhs_nodes[srctype]->shape[0],
          rhs_nodes[dsttype]->shape[0],
          new_lhs[etype],
          new_rhs[etype]));
    }
  }

  HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);

  // return the new graph, the new src nodes, and new edges
  return std::make_tuple(new_graph, lhs_nodes, induced_edges);
}
}  // namespace


std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
CudaToBlock(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs) {
  std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>> ret;
  ATEN_ID_TYPE_SWITCH(graph->DataType(), IdType, {
    ret = ToBlockInternal<IdType>(graph, rhs_nodes, include_rhs_in_lhs);
  });
  return ret;
}


}  // namespace cuda
}  // namespace transform
}  // namespace dgl
