/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

#include "cuda_to_block.h"

#include "cuda_runtime.h"
#include <cub/cub.cuh>

#include <dgl/runtime/device_api.h>
#include <dgl/immutable_graph.h>

using namespace dgl::aten;

namespace dgl {
namespace transform {

namespace
{

// helper types

template<typename IdType>
struct Mapping {
  IdType key;
  IdType local;
  int64_t index;
};

template<typename IdType>
struct EmptyKey
{
  constexpr static const IdType value = static_cast<IdType>(-1);
};

inline __device__ int64_t atomicCAS(
    int64_t * const address,
    const int64_t compare,
    const int64_t val)
{
  using Type = unsigned long long int;

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return ::atomicCAS((Type*)address,
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}

inline __device__ int32_t atomicCAS(
    int32_t * const address,
    const int32_t compare,
    const int32_t val)
{
  using Type = int;

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return ::atomicCAS((Type*)address,
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}



template<typename IdType>
struct BlockPrefixCallbackOp
{
    IdType m_running_total;

    __device__ BlockPrefixCallbackOp(
        const IdType running_total) :
      m_running_total(running_total)
    {
    }

    __device__ IdType operator()(const IdType block_aggregate)
    {
        const IdType old_prefix = m_running_total;
        m_running_total += block_aggregate;
        return old_prefix;
    }
};

template<typename IdType>
inline __device__ IdType hash(
    const IdType mask,
    const IdType id)
{
  return id & mask;
}


template<typename IdType>
inline __device__ bool attempt_insert_at(
    const IdType pos,
    const IdType id,
    const size_t index,
    Mapping<IdType> * const table)
{
  const IdType key = atomicCAS(&table[pos].key, EmptyKey<IdType>::value, id);
  if (key == EmptyKey<IdType>::value || key == id) {
    // we either set a match key, or found a matching key, so then place the
    // minimum index in position
    atomicMin(reinterpret_cast<unsigned long long*>(&table[pos].index),
        static_cast<unsigned long long>(index));
    return true;
  } else {
    // we need to search elsewhere
    return false;
  }
}

template<typename IdType>
inline __device__ void insert_hashmap(
    const IdType id,
    const size_t index,
    Mapping<IdType> * const table,
    const IdType table_size)
{
  IdType pos = id % table_size;

  // linearly scan for an empty slot or matching entry
  IdType delta = 1;
  while (!attempt_insert_at(pos, id, index, table)) {
    pos = (pos+delta) % table_size;
    delta +=1;
  }
}

template<typename IdType>
inline __device__ IdType search_hashmap_for_pos(
    const IdType id,
    const Mapping<IdType> * const table,
    const IdType table_size)
{
  IdType pos = id % table_size;

  // linearly scan for matching entry
  IdType delta = 1;
  while (table[pos].key != id) {
    assert(table[pos].key != EmptyKey<IdType>::value);
    pos = (pos+delta) % table_size;
    delta +=1;
  }
  assert(pos < table_size);

  return pos;
}

template<typename IdType>
inline __device__ const Mapping<IdType> * search_hashmap(
    const IdType id,
    const Mapping<IdType> * const table,
    const IdType table_size)
{
  const IdType pos = search_hashmap_for_pos(id, table, table_size);

  return table+pos;
}

template<typename IdType>
inline __device__ Mapping<IdType> * search_hashmap(
    const IdType id,
    Mapping<IdType> * const table,
    const IdType table_size)
{
  const IdType pos = search_hashmap_for_pos(id, table, table_size);

  return table+pos;
}




/**
* @brief This generates a hash map where the keys are the global node numbers,
* and the values are indexes.
*
* @tparam IdType The type of of id.
* @tparam BLOCK_SIZE The size of the thread block.
* @tparam TILE_SIZE The number of entries each thread block will process.
* @param nodes The nodes ot insert.
* @param num_nodes The number of nodes to insert.
* @param hash_size The size of the hash table.
* @param pairs The hash table.
*/
template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap(
    const IdType * const nodes,
    const size_t num_nodes,
    const IdType hash_size,
    Mapping<IdType> * const mappings)
{
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      insert_hashmap(nodes[index], index, mappings, hash_size);
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
    const IdType hash_size,
    const Mapping<IdType> * const pairs,
    IdType * const num_unique_nodes)
{
  assert(BLOCK_SIZE == blockDim.x);

  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;

  const size_t block_start = TILE_SIZE*blockIdx.x;
  const size_t block_end = TILE_SIZE*(blockIdx.x+1);

  IdType count = 0;

  #pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end; index += BLOCK_SIZE) {
    if (index < num_nodes) {
      const Mapping<IdType>& mapping = *search_hashmap(
          nodes[index], pairs, hash_size);
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


template<typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(
    const IdType * const nodes,
    const size_t num_nodes,
    const IdType hash_size,
    Mapping<IdType> * const mappings,
    const IdType * const num_nodes_prefix,
    IdType * const global_nodes,
    int64_t * const global_node_count)
{
  assert(BLOCK_SIZE == blockDim.x);

  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<IdType, BLOCK_SIZE>;

  constexpr const size_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

  __shared__ typename BlockScan::TempStorage temp_space;

	BlockPrefixCallbackOp<IdType> prefix_op(num_nodes_prefix[blockIdx.x]);

  // count successful placements
  for (size_t i = 0; i < VALS_PER_THREAD; ++i) {
		const size_t index = threadIdx.x + i*BLOCK_SIZE + blockIdx.x*TILE_SIZE;

    IdType flag;
    Mapping<IdType>* kv;
    if (index < num_nodes) {
      kv = search_hashmap(nodes[index], mappings, hash_size);
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
      kv->local = flag;
      global_nodes[flag] = nodes[index];
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *global_node_count = num_nodes_prefix[gridDim.x];
  }
}

template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__device__ void map_vertex_ids(
    IdType * const global,
    const IdType num_vertices,
    const Mapping<IdType> * const mappings,
    const IdType hash_size)
{
  assert(BLOCK_SIZE == blockDim.x);

  const IdType tile_start = TILE_SIZE*blockIdx.x;
  const IdType tile_end = min(TILE_SIZE*(blockIdx.x+1), num_vertices);

  for (IdType idx = threadIdx.x+tile_start; idx < tile_end; idx+=BLOCK_SIZE) {
    const Mapping<IdType>& mapping = *search_hashmap(global[idx], mappings, 
        hash_size);
    global[idx] = mapping.local;
  }
}


template<typename IdType, int BLOCK_SIZE, IdType TILE_SIZE>
__global__ void map_edge_ids(
    IdType * const global_srcs_device,
    IdType * const global_dsts_device,
    const IdType num_edges,
    const Mapping<IdType> * const src_mapping,
    const IdType src_hash_size,
    const Mapping<IdType> * const dst_mapping,
    const IdType dst_hash_size)
{
  assert(BLOCK_SIZE == blockDim.x);

  if (blockIdx.y == 0) {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_srcs_device,
        num_edges,
        src_mapping,
        src_hash_size);
  } else {
    map_vertex_ids<IdType, BLOCK_SIZE, TILE_SIZE>(
        global_dsts_device,
        num_edges,
        dst_mapping,
        dst_hash_size);
  }
}


template<typename IdType>
inline size_t round_up_div(
    const IdType num,
    const size_t divisor)
{
  return static_cast<IdType>(num/divisor) + (num % divisor == 0 ? 0 : 1);
}

template<typename IdType>
inline IdType round_up(
    const IdType num,
    const size_t unit)
{
  return round_up_div(num, unit)*unit;
}

template<typename T>
void checked_memset(
  T * const ptr,
  const int val,
  const size_t num,
  cudaStream_t stream)
{
  cudaError_t err = cudaMemsetAsync(ptr, val, num*sizeof(T), stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to execute memset: " + std::to_string(err)
    + " : " + std::string(cudaGetErrorString(err)));
  }
}

template<typename T>
void checked_d2d_copy(
  T * const dst_device,
  const T * const src_device,
  const size_t num,
  cudaStream_t stream)
{
  cudaError_t err = cudaMemcpyAsync(dst_device, src_device,
      sizeof(*dst_device)*num, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to launch d2d memcpy: " + std::to_string(err)
    + " : " + std::string(cudaGetErrorString(err)));
  }
}

template<typename T>
void checked_h2d_copy(
  T * const dst_device,
  const T * const src_host,
  const size_t num,
  cudaStream_t stream)
{
  cudaError_t err = cudaMemcpyAsync(dst_device, src_host,
      sizeof(*dst_device)*num, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to launch h2d memcpy: " + std::to_string(err)
    + " : " + std::string(cudaGetErrorString(err)));
  }
}

template<typename T>
void checked_d2h_copy(
  T * const dst_host,
  const T * const src_device,
  const size_t num,
  cudaStream_t stream)
{
  cudaError_t err = cudaMemcpyAsync(dst_host, src_device,
      sizeof(*dst_host)*num, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to launch d2h memcpy: " + std::to_string(err)
    + " : " + std::string(cudaGetErrorString(err)));
  }
}



void check_cub_call(
    const cudaError_t err)
{
  if (err != cudaSuccess) {
    throw std::runtime_error("Error running cub: " + std::to_string(err)
        + " : '" + std::string(cudaGetErrorString(err)) + "'.");
  }
}

void check_last_error()
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to launch kernel: "
    + std::to_string(err) + " : " +
    std::string(cudaGetErrorString(err)));
  }
}



template<typename IdType>
class DeviceNodeMap
{
  public:
    constexpr static const int HASH_SCALING = 2;

    DeviceNodeMap(
        const std::vector<int64_t>& num_nodes,
        DGLContext ctx) :
      m_num_types(num_nodes.size()),
      m_rhs_offset(m_num_types/2),
      m_masks(m_num_types),
      m_hash_prefix(m_num_types+1, 0),
      m_hash_tables(nullptr),
      m_num_nodes_ptr(nullptr) // configure in constructor
    {
      auto device = runtime::DeviceAPI::Get(ctx);

      for (int64_t i = 0; i < m_num_types; ++i) {
        const size_t hash_size = getHashSize(num_nodes[i]);
        m_masks[i] = hash_size -1;
        m_hash_prefix[i+1] = m_hash_prefix[i] + hash_size;
      }

      // configure workspace
      m_hash_tables = static_cast<Mapping<IdType>*>(device->AllocWorkspace(ctx,
          m_hash_prefix.back()*sizeof(*m_hash_tables)));
      m_num_nodes_ptr = static_cast<IdType*>(device->AllocWorkspace(ctx,
          m_num_types*sizeof(*m_num_nodes_ptr)));
    }

    IdType * num_nodes_ptrs()
    {
      return m_num_nodes_ptr;
    }

    IdType * num_nodes_ptr(
        const size_t index) 
    {
      return m_num_nodes_ptr + index;
    }

    Mapping<IdType> * hash_table(
        const size_t index)
    {
      return m_hash_tables + m_hash_prefix[index];
    }

    const Mapping<IdType> * hash_table(
        const size_t index) const
    {
      return m_hash_tables + m_hash_prefix[index];
    }

    IdType hash_mask(
        const size_t index) const
    {
      return m_masks[index];
    }

    IdType hash_size(
        const size_t index) const
    {
      return m_hash_prefix[index+1]-m_hash_prefix[index];
    }

    const Mapping<IdType> * lhs_hash_table(
        const size_t index) const
    {
      return hash_table(index);
    }

    const Mapping<IdType> * rhs_hash_table(
        const size_t index) const
    {
      return hash_table(index+m_rhs_offset);
    }

    IdType lhs_hash_mask(
        const size_t index) const
    {
      return hash_mask(index);
    }

    IdType rhs_hash_mask(
        const size_t index) const
    {
      return hash_mask(m_rhs_offset+index);
    }

    IdType lhs_hash_size(
        const size_t index) const
    {
      return hash_size(index);
    }

    IdType rhs_hash_size(
        const size_t index) const
    {
      return hash_size(m_rhs_offset+index);
    }

    const IdType * lhs_num_nodes_ptr(
        const int64_t ntype) const
    {
      return m_num_nodes_ptr+ntype;
    }

    const IdType * rhs_num_nodes_ptr(
        const int64_t ntype) const
    {
      return m_num_nodes_ptr+m_rhs_offset+ntype;
    }

    size_t size() const
    {
      return m_masks.size();
    }

  private:
    size_t m_num_types;
    size_t m_rhs_offset;
    std::vector<int> m_masks;
    std::vector<size_t> m_hash_prefix;
    Mapping<IdType> * m_hash_tables;
    IdType * m_num_nodes_ptr;

    static size_t getHashSize(const size_t num_nodes)
    {
      const size_t num_bits = static_cast<size_t>(std::ceil(std::log2(static_cast<double>(num_nodes))));
      const size_t next_pow2 = 1ull << num_bits;

      // make sure how hash size is not a power of 2
      const size_t hash_size = (next_pow2 << HASH_SCALING)-1;

      return hash_size;
    }
};

template<typename IdType>
class DeviceNodeMapMaker
{
  public:
    constexpr static const int BLOCK_SIZE = 256;
    constexpr static const size_t TILE_SIZE = 1024;

    static size_t requiredPrefixSumBytes(
        const size_t max_num)
    {
      size_t prefix_sum_bytes;
      check_cub_call(cub::DeviceScan::ExclusiveSum(
            nullptr,
            prefix_sum_bytes,
            static_cast<IdType*>(nullptr),
            static_cast<IdType*>(nullptr),
            round_up(max_num, BLOCK_SIZE)));
      return round_up(prefix_sum_bytes, 8);
    }

    DeviceNodeMapMaker(
        const std::vector<int64_t> maxNodesPerType) :
        m_maxNumNodes(0),
        m_workspaceSizeBytes(0)
    {
      m_maxNumNodes = *std::max_element(maxNodesPerType.begin(),
          maxNodesPerType.end());

      m_workspaceSizeBytes = requiredPrefixSumBytes(m_maxNumNodes) + sizeof(IdType)*(m_maxNumNodes+1);
    }

    size_t requiredWorkspaceBytes() const
    {
      return m_workspaceSizeBytes;
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
    void make(
        const std::vector<IdArray>& lhs_nodes,
        const std::vector<IdArray>& rhs_nodes,
        DeviceNodeMap<IdType> * const node_maps,
        int64_t * const count_lhs_device,
        std::vector<IdArray>& lhs_device,
        void * const workspace,
        const size_t workspaceSize,
        cudaStream_t stream)
    {
      if (workspaceSize < requiredWorkspaceBytes()) {
        throw std::runtime_error("Not enough space for sorting: " +
            std::to_string(workspaceSize) + " / " +
            std::to_string(requiredWorkspaceBytes()));
      }

      void * const prefix_sum_space = workspace;
      size_t prefix_sum_bytes = requiredPrefixSumBytes(
          m_maxNumNodes);

      IdType * const node_prefix = reinterpret_cast<IdType*>(static_cast<uint8_t*>(prefix_sum_space) +
          prefix_sum_bytes);

      const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

      checked_memset(
        count_lhs_device,
        0,
        num_ntypes*sizeof(*count_lhs_device),
        stream);


      // launch kernel to build hash table
      for (int64_t i = 0; i < num_ntypes; ++i) {
        const IdArray& nodes = i < lhs_nodes.size() ? lhs_nodes[i] : rhs_nodes[i-lhs_nodes.size()];

        if (nodes->shape[0] > 0) {
          checked_memset(node_maps->hash_table(i), -1, node_maps->hash_size(i),
              stream);

          const dim3 grid(round_up_div(nodes->shape[0], BLOCK_SIZE));
          const dim3 block(BLOCK_SIZE);

          generate_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
              nodes.Ptr<IdType>(),
              nodes->shape[0],
              node_maps->hash_size(i),
              node_maps->hash_table(i));
          check_last_error();

          count_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
              nodes.Ptr<IdType>(),
              nodes->shape[0],
              node_maps->hash_size(i),
              node_maps->hash_table(i),
              node_prefix);
          check_last_error();

          check_cub_call(cub::DeviceScan::ExclusiveSum(
              prefix_sum_space,
              prefix_sum_bytes,
              node_prefix,
              node_prefix,
              grid.x+1, stream));

          compact_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
              nodes.Ptr<IdType>(),
              nodes->shape[0],
              node_maps->hash_size(i),
              node_maps->hash_table(i),
              node_prefix,
              lhs_device[i].Ptr<IdType>(),
              count_lhs_device+i);
          check_last_error();
        } else {
          checked_memset(
            count_lhs_device+i,
            0,
            sizeof(*count_lhs_device),
            stream);
        }
      }
    }

  private:
  IdType m_maxNumNodes;
  size_t m_workspaceSizeBytes;

};

template<typename IdType>
void map_edges(
    HeteroGraphPtr graph,
    std::vector<EdgeArray>& edge_sets,
    const DeviceNodeMap<IdType>& node_map,
    cudaStream_t stream)
{
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;

  // TODO: modify to be a single kernel launch

  for (int64_t etype = 0; etype < edge_sets.size(); ++etype) {
    EdgeArray& edges = edge_sets[etype];
    if (!aten::IsNullArray(edges.src)) {
      const int64_t num_edges = edges.src->shape[0];
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const int src_type = src_dst_types.first;
      const int dst_type = src_dst_types.second;

      const dim3 grid(round_up_div(num_edges, TILE_SIZE));
      const dim3 block(BLOCK_SIZE);

      // map the srcs
      map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE><<<
        grid,
        block,
        0,
        stream>>>(
        edges.src.Ptr<IdType>(),
        edges.dst.Ptr<IdType>(),
        num_edges,
        node_map.lhs_hash_table(src_type),
        node_map.lhs_hash_size(src_type),
        node_map.rhs_hash_table(dst_type),
        node_map.rhs_hash_size(dst_type));

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        throw std::runtime_error("Error launching kernel: " + std::to_string(err)
        + " : '" + std::string(cudaGetErrorString(err)) + "'.");
      }
    }
  }
}



template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlockInternal(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs)
{
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
    maxNodesPerType[ntype] += rhs_nodes[ntype]->shape[0];

    if (include_rhs_in_lhs) {
      maxNodesPerType[ntype+num_ntypes] += rhs_nodes[ntype]->shape[0];
    }
  }
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    maxNodesPerType[srctype+num_ntypes] += edge_arrays[etype].src->shape[0];
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

  // allocate space for map creation process
  DeviceNodeMapMaker<IdType> maker(maxNodesPerType);
  const size_t scratch_size = maker.requiredWorkspaceBytes();
  void * const scratch_device = device->AllocWorkspace(ctx, scratch_size);

  DeviceNodeMap<IdType> node_maps(maxNodesPerType, ctx);

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
  maker.make(
      src_nodes,
      rhs_nodes,
      &node_maps,
      count_lhs_device,
      lhs_nodes,
      scratch_device,
      scratch_size,
      stream);

  cudaEvent_t event;
  cudaEventCreate(&event);

  // start copy for number of nodes per type 
  int64_t * num_nodes_per_type;
  cudaMallocHost(&num_nodes_per_type, sizeof(*num_nodes_per_type)*num_ntypes*2);
  cudaMemcpyAsync(num_nodes_per_type,
      count_lhs_device, sizeof(*num_nodes_per_type)*num_ntypes*2,
      cudaMemcpyDeviceToHost,
      stream);

  cudaEventRecord(event, stream);

  // map node numberings from global to local, and build pointer for CSR
  map_edges(graph, edge_arrays, node_maps, stream);

  std::vector<IdArray> induced_edges;
  induced_edges.reserve(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].id.defined()) {
      induced_edges.push_back(edge_arrays[etype].id);
    } else {
      induced_edges.push_back(aten::NullArray());
    }
  }

  // build metagraph -- small enough to be done on CPU
  const auto meta_graph = graph->meta_graph();
  const EdgeArray etypes = meta_graph->Edges("eid");
  const IdArray new_dst = Add(etypes.dst, num_ntypes);
  const auto new_meta_graph = ImmutableGraph::CreateFromCOO(
      num_ntypes * 2, etypes.src, new_dst);

  // wait for the node counts to finish transferring
  cudaEventSynchronize(event);
  device->FreeWorkspace(ctx, count_lhs_device);

  // resize lhs nodes
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    lhs_nodes[ntype]->shape[0] = num_nodes_per_type[ntype];
  }

  // build the heterograph
  std::vector<HeteroGraphPtr> rel_graphs;
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t srctype = src_dst_types.first;
    const dgl_type_t dsttype = src_dst_types.second;

    if (rhs_nodes[dsttype].NumElements() == 0) {
      // No rhs nodes are given for this edge type. Create an empty graph.
      rel_graphs.push_back(CreateFromCOO(
          2, lhs_nodes[srctype].NumElements(), rhs_nodes[dsttype].NumElements(),
          aten::NullArray(), aten::NullArray()));
    } else {
      rel_graphs.push_back(CreateFromCOO(
          2,
          rhs_nodes[dsttype].NumElements(),
          lhs_nodes[srctype].NumElements(),
          edge_arrays[etype].src,
          edge_arrays[etype].dst));
    }
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, std::vector<int64_t>(
      num_nodes_per_type, num_nodes_per_type+num_ntypes));

  // return the new graph, the new src nodes, and new edges
  return std::make_tuple(new_graph, lhs_nodes, induced_edges);
}

}



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


}  // namespace transform
}  // namespace dgl
