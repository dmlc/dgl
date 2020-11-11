/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

#include "cuda_to_block.h"

#include "cuda_runtime.h"

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
class DeviceNodeMap
{
  public:
    constexpr static const int HASH_SCALING = 2;

    DeviceNodeMap(
        const std::vector<int64_t>& num_nodes,
        const int64_t num_ntypes) :
      m_num_types(num_nodes.size()),
      m_rhs_offset(m_num_types/2),
      m_masks(m_num_types),
      m_hash_prefix(m_num_types+1, 0),
      m_hash_tables(nullptr),
      m_values(num_types),
      m_num_nodes_ptr(nullptr) // configure in constructor
    {
      for (int64_t i = 0; i < m_num_types; ++i) {
        const size_t hash_size = getHashSize(num_nodes[i]);
        m_masks[i] = hash_size -1;
        m_hash_prefix[i+1] = m_hash_prefix[i] + hash_size;
      }

      // configure workspace
      m_hash_tables = device->AllocWorkspace(m_hash_prefix.back()*sizeof(Mapping<IdType>));
      m_num_nodes_ptr = device->AllocWorkspace(m_num_types*sizeof(IdType));

      for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
        m_values[ntype] = NewIdArray(num_nodes[ntype], ctx, sizeof(IdType)*8);
      }
    }

    std::vector<IdArray> get_unique_lhs_nodes(
        int64_t * const num_nodes_per_type)
    {
      std::vector<IdArray> lhs;
      lhs.reserve(m_rhs_offset);
      for (int64_t ntype = 0; ntype < m_rhs_offset; ++ntype) {
        lhs.emplace_back(m_values[ntype].CreateView(
            std::vector<int64_t>{num_nodes_per_type[ntype]}, m_values[ntype]->dtype, 0));
      }

      return lhs;
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

    void node_counts_to_host(
        std::vector<int64_t> * const counts,
        cudaStream_t stream) const
    {
      std::vector<IdType> real_counts(counts->size());
      node_counts_to_host(real_counts.data(), real_counts.size(), stream); 
      checked_sync(stream);

      for (size_t i = 0; i < real_counts.size(); ++i) {
        // up cast
        (*counts)[i] = real_counts[i];
      }
    }

    void node_counts_to_host(
        IdType * const counts,
        const size_t num_counts,
        cudaStream_t stream) const
    {
      if (num_counts != size()) {
        throw std::runtime_error("Mismiatch for node counts: " +
            std::to_string(num_counts) + " / " + std::to_string(size()));
      }

      checked_d2h_copy(counts, m_num_nodes_ptr, size(), stream);
    }

    const IdType * unique_nodes(
        const int64_t ntype) const
    {
      if (ntype >= m_values_prefix.size()) {
        throw std::runtime_error("Invalid ntype: " + std::to_string(ntype)
            + " / " + std::to_string(m_values_prefix.size()-1));
      }

      return m_values + m_values_prefix[ntype];
    }

    IdType * unique_nodes(
        const int64_t ntype)
    {
      if (ntype >= m_values_prefix.size()) {
        throw std::runtime_error("Invalid ntype: " + std::to_string(ntype)
            + " / " + std::to_string(m_values_prefix.size()-1));
      }

      return m_values + m_values_prefix[ntype];
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

    const IdType * lhs_nodes(
        const int64_t ntype) const
    {
      if (ntype >= m_rhs_offset) {
        throw std::runtime_error("Invalid lhs index: " + std::to_string(ntype)
            + " / " + std::to_string(m_rhs_offset));
      }
      return unique_nodes(ntype);
    }

    const IdType * rhs_nodes(
        const int64_t ntype) const
    {
      return m_values + m_values_prefix[m_rhs_offset+ntype];
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
    std::vector<size_t> m_values_prefix;
    Mapping<IdType> * m_hash_tables;
    IdType * m_values;
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
        IdArray& prefix_lhs_device,
        IdArray& lhs_device,
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

      checked_memset(prefix_lhs_device.Ptr<IdType>(), 0, 1, stream);

      const int64_t num_ntypes = lhs_nodes.size() + rhs_nodes.size();

      // launch kernel to build hash table
      for (size_t i = 0; i < nodes.numTypes(); ++i) {
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
              nodes.Ptr<IdType>()(),
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
              nodes.Ptr<IdType>()(),
              nodes->shape[0],
              node_maps->hash_size(i),
              node_maps->hash_table(i),
              node_prefix,
              lhs_device.Ptr<IdType>(),
              prefix_lhs_device.Ptr<IdType>+i);
          check_last_error();
        } else {
          checked_d2d_copy(
            prefix_lhs_device.Ptr<IdType>()+i+1,
            prefix_lhs_device.Ptr<IdType>()+i,
            1, stream);
        }
      }
    }

  private:
  IdType m_maxNumNodes;
  size_t m_workspaceSizeBytes;

};

// helper functions

template<typename IdType>
void copy_source_and_dest_nodes_to_device(
    HeteroGraphPtr graph,  
    const IdArray& rhs_nodes,
    const std::vector<EdgeArray>& edge_arrays,
    std::vector<IdType>& srcs_device,
    IdArray nodes_device,
    IdType * const node_buffer_device,
    const size_t total_nodes,
    const bool include_rhs_in_lhs,
    cudaStream_t stream)
{
  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();

  std::vector<IdType> node_counts(num_ntypes*2, 0);
  std::vector<std::vector<Array<IdType>>> nodes(num_ntypes*2);
  size_t totalNumNodes = 0;

  // insert dest first
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    const IdType num_dest_nodes = rhs_nodes[ntype].size();
    check_space(totalNumNodes, num_dest_nodes, total_nodes);
    totalNumNodes += num_dest_nodes;
    // add to rhs side
    node_counts[num_ntypes+ntype] += num_dest_nodes;
    nodes[num_ntypes+ntype].emplace_back(
        rhs_nodes[ntype].nodes(),
        num_dest_nodes);
    if (include_rhs_in_lhs) {
      // add to lhs side
      node_counts[ntype] += num_dest_nodes;
      nodes[ntype].emplace_back(
          rhs_nodes[ntype].nodes(),
          num_dest_nodes);
      totalNumNodes += num_dest_nodes;
    }
  }

  // then source
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].src.defined()) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;

      const IdType num_source_nodes = edge_arrays[etype].src.NumElements();
      check_space(totalNumNodes, num_source_nodes, total_nodes);
      totalNumNodes += num_source_nodes;

      // add to lhs side
      node_counts[srctype] += num_source_nodes;
      nodes[srctype].emplace_back(
          srcs_device[etype].Ptr<IdType>(),
          num_source_nodes);
    }
  }

  // prefix sum node counts
  size_t offset = 0;
  for (int64_t ntype = 0; ntype < num_ntypes*2; ++ntype) {
    nodes_device->set(DeviceNodes<IdType>(nodes[ntype], ntype,
        node_buffer_device + offset, stream));
    assert(node_counts[ntype] == (*nodes_device)[ntype].size());
    offset += node_counts[ntype];
  }
}

template<typename IdType>
void count_nodes(
    HeteroGraphPtr graph,  
    const std::vector<IdArray>& rhs_nodes,
    const std::vector<EdgeArray>& edge_arrays,
    const bool include_rhs_in_lhs,
    int64_t* const totalSourceNodesPtr,
    int64_t* const totalDestNodesPtr,
    int64_t* const maxNodesPtr)
{
  const int64_t num_etypes = graph->NumEdgeTypes();
  const int64_t num_ntypes = graph->NumVertexTypes();

  std::vector<size_t> node_counts(num_ntypes*2, 0);

  // sum number of dst nodes
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    const IdType num_nodes = rhs_nodes[ntype].NumElements();

    node_counts[num_ntypes+ntype] += num_nodes;
    if (include_rhs_in_lhs) {
      node_counts[ntype] += num_nodes;
    }
  }

  // sum number of src nodes
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].src.defined()) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t srctype = src_dst_types.first;

      node_counts[srctype] += edge_arrays[etype].src.NumElements();
    }
  }

  // find totals and max
  int64_t maxNodes = 0;
  int64_t totalSourceNodes = 0;
  int64_t totalDestNodes = 0;
  for (size_t i = 0; i < node_counts.size(); ++i) {
    if (node_counts[i] > maxNodes) {
      maxNodes = node_counts[i];
    }

    if (i < num_ntypes) {
      totalSourceNodes += node_counts[i];
    } else {
      totalDestNodes += node_counts[i];
    }
  }

  *totalSourceNodesPtr = totalSourceNodes;
  *totalDestNodesPtr = totalDestNodes;
  *maxNodesPtr = maxNodes;
}

template<typename IdType>
void map_edges(
    std::vector<EdgeArray>& edges,
    const DeviceNodeMap<IdType>& node_map,
    cudaStream_t stream)
{
  constexpr const int BLOCK_SIZE = 128;
  constexpr const size_t TILE_SIZE = 1024;

  // TODO: modify to be a single kernel launch

  for (DeviceEdges<IdType>& edges : *edge_sets) {
    if (edges.size() > 0) {
      const int src_type = edges.src_type();
      const int dst_type = edges.dst_type();

      const dim3 grid(round_up_div(edges.size(), TILE_SIZE));
      const dim3 block(BLOCK_SIZE);


      // map the srcs
      map_edge_ids<IdType, BLOCK_SIZE, TILE_SIZE><<<
        grid,
        block,
        0,
        stream>>>(
        edges.sources(),
        edges.size(),
        node_map.lhs_hash_table(src_type),
        node_map.lhs_hash_size(src_type));

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

  DeviceNodeMap<IdType> node_maps(maxNodesPerType);


  // populate the mappings
  maker.make(
      src_nodes,
      rhs_nodes,
      &node_maps, scratch_device, scratch_size, stream);

  // start copy for number of nodes per type 
  int64_t * num_nodes_per_type;
  cudaMallocHost(&num_nodes_per_type, sizeof(*num_nodes_per_type)*num_etypes);
  cudaMemcpyAsync(num_nodes_per_type,
      maker.numNodesPerTypePtr(), sizeof(*num_nodes_per_type)*num_etypes,
      cudaMemcpyDeviceToHost);


  // map node numberings from global to local, and build pointer for CSR
  map_edges(edge_arrays, node_maps, stream);

  std::vector<IdArray> lhs_nodes = node_maps.get_unique_lhs_nodes(
      num_nodes_per_type);

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
