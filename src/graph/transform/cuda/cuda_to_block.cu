/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

#include "cuda_to_block.h"

#include "cuda_runtime.h"

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
class DeviceNodeMap
{
  public:
    constexpr static const int HASH_SCALING = 2;

    static size_t getStorageSize(
        const size_t num_nodes,
        const size_t num_ntypes)
    {
      const size_t total_ntypes = num_ntypes*2;

      const size_t hash_size = getHashSize(num_nodes);

      const size_t hash_bytes_per_type = hash_size*sizeof(Mapping<IdType>);
      const size_t total_hash_bytes = hash_bytes_per_type*total_ntypes;

      const size_t value_bytes_per_type = num_nodes*sizeof(IdType);
      const size_t total_value_bytes = value_bytes_per_type*total_ntypes;

      const size_t count_bytes_per_type = sizeof(IdType); 
      const size_t total_count_bytes = count_bytes_per_type*total_ntypes;

      return total_hash_bytes + total_value_bytes + total_count_bytes;
    }

    DeviceNodeMap(
        void * const workspace,
        const size_t workspace_size,
        const size_t max_num_nodes,
        const int64_t num_ntypes) :
      m_num_types(num_ntypes*2),
      m_rhs_offset(num_ntypes),
      m_masks(m_num_types),
      m_hash_prefix(m_num_types+1, 0),
      m_values_prefix(m_num_types+1, 0),
      m_hash_tables(nullptr),
      m_values(nullptr),
      m_num_nodes_ptr(nullptr) // configure in constructor
    {
      const size_t required_space = getStorageSize(max_num_nodes, num_ntypes);
      if (workspace_size < required_space) {
        throw std::runtime_error("Invalid GPU memory for DeviceNodeMap: " +
            std::to_string(workspace_size) + " / " +
            std::to_string(required_space));
      }

      const size_t hash_size = getHashSize(max_num_nodes);

      for (size_t i = 0; i < m_num_types; ++i) {
        m_masks[i] = hash_size -1;
        m_hash_prefix[i+1] = m_hash_prefix[i] + hash_size;
        m_values_prefix[i+1] = m_values_prefix[i] + max_num_nodes;
      }

      // configure workspace
      m_hash_tables = static_cast<Mapping<IdType>*>(workspace);
      m_values = reinterpret_cast<IdType*>(m_hash_tables+m_hash_prefix.back());
      m_num_nodes_ptr = m_values + m_values_prefix.back();
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

    void sources_to_host(
        std::vector<IdArray>* const lhs_nodes,
        cudaStream_t stream)
    {
      std::vector<IdType> numNodesHost(size());
      node_counts_to_host(numNodesHost.data(), numNodesHost.size(), stream); 

      // sync so we can read the node counts
      checked_sync(stream);

      for (size_t i = 0; i < m_rhs_offset; ++i) {
        (*lhs_nodes)[i] = NewIdArray(numNodesHost[i], DLContext{kDLCPU, 0}, sizeof(IdType)*8);
        checked_d2h_copy( (*lhs_nodes)[i].Ptr<IdType>(), this->lhs_nodes(i),
            numNodesHost[i], stream);
      }
      checked_sync(stream);
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
        const IdType numNodes) :
        m_maxNumNodes(round_up(numNodes,8)), // make sure its aligned
        m_workspaceSizeBytes(requiredPrefixSumBytes(m_maxNumNodes) + sizeof(IdType)*(m_maxNumNodes+1))
    {
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
        const IdArray& nodes,
        DeviceNodeMap<IdType> * const node_maps,
        IdArray * const prefix_lhs_device,
        IdArray * const lhs_device,
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

      checked_memset(prefix_lhs_device->Ptr<IdType>(), 0, 1, stream);

      // launch kernel to build hash table
      for (size_t i = 0; i < nodes.numTypes(); ++i) {
        if (nodes[i].size() > 0) {
          checked_memset(node_maps->hash_table(i), -1, node_maps->hash_size(i),
              stream);

          const dim3 grid(round_up_div(nodes[i].size(), BLOCK_SIZE));
          const dim3 block(BLOCK_SIZE);

          generate_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
              nodes[i].nodes(),
              nodes[i].size(),
              node_maps->hash_size(i),
              node_maps->hash_table(i));
          check_last_error();

          count_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
              nodes[i].nodes(),
              nodes[i].size(),
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
              nodes[i].nodes(),
              nodes[i].size(),
              node_maps->hash_size(i),
              node_maps->hash_table(i),
              node_prefix,
              lhs_device->Ptr<IdType>(),
              prefix_lhs_device->Ptr<IdType>+i);
          check_last_error();
        } else {
          checked_d2d_copy(
            prefix_lhs_device->Ptr<IdType>()+i+1,
            prefix_lhs_device->Ptr<IdType>()+i,
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
          srcs_device[etype]->Ptr<IdType>(),
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
    ToBlockMemory<IdType>& memory,
    DeviceNodeSet<IdType>* const rhs_nodes,
    DeviceEdgeSet<IdType> * const edge_sets,
    std::vector<IdType*> const edge_ptrs,
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

      DeviceNodes<IdType>& rhs = (*rhs_nodes)[dst_type];

      const dim3 grid(round_up_div(edges.size(), TILE_SIZE));
      const dim3 block(BLOCK_SIZE);

      const IdType num_rhs_nodes = rhs.size();

      // reduce the dsts to an ind_ptr, by taking the maximum index+1 per set
      // of contiguous values i, and saving it ind_ptr[i+1].
      cub::CountingInputIterator<IdType> itr(1);
      cub::Max maxOp;

      IdType * unique_values = memory.get_typed_memory(
          ToBlockMemory<IdType>::UNIQUE_DST_NODES, edges.size()+1);
      IdType * temp_edge_ptr = memory.get_typed_memory(
          ToBlockMemory<IdType>::TEMP_EDGE_PTR, num_rhs_nodes+1);
      checked_memset(temp_edge_ptr, 0, 1, stream);

      size_t storage_bytes = 0;
      cub::DeviceReduce::ReduceByKey(
          nullptr,
          storage_bytes,
          edges.dests(),
          unique_values,
          itr,
          temp_edge_ptr+1,
          unique_values+edges.size(),
          maxOp,
          edges.size(),
          stream);

      void * storage = memory.get_memory(
          ToBlockMemory<IdType>::SCRATCH, storage_bytes);
      cub::DeviceReduce::ReduceByKey(
          storage,
          storage_bytes,
          edges.dests(),
          unique_values,
          itr,
          temp_edge_ptr+1,
          unique_values+edges.size(),
          maxOp,
          edges.size(),
          stream);

      // map the unique_values to local numbers
      map_rhs_nodes<IdType, BLOCK_SIZE, TILE_SIZE><<<
        dim3(round_up_div(num_rhs_nodes, TILE_SIZE),2),
        BLOCK_SIZE,
        0,
        stream>>>(
        unique_values,
        unique_values+edges.size(),
        rhs.nodes(),
        num_rhs_nodes,
        node_map.lhs_hash_table(dst_type),
        node_map.lhs_hash_size(dst_type));

      // the next challenge, is for dsts which have no sources,
      // and need to be included in our ind_ptr array

      // so we have the subset of dst nodes with edges `unique_values`,
      // and the full set of dst nodes `rhs_nodes`, the
      // ind_ptr for the `unique_values` nodes, and a mapping of nodes to local
      // ids

      // Agl 1:
      // per dst node: 
      //  binary search the unique values
      //  saving the lower bound to the dst node index--nodes with edges
      //  will find themselves, and nodes without will find the preceding
      //  value
      remap_dsts_ptr<IdType, BLOCK_SIZE><<<
          round_up_div(num_rhs_nodes, BLOCK_SIZE), BLOCK_SIZE, 0, stream>>>(
        temp_edge_ptr,
        unique_values,
        rhs.nodes(),
        edge_ptrs[edges.etype()],
        unique_values+edges.size(),
        num_rhs_nodes
      );

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

  const auto& ctx = graph->ctx;
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

  // build src and dst lookup tables. If src and dst are to be merged, we will
  // use a single set of node types, otherwise, we will duplicate the node
  // types, with the first half being dst, and the second half being srce
  std::vector<IdArray> nodeSets;

  int64_t totalSourceNodes;
  int64_t totalDestNodes;
  int64_t maxNodes;
  count_nodes(graph, rhs_nodes, edge_arrays, include_rhs_in_lhs,
      &totalSourceNodes, &totalDestNodes, &maxNodes);

  // first we need to get all of the nodes up to the GPU. We will use a unified
  // buffer with src in first half, and dst in seconds half
  const size_t total_nodes = totalDestNodes+totalSourceNodes;

  // create a map maker, with everything it needs to know 
  DeviceNodeMapMaker<IdType> maker(rhs_nodes, edge_arrays, include_rhs_in_lhs);

  const size_t scratch_size = maker.requiredWorkspaceBytes();
  void * const scratch_device = device->AllocWorkspace(ctx, scratch_size);
 
  // setup prefix sum
  std::vector<size_t> edge_ptr_prefix(num_etypes+1,0);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    const auto src_dst_types = graph->GetEndpointTypes(etype);
    const dgl_type_t dsttype = src_dst_types.second;
    const IdType num_nodes = rhs_nodes[dsttype].NumElements();
    edge_ptr_prefix[etype+1] = edge_ptr_prefix[etype] + num_nodes;
  }
  std::vector<IdType*> edge_ptrs_device(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    edge_ptrs_device[etype] = pinned_memory->dsts_device() + edge_ptr_prefix[etype];
  }

  const size_t node_map_space_size = DeviceNodeMap<IdType>::getStorageSize(
      maxNodes,
      num_ntypes);

  void * const node_map_space = device->AllocWorkspace(ctx, node_map_space_size);

  DeviceNodeMap<IdType> node_maps(
      node_map_space,
      node_map_space_size,
      maxNodes,
      num_ntypes);

  // populate the mappings
  maker.make(nodes_device, &node_maps, pinned_memory, scratch_device,
      scratch_size, stream);

  // start copy for number of nodes per type 
  int64_t * num_nodes_per_type;
  cudaMallocHost(&num_nodes_per_type, sizeof(*num_nodes_per_type)*num_etypes);
  cudaMemcpyAsync(num_nodes_per_type,
      maker.numNodesPerTypePtr(), sizeof(*num_nodes_per_type)*num_etypes,
      cudaMemcpyDeviceToHost);


  // map node numberings from global to local, and build pointer for CSR
  map_edges(*pinned_memory, &rhs_nodes_device, &edges_device, edge_ptrs_device, node_maps, stream);

  const std::vector<int64_t> num_nodes_per_type = ?;

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
      rel_graphs.push_back(CreateFromCSC(
          2,
          rhs_nodes[dsttype].NumElements(),
          lhs_nodes[srctype].NumElements(),
          edge_ptrs[etype],
          edge_arrays[etype].src,
          edge_arrays[etype].id));
    }
  }

  const HeteroGraphPtr new_graph = CreateHeteroGraph(
      new_meta_graph, rel_graphs, num_nodes_per_type);

  // return the new graph, the new src nodes, and new edges
  return std::make_tuple(new_graph, lhs_nodes, induced_edges);
}

}



std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
CUDABlocker::ToBlock(
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
