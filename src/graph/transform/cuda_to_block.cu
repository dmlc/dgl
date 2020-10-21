/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda_to_block.cu
 * \brief Functions to convert a set of edges into a graph block with local
 * ids.
 */

namespace
{

template<typename IdType>
std::tuple<HeteroGraphPtr, std::vector<IdArray>, std::vector<IdArray>>
ToBlockInternal(
    HeteroGraphPtr graph,
    const std::vector<IdArray> &rhs_nodes,
    bool include_rhs_in_lhs,
    ToBlockMemory<IdType> * const pinned_memory)
{
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
  std::vector<DeviceNodes<IdType>> nodeSets;

  IdType totalSourceNodes;
  IdType totalDestNodes;
  IdType maxNodes;
  count_nodes(graph, rhs_nodes, edge_arrays, include_rhs_in_lhs,
      &totalSourceNodes, &totalDestNodes, &maxNodes);

  // first we need to get all of the nodes up to the GPU. We will use a unified
  // buffer with src in first half, and dst in seconds half
  const size_t total_nodes = totalDestNodes+totalSourceNodes;

  IdType * const node_buffer_device = pinned_memory->get_typed_memory(
      ToBlockMemory<IdType>::NODE_BUFFER, total_nodes);

  const size_t node_map_space_size = DeviceNodeMap<IdType>::getStorageSize(
      maxNodes,
      num_ntypes);

  void * const node_map_space = pinned_memory->get_memory(
      ToBlockMemory<IdType>::NODE_MAP, node_map_space_size);

  // we really just need the maximum number of nodes
  DeviceNodeMapMaker<IdType> maker(total_nodes);

  const size_t scratch_size = maker.requiredWorkspaceBytes();
  void * const scratch_device = pinned_memory->get_memory(
      ToBlockMemory<IdType>::SCRATCH, scratch_size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  DeviceNodeSet<IdType> nodes_device(num_ntypes*2); 
 
  // allocate space for edges on the device
  DeviceEdgeSet<IdType> edges_device = pinned_memory->copy_data_to_device(
      graph, rhs_nodes, edge_arrays, stream);

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

  DeviceNodeSet<IdType> rhs_nodes_device = pinned_memory->rhs_nodes_on_device();

  // this will create the sets of nodes, along with their prefix sum. The
  // lhs nodes will be first, and the rhs nodes will be second
  copy_source_and_dest_nodes_to_device(
      graph,
      rhs_nodes_device,
      edge_arrays,
      pinned_memory,
      &nodes_device,
      node_buffer_device,
      total_nodes,
      include_rhs_in_lhs,
      stream);

  DeviceNodeMap<IdType> node_maps(
      node_map_space,
      node_map_space_size,
      maxNodes,
      num_ntypes);

  maker.make(nodes_device, &node_maps, pinned_memory, scratch_device,
      scratch_size, stream);

  cudaEvent_t nums_copied_event;
  cudaEventCreate(&nums_copied_event);

  pinned_memory->copy_nums_to_host(stream);

  cudaEventRecord(nums_copied_event, stream);

  // map node numberings from global to local, and build pointer
  map_edges(*pinned_memory, &rhs_nodes_device, &edges_device, edge_ptrs_device, node_maps, stream);

  cudaEventSynchronize(nums_copied_event);

  pinned_memory->copy_data_to_host(stream);
  const std::vector<int64_t> num_nodes_per_type = pinned_memory->num_lhs_host();

  cudaEventDestroy(nums_copied_event);

  // this is overlapping CPU work
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


  // wait for all of our GPU work to finish -- really this is false, since
  // we're using paged memory
  checked_sync(stream);
  cudaStreamDestroy(stream);

  // copy pinned cpu data for vertices
  std::vector<IdArray> lhs_nodes(num_ntypes);
  for (int64_t ntype = 0; ntype < num_ntypes; ++ntype) {
    const size_t num = num_nodes_per_type[ntype];
    lhs_nodes[ntype] = NewIdArray(
        num,
        DLContext{kDLCPU, 0},
        sizeof(IdType)*8);
    std::copy(
        pinned_memory->lhs_host(ntype),
        pinned_memory->lhs_host(ntype)+num,
        lhs_nodes[ntype].Ptr<IdType>());
  }

  // copy pinned cpu data for edges
  std::vector<IdArray> edge_ptrs(num_etypes);
  for (int64_t etype = 0; etype < num_etypes; ++etype) {
    if (edge_arrays[etype].src.defined()) {
      const auto src_dst_types = graph->GetEndpointTypes(etype);
      const dgl_type_t dsttype = src_dst_types.second;

      const size_t num = edge_arrays[etype].src.NumElements();
      std::copy(
          pinned_memory->srcs_host(etype),
          pinned_memory->srcs_host(etype)+num,
          edge_arrays[etype].src.Ptr<IdType>());

      const IdType num_ptrs = rhs_nodes[dsttype].NumElements()+1;
      edge_ptrs[etype] = NewIdArray(
          num_ptrs,
          DLContext{kDLCPU, 0}, sizeof(IdType)*8);
      std::copy(
          pinned_memory->dsts_host()+edge_ptr_prefix[etype],
          pinned_memory->dsts_host()+edge_ptr_prefix[etype]+num_ptrs,
          edge_ptrs[etype].Ptr<IdType>());
    }
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
      // copy from pinned memory
      std::copy(pinned_memory->srcs_host(etype),
          pinned_memory->srcs_host(etype)+edge_arrays[etype].src.NumElements(),
          edge_arrays[etype].src.Ptr<IdType>());
      std::copy(pinned_memory->dsts_host(etype),
          pinned_memory->dsts_host(etype)+edge_arrays[etype].src.NumElements(),
          edge_arrays[etype].dst.Ptr<IdType>());

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
    if (sizeof(IdType) != m_sizeof_idtype) {
      clear();
      m_sizeof_idtype = sizeof(IdType);
      m_memory = new ToBlockMemory<IdType>();
    }

    ret = ToBlockInternal<IdType>(graph, rhs_nodes, include_rhs_in_lhs,
        static_cast<ToBlockMemory<IdType>*>(m_memory));
  });
  return ret;
}


