import torch, dgl

def embed(hgraph, embeddors, entities_name, type_, start_idx, end_idx):
    """
    Args:
    entities_name - "nodes" or "edges".
    type_ - "atom", "motif", or "attachment_config" for nodes,
        otherwise canonical (ntype, etype, ntype) of the hgraph.
    start_idx - Get embeddings for the specified entities starting at this index.
    end_idx - Get embeddings for the specified entities from start_idx upto and including this index.
    """
    
    entities = getattr(hgraph, entities_name)
    entity_data = entities[type_].data
    embeddor_type = type_[0] if type(type_) == tuple else type_
    embeddor = embeddors[entities_name][embeddor_type]
    entity_data["rep"][start_idx: end_idx + 1] = embeddor(entity_data["vocab_idx"][start_idx: end_idx + 1])

def init_features_for_hgraph(hgraph, hidden_size, rep_size):
    for node_type in hgraph.ntypes:
        num_nodes = hgraph.number_of_nodes(node_type)
        if "vocab_idx" not in hgraph.nodes[node_type].data:
            hgraph.nodes[node_type].data["vocab_idx"] = torch.zeros(num_nodes, device = hgraph.device).long()
        hgraph.nodes[node_type].data["rep"] = torch.zeros(num_nodes, rep_size, device = hgraph.device)
        hgraph.nodes[node_type].data["sum_incoming_hidden_and_cell"] = torch.zeros(num_nodes, 2, hidden_size, device = hgraph.device)
        hgraph.nodes[node_type].data["sum_prev_level_reps"] = torch.zeros(num_nodes, rep_size, device = hgraph.device)

    for edge_type in hgraph.canonical_etypes:
        num_edges = hgraph.number_of_edges(edge_type)
        if (edge_type[1] == "attaches to" and
            "attachment_motif_id_pair" not in hgraph.edges[edge_type].data):
            hgraph.edges[edge_type].data["attachment_motif_id_pair"] = torch.zeros(
                hgraph.number_of_edges(edge_type), 2, device = hgraph.device
            ).long()
        elif edge_type[1] != "attaches to":    
            if "vocab_idx" not in hgraph.edges[edge_type].data:
                hgraph.edges[edge_type].data["vocab_idx"] = torch.zeros(num_edges, device = hgraph.device).long()
            hgraph.edges[edge_type].data["rep"] = torch.zeros(num_edges, rep_size, device = hgraph.device)

def get_empty_hgraph(hidden_size, rep_size, device):
    hgraph = dgl.heterograph({
            ("atom", "bond", "atom"): ([], []),
            ("atom", "of", "attachment_config"): ([], []),
            ("attachment_config", "attaches to", "attachment_config"): ([], []),
            ("attachment_config", "ancestry", "attachment_config"): ([], []),
            ("attachment_config", "of", "motif"): ([], []),
            ("motif", "attaches to", "motif"): ([], []),
            ("motif", "ancestry", "motif"): ([], [])
    }).to(device)

    init_features_for_hgraph(hgraph, hidden_size, rep_size)

    return hgraph

def add_graph_to_graph(main_graph, add_graph):
    """
    Adds the edges, nodes, and data of the add_graph to the main_graph.
    """

    for node_type in main_graph.ntypes:
        data = add_graph.nodes[node_type].data
        main_graph.add_nodes(
            add_graph.number_of_nodes(ntype = node_type),
            data if data != {} else None,
            node_type
        )
   
    for edge_type in main_graph.canonical_etypes:
        src_type, _, dst_type = edge_type
        edges_src_idxs, edges_dst_idxs = add_graph.edges(etype = edge_type)
        new_edges_src_idxs = (main_graph.number_of_nodes(ntype = src_type)
                              - add_graph.number_of_nodes(ntype = src_type)
                              + edges_src_idxs)
        new_edges_dst_idxs = (main_graph.number_of_nodes(ntype = dst_type)
                              - add_graph.number_of_nodes(ntype = dst_type)
                              + edges_dst_idxs)
        data = add_graph.edges[edge_type].data
        main_graph.add_edges(
            new_edges_src_idxs,
            new_edges_dst_idxs,
            data if data != {} else None,
            edge_type
        )
   
    return main_graph
