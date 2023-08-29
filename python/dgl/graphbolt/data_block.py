"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

import dgl

from .sampled_subgraph import SampledSubgraph

__all__ = ["DataBlock"]


@dataclass
class DataBlock:
    r"""A composite data class for data structure in the graphbolt. It is
    designed to facilitate the exchange of data among different components
    involved in processing data. The purpose of this class is to unify the
    representation of input and output data across different stages, ensuring
    consistency and ease of use throughout the loading process."""

    sampled_subgraphs: List[SampledSubgraph] = None
    """A list of 'SampledSubgraph's, each one corresponding to one layer,
    representing a subset of a larger graph structure.
    """

    node_feature: Dict[Tuple[str, str], torch.Tensor] = None
    """A representation of node features.
    Keys are tuples of '(node_type, feature_name)' and the values are
    corresponding features. Note that for a homogeneous graph, where there are
    no node types, 'node_type' should be None.
    """

    edge_feature: List[Dict[Tuple[str, str], torch.Tensor]] = None
    """Edge features associated with the 'sampled_subgraphs'.
    The keys are tuples in the format '(edge_type, feature_name)', and the
    values represent the corresponding features. In the case of a homogeneous
    graph where no edge types exist, 'edge_type' should be set to None.
    Note 'edge_type' are of format 'str:str:str'.
    """

    input_nodes: Union[
        torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]
    ] = None
    """A representation of input nodes in the outermost layer. Conatins all nodes
       in the 'sampled_subgraphs'.
    - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `input_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node id.
    """

    def to_dgl_graphs(self):
        """Transforming a data graph into DGL graphs necessitates constructing a
        graphical structure and assigning features to the nodes and edges within
        the graphs.
        """
        if not self.sampled_subgraphs:
            return None

        is_heterogeneous = isinstance(
            self.sampled_subgraphs[0].node_pairs, Dict
        )

        graphs = [
            dgl.heterograph(subgraph.node_pairs)
            if is_heterogeneous
            else dgl.graph(subgraph.node_pairs)
            for subgraph in self.sampled_subgraphs
        ]

        if is_heterogeneous:
            # Assign node features to the outermost layer's nodes.
            if self.node_feature:
                for (
                    node_type,
                    feature_name,
                ), feature in self.node_feature.items():
                    graphs[0].nodes[node_type].data[feature_name] = feature
            # Assign edge features.
            if self.edge_feature:
                for graph, edge_feature in zip(graphs, self.edge_feature):
                    for (
                        edge_type,
                        feature_name,
                    ), feature in edge_feature.items():
                        edge_type = tuple(edge_type.split(":"))
                        graph.edges[edge_type].data[feature_name] = feature
            # Assign reverse node ids to the outermost layer's nodes.
            reverse_row_node_ids = self.sampled_subgraphs[
                0
            ].reverse_row_node_ids
            if reverse_row_node_ids:
                for node_type, reverse_ids in reverse_row_node_ids.items():
                    graphs[0].nodes[node_type].data[dgl.NID] = reverse_ids
            # Assign reverse edges ids.
            for graph, subgraph in zip(graphs, self.sampled_subgraphs):
                if subgraph.reverse_edge_ids:
                    for (
                        edge_type,
                        reverse_ids,
                    ) in subgraph.reverse_edge_ids.items():
                        graph.edges[edge_type].data[dgl.EID] = reverse_ids
        else:
            # Assign node features to the outermost layer's nodes.
            if self.node_feature:
                for (_, feature_name), feature in self.node_feature.items():
                    graphs[0].ndata[feature_name] = feature
            # Assign edge features.
            if self.edge_feature:
                for graph, edge_feature in zip(graphs, self.edge_feature):
                    for (_, feature_name), feature in edge_feature.items():
                        graph.edata[feature_name] = feature
            # Assign reverse node ids.
            reverse_row_node_ids = self.sampled_subgraphs[
                0
            ].reverse_row_node_ids
            if reverse_row_node_ids is not None:
                graphs[0].ndata[dgl.NID] = reverse_row_node_ids
            # Assign reverse edges ids.
            for graph, subgraph in zip(graphs, self.sampled_subgraphs):
                if subgraph.reverse_edge_ids is not None:
                    graph.edata[dgl.EID] = subgraph.reverse_edge_ids

        return graphs
