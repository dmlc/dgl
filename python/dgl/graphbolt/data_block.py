"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

import dgl

from .base import etype_str_to_tuple
from .sampled_subgraph import SampledSubgraph

__all__ = ["DataBlock", "NodeClassificationBlock", "LinkPredictionBlock"]


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

    node_features: Union[
        Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]
    ] = None
    """A representation of node features.
      - If keys are single strings: It means the graph is homogeneous, and the
      keys are feature names.
      - If keys are tuples: It means the graph is heterogeneous, and the keys
      are tuples of '(node_type, feature_name)'.
    """

    edge_features: List[
        Union[Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]]
    ] = None
    """Edge features associated with the 'sampled_subgraphs'.
      - If keys are single strings: It means the graph is homogeneous, and the
      keys are feature names.
      - If keys are tuples: It means the graph is heterogeneous, and the keys
      are tuples of '(edge_type, feature_name)'. Note, edge type is single
      string of format 'str:str:str'.
    """

    input_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
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
                        edge_type = etype_str_to_tuple(edge_type)
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


@dataclass
class NodeClassificationBlock(DataBlock):
    r"""A subclass of 'UnifiedDataStruct', specialized for handling node level
    tasks."""

    seed_node: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of seed nodes used for sampling in the graph.
    - If `seed_node` is a tensor: It indicates the graph is homogeneous.
    - If `seed_node` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node ids.
    """

    label: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with seed nodes in the graph.
    - If `label` is a tensor: It indicates the graph is homogeneous.
    - If `label` is a dictionary: The keys should be node type and the
      value should be corresponding node labels to given 'seed_node'.
    """


@dataclass
class LinkPredictionBlock(DataBlock):
    r"""A subclass of 'UnifiedDataStruct', specialized for handling edge level
    tasks."""

    node_pair: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of seed node pairs utilized in link prediction tasks.
    - If `node_pair` is a tuple: It indicates a homogeneous graph where each
      tuple contains two tensors representing source-destination node pairs.
    - If `node_pair` is a dictionary: The keys should be edge type, and the
      value should be a tuple of tensors representing node pairs of the given
      type.
    """

    label: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with the link prediction task.
    - If `label` is a tensor: It indicates a homogeneous graph. The value are
      edge labels corresponding to given 'node_pair'.
    - If `label` is a dictionary: The keys should be edge type, and the value
      should correspond to given 'node_pair'.
    """

    negative_head: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of negative samples for the head nodes in the link
    prediction task.
    - If `negative_head` is a tensor: It indicates a homogeneous graph.
    - If `negative_head` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    """

    negative_tail: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of negative samples for the tail nodes in the link
    prediction task.
    - If `negative_tail` is a tensor: It indicates a homogeneous graph.
    - If `negative_tail` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    """

    compacted_node_pair: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of compacted node pairs corresponding to 'node_pair', where
    all node ids inside are compacted.
    """

    compacted_negative_head: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of compacted nodes corresponding to 'negative_head', where
    all node ids inside are compacted.
    """

    compacted_negative_tail: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of compacted nodes corresponding to 'negative_tail', where
    all node ids inside are compacted.
    """
