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

    def to_dgl_block(self):
        """Transforming a data block into DGL blocks necessitates constructing a
        graphical structure and assigning features to the nodes and edges within
        the blocks.
        """
        if not self.sampled_subgraphs:
            return None

        blocks = [
            dgl.create_block(
                subgraph.node_pairs,
            )
            for subgraph in self.sampled_subgraphs
        ]

        if isinstance(self.sampled_subgraphs[0], Dict):
            # Assign node features to the outermost layer's source nodes.
            if self.node_feature:
                for (
                    node_type,
                    feature_name,
                ), feature in self.node_feature.items():
                    blocks[0].srcnodes[node_type].data[feature_name] = feature
            # Assign edge features.
            if self.edge_feature:
                for block, edge_feature in zip(blocks, self.edge_feature):
                    for (
                        edge_type,
                        feature_name,
                    ), feature in edge_feature.items():
                        block.edges[edge_type].data[feature_name] = feature
            # Assign reverse node ids to the outermost layer's source nodes.
            reverse_row_node_ids = self.sampled_subgraphs[
                0
            ].reverse_row_node_ids
            if reverse_row_node_ids:
                for node_type, reverse_ids in reverse_row_node_ids.items():
                    blocks[0].srcnodes[node_type].data[dgl.NID] = reverse_ids
            # Assign reverse edges ids.
            for block, subgraph in zip(blocks, self.sampled_subgraphs):
                if subgraph.reverse_edge_ids:
                    for (
                        edge_type,
                        reverse_ids,
                    ) in subgraph.reverse_edge_ids.items():
                        block.edges[edge_type].data[dgl.EID] = reverse_ids
        else:
            # Assign node features to the out most layer source nodes.
            if self.node_feature:
                for (_, feature_name), feature in self.node_feature.items():
                    blocks[0].srcdata[feature_name] = feature
            # Assign edge features.
            if self.edge_feature:
                for block, edge_feature in zip(blocks, self.edge_feature):
                    for (_, feature_name), feature in edge_feature.items():
                        block.edata[feature_name] = feature
            # Assign reverse node ids.
            reverse_row_node_ids = self.sampled_subgraphs[
                0
            ].reverse_row_node_ids
            if reverse_row_node_ids:
                blocks[0].srcdata[dgl.NID] = reverse_row_node_ids
            # Assign reverse edges ids.
            for block, subgraph in zip(blocks, self.sampled_subgraphs):
                if subgraph.reverse_edge_ids:
                    block.edata[dgl.EID] = subgraph.reverse_edge_ids

        return blocks
