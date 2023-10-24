"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

import dgl
from dgl.heterograph import DGLBlock
from dgl.utils import recursive_apply

from .base import etype_str_to_tuple
from .sampled_subgraph import SampledSubgraph

__all__ = ["DGLMiniBatch", "MiniBatch"]


@dataclass
class DGLMiniBatch:
    r"""A data class designed for the DGL library, encompassing all the
    necessary fields for computation using the DGL library."""

    blocks: List[DGLBlock] = None
    """A list of 'DGLBlock's, each one corresponding to one layer, representing
    a bipartite graph used for message passing.
    """

    input_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """A representation of input nodes in the outermost layer. Conatins all
       nodes in the 'blocks'.
    - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `input_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node id.
    """

    output_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """Representation of output nodes, usually also the seed nodes, used for
    sampling in the graph.
    - If `output_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `output_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node ids.
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
    """Edge features associated with the 'blocks'.
      - If keys are single strings: It means the graph is homogeneous, and the
      keys are feature names.
      - If keys are tuples: It means the graph is heterogeneous, and the keys
      are tuples of '(edge_type, feature_name)'. Note, edge type is a triplet
      of format (str, str, str).
    """

    labels: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """Labels associated with seed nodes / node pairs in the graph.
    - If `labels` is a tensor: It indicates the graph is homogeneous. The value
      are corresponding labels to given 'output_nodes' or 'node_pairs'.
    - If `labels` is a dictionary: The keys are node or edge type and the value
      should be corresponding labels to given 'output_nodes' or 'node_pairs'.
    """

    positive_node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """Representation of positive graphs used for evaluating or computing loss
    in link prediction tasks.
    - If `positive_node_pairs` is a tuple: It indicates a homogeneous graph
    containing two tensors representing source-destination node pairs.
    - If `positive_node_pairs` is a dictionary: The keys should be edge type,
    and the value should be a tuple of tensors representing node pairs of the
    given type.
    """

    negative_node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """Representation of negative graphs used for evaluating or computing loss in
    link prediction tasks.
    - If `negative_node_pairs` is a tuple: It indicates a homogeneous graph
    containing two tensors representing source-destination node pairs.
    - If `negative_node_pairs` is a dictionary: The keys should be edge type,
    and the value should be a tuple of tensors representing node pairs of the
    given type.
    """

    def __repr__(self) -> str:
        return _dgl_minibatch_str(self)

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Copy `DGLMiniBatch` to the specified device using reflection."""

        def _to(x, device):
            return x.to(device) if hasattr(x, "to") else x

        for attr in dir(self):
            # Only copy member variables.
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                setattr(
                    self,
                    attr,
                    recursive_apply(
                        getattr(self, attr), lambda x: _to(x, device)
                    ),
                )

        return self


@dataclass
class MiniBatch:
    r"""A composite data class for data structure in the graphbolt.

    It is designed to facilitate the exchange of data among different components
    involved in processing data. The purpose of this class is to unify the
    representation of input and output data across different stages, ensuring
    consistency and ease of use throughout the loading process."""

    seed_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of seed nodes used for sampling in the graph.
    - If `seed_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `seed_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node ids.
    """

    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of seed node pairs utilized in link prediction tasks.
    - If `node_pairs` is a tuple: It indicates a homogeneous graph where each
      tuple contains two tensors representing source-destination node pairs.
    - If `node_pairs` is a dictionary: The keys should be edge type, and the
      value should be a tuple of tensors representing node pairs of the given
      type.
    """

    labels: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with seed nodes / node pairs in the graph.
    - If `labels` is a tensor: It indicates the graph is homogeneous. The value
      should be corresponding labels to given 'seed_nodes' or 'node_pairs'.
    - If `labels` is a dictionary: The keys should be node or edge type and the
      value should be corresponding labels to given 'seed_nodes' or 'node_pairs'.
    """

    negative_srcs: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of negative samples for the head nodes in the link
    prediction task.
    - If `negative_srcs` is a tensor: It indicates a homogeneous graph.
    - If `negative_srcs` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    """

    negative_dsts: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of negative samples for the tail nodes in the link
    prediction task.
    - If `negative_dsts` is a tensor: It indicates a homogeneous graph.
    - If `negative_dsts` is a dictionary: The key should be edge type, and the
      value should correspond to the negative samples for head nodes of the
      given type.
    """

    sampled_subgraphs: List[SampledSubgraph] = None
    """A list of 'SampledSubgraph's, each one corresponding to one layer,
    representing a subset of a larger graph structure.
    """

    input_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """A representation of input nodes in the outermost layer. Conatins all nodes
       in the 'sampled_subgraphs'.
    - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
    - If `input_nodes` is a dictionary: The keys should be node type and the
      value should be corresponding heterogeneous node id.
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

    compacted_node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ] = None
    """
    Representation of compacted node pairs corresponding to 'node_pairs', where
    all node ids inside are compacted.
    """

    compacted_negative_srcs: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of compacted nodes corresponding to 'negative_srcs', where
    all node ids inside are compacted.
    """

    compacted_negative_dsts: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Representation of compacted nodes corresponding to 'negative_dsts', where
    all node ids inside are compacted.
    """

    def __repr__(self) -> str:
        return _minibatch_str(self)

    def _to_dgl_blocks(self):
        """Transforming a `MiniBatch` into DGL blocks necessitates constructing
        a graphical structure and ID mappings.
        """
        if not self.sampled_subgraphs:
            return None

        is_heterogeneous = isinstance(
            self.sampled_subgraphs[0].node_pairs, Dict
        )

        blocks = []
        for subgraph in self.sampled_subgraphs:
            original_row_node_ids = subgraph.original_row_node_ids
            assert (
                original_row_node_ids is not None
            ), "Missing `original_row_node_ids` in sampled subgraph."
            original_column_node_ids = subgraph.original_column_node_ids
            assert (
                original_column_node_ids is not None
            ), "Missing `original_column_node_ids` in sampled subgraph."
            if is_heterogeneous:
                node_pairs = {
                    etype_str_to_tuple(etype): v
                    for etype, v in subgraph.node_pairs.items()
                }
                num_src_nodes = {
                    ntype: nodes.size(0)
                    for ntype, nodes in original_row_node_ids.items()
                }
                num_dst_nodes = {
                    ntype: nodes.size(0)
                    for ntype, nodes in original_column_node_ids.items()
                }
            else:
                node_pairs = subgraph.node_pairs
                num_src_nodes = original_row_node_ids.size(0)
                num_dst_nodes = original_column_node_ids.size(0)
            blocks.append(
                dgl.create_block(
                    node_pairs,
                    num_src_nodes=num_src_nodes,
                    num_dst_nodes=num_dst_nodes,
                )
            )

        if is_heterogeneous:
            # Assign reverse node ids to the outermost layer's source nodes.
            for node_type, reverse_ids in self.sampled_subgraphs[
                0
            ].original_row_node_ids.items():
                blocks[0].srcnodes[node_type].data[dgl.NID] = reverse_ids
            # Assign reverse edges ids.
            for block, subgraph in zip(blocks, self.sampled_subgraphs):
                if subgraph.original_edge_ids:
                    for (
                        edge_type,
                        reverse_ids,
                    ) in subgraph.original_edge_ids.items():
                        block.edges[etype_str_to_tuple(edge_type)].data[
                            dgl.EID
                        ] = reverse_ids
        else:
            blocks[0].srcdata[dgl.NID] = self.sampled_subgraphs[
                0
            ].original_row_node_ids
            # Assign reverse edges ids.
            for block, subgraph in zip(blocks, self.sampled_subgraphs):
                if subgraph.original_edge_ids is not None:
                    block.edata[dgl.EID] = subgraph.original_edge_ids
        return blocks

    def to_dgl(self):
        """Converting a `MiniBatch` into a DGL MiniBatch that contains
        everything necessary for computation."
        """
        minibatch = DGLMiniBatch(
            blocks=self._to_dgl_blocks(),
            node_features=self.node_features,
            edge_features=self.edge_features,
            labels=self.labels,
        )
        # Need input nodes to fetch feature.
        if self.node_features is None:
            minibatch.input_nodes = self.input_nodes
        # Need output nodes to fetch label.
        if self.labels is None:
            minibatch.output_nodes = self.seed_nodes
        assert (
            minibatch.blocks is not None
        ), "Sampled subgraphs for computation are missing."

        # For link prediction tasks.
        if self.compacted_node_pairs is not None:
            minibatch.positive_node_pairs = self.compacted_node_pairs
            # Build negative graph.
            if (
                self.compacted_negative_srcs is not None
                and self.compacted_negative_dsts is not None
            ):
                # For homogeneous graph.
                if isinstance(self.compacted_negative_srcs, torch.Tensor):
                    minibatch.negative_node_pairs = (
                        self.compacted_negative_srcs.view(-1),
                        self.compacted_negative_dsts.view(-1),
                    )
                # For heterogeneous graph.
                else:
                    minibatch.negative_node_pairs = {
                        etype: (
                            neg_src.view(-1),
                            self.compacted_negative_dsts[etype].view(-1),
                        )
                        for etype, neg_src in self.compacted_negative_srcs.items()
                    }
            elif self.compacted_negative_srcs is not None:
                # For homogeneous graph.
                if isinstance(self.compacted_negative_srcs, torch.Tensor):
                    negative_ratio = self.compacted_negative_srcs.size(1)
                    minibatch.negative_node_pairs = (
                        self.compacted_negative_srcs.view(-1),
                        self.compacted_node_pairs[1].repeat_interleave(
                            negative_ratio
                        ),
                    )
                # For heterogeneous graph.
                else:
                    negative_ratio = list(
                        self.compacted_negative_srcs.values()
                    )[0].size(1)
                    minibatch.negative_node_pairs = {
                        etype: (
                            neg_src.view(-1),
                            self.compacted_node_pairs[etype][
                                1
                            ].repeat_interleave(negative_ratio),
                        )
                        for etype, neg_src in self.compacted_negative_srcs.items()
                    }
            elif self.compacted_negative_dsts is not None:
                # For homogeneous graph.
                if isinstance(self.compacted_negative_dsts, torch.Tensor):
                    negative_ratio = self.compacted_negative_dsts.size(1)
                    minibatch.negative_node_pairs = (
                        self.compacted_node_pairs[0].repeat_interleave(
                            negative_ratio
                        ),
                        self.compacted_negative_dsts.view(-1),
                    )
                # For heterogeneous graph.
                else:
                    negative_ratio = list(
                        self.compacted_negative_dsts.values()
                    )[0].size(1)
                    minibatch.negative_node_pairs = {
                        etype: (
                            self.compacted_node_pairs[etype][
                                0
                            ].repeat_interleave(negative_ratio),
                            neg_dst.view(-1),
                        )
                        for etype, neg_dst in self.compacted_negative_dsts.items()
                    }
        return minibatch

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Copy `MiniBatch` to the specified device using reflection."""

        def _to(x, device):
            return x.to(device) if hasattr(x, "to") else x

        for attr in dir(self):
            # Only copy member variables.
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                setattr(
                    self,
                    attr,
                    recursive_apply(
                        getattr(self, attr), lambda x: _to(x, device)
                    ),
                )

        return self


def _minibatch_str(minibatch: MiniBatch) -> str:
    final_str = ""
    # Get all attributes in the class except methods.

    def _get_attributes(_obj) -> list:
        attributes = [
            attribute
            for attribute in dir(_obj)
            if not attribute.startswith("__")
            and not callable(getattr(_obj, attribute))
        ]
        return attributes

    attributes = _get_attributes(minibatch)
    attributes.reverse()
    # Insert key with its value into the string.
    for name in attributes:
        val = getattr(minibatch, name)

        def _add_indent(_str, indent):
            lines = _str.split("\n")
            lines = [lines[0]] + [
                " " * (indent + 10) + line for line in lines[1:]
            ]
            return "\n".join(lines)

        # Let the variables in the list occupy one line each, and adjust the
        # indentation on top of the original if the original data output has
        # line feeds.
        if isinstance(val, list):
            if len(val) == 0:
                val = "[]"
            # Special handling of SampledSubgraphImpl data. Each element of
            # the data occupies one row and is further structured.
            elif isinstance(
                val[0],
                dgl.graphbolt.impl.sampled_subgraph_impl.SampledSubgraphImpl,
            ):
                sampledsubgraph_strs = []
                for sampledsubgraph in val:
                    ss_attributes = _get_attributes(sampledsubgraph)
                    sampledsubgraph_str = "SampledSubgraphImpl("
                    for ss_name in ss_attributes:
                        ss_val = str(getattr(sampledsubgraph, ss_name))
                        sampledsubgraph_str = (
                            sampledsubgraph_str
                            + f"{ss_name}={_add_indent(ss_val, len(ss_name)+1)},\n"
                            + " " * 20
                        )
                    sampledsubgraph_strs.append(sampledsubgraph_str[:-21] + ")")
                val = "[" + ",\n".join(sampledsubgraph_strs) + "]"
            else:
                val = [
                    _add_indent(
                        str(val_str), len(str(val_str).split("': ")[0]) - 6
                    )
                    for val_str in val
                ]
                val = "[" + ",\n".join(val) + "]"
        else:
            val = str(val)
        final_str = (
            final_str + f"{name}={_add_indent(val, len(name)+1)},\n" + " " * 10
        )
    return "MiniBatch(" + final_str[:-3] + ")"


def _dgl_minibatch_str(dglminibatch: DGLMiniBatch) -> str:
    final_str = ""
    # Get all attributes in the class except methods.

    def _get_attributes(_obj) -> list:
        attributes = [
            attribute
            for attribute in dir(_obj)
            if not attribute.startswith("__")
            and not callable(getattr(_obj, attribute))
        ]
        return attributes

    attributes = _get_attributes(dglminibatch)
    attributes.reverse()
    # Insert key with its value into the string.
    for name in attributes:
        val = getattr(dglminibatch, name)

        def _add_indent(_str, indent):
            lines = _str.split("\n")
            lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
            return "\n".join(lines)

        # Let the variables in the list occupy one line each, and adjust the
        # indentation on top of the original if the original data output has
        # line feeds.
        if isinstance(val, list):
            if len(val) == 0:
                val = "[]"
            # Special handling of blocks data. Each element of list occupies
            # one row and is further structured.
            elif name == "blocks":
                blocks_strs = []
                for block in val:
                    block_str = str(block).replace(" ", "\n")
                    block_str = _add_indent(block_str, len("Block") + 1)
                    blocks_strs.append(block_str)
                val = "[" + ",\n".join(blocks_strs) + "]"
            else:
                val = [
                    _add_indent(
                        str(val_str), len(str(val_str).split("': ")[0]) + 3
                    )
                    for val_str in val
                ]
                val = "[" + ",\n".join(val) + "]"
        else:
            val = str(val)
        final_str = (
            final_str + f"{name}={_add_indent(val, len(name)+15)},\n" + " " * 13
        )
    return "DGLMiniBatch(" + final_str[:-3] + ")"
