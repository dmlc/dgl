"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

import dgl
from dgl.utils import recursive_apply

from .base import CSCFormatBase, etype_str_to_tuple, expand_indptr
from .internal import get_attributes
from .sampled_subgraph import SampledSubgraph

__all__ = ["MiniBatch"]


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

    seeds: Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ] = None
    """
    Representation of seed items utilized in node classification tasks, link
    prediction tasks and hyperlinks tasks.
    - If `seeds` is a tensor: it indicates that the seeds originate from a
      homogeneous graph. It can be either a 1-dimensional or 2-dimensional
      tensor:
        - 1-dimensional tensor: Each element directly represents a seed node
          within the graph.
        - 2-dimensional tensor: Each row designates a seed item, which can
          encompass various entities such as edges, hyperlinks, or other graph
          components depending on the specific context.
    - If `seeds` is a dictionary: it indicates that the seeds originate from a
      heterogeneous graph. The keys should be edge or node type, and the value
      should be a tensor, which can be either a 1-dimensional or 2-dimensional
      tensor:
        - 1-dimensional tensor: Each element directly represents a seed node
        of the given type within the graph.
        - 2-dimensional tensor: Each row designates a seed item of the given
          type, which can encompass various entities such as edges, hyperlinks,
          or other graph components depending on the specific context.
    """

    indexes: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Indexes associated with seed nodes / node pairs in the graph, which
    indicates to which query a seed node / node pair belongs.
    - If `indexes` is a tensor: It indicates the graph is homogeneous. The
      value should be corresponding query to given 'seed_nodes' or
      'node_pairs'.
    - If `indexes` is a dictionary: It indicates the graph is
      heterogeneous. The keys should be node or edge type and the value should
      be corresponding query to given 'seed_nodes' or 'node_pairs'. For each
      key, indexes are consecutive integers starting from zero.
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

    compacted_seeds: Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ] = None
    """
    Representation of compacted seeds corresponding to 'seeds', where
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

    def node_ids(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """A representation of input nodes in the outermost layer. Contains all
        nodes in the `sampled_subgraphs`.
        - If `input_nodes` is a tensor: It indicates the graph is homogeneous.
        - If `input_nodes` is a dictionary: The keys should be node type and the
          value should be corresponding heterogeneous node id.
        """
        return self.input_nodes

    def num_layers(self) -> int:
        """Return the number of layers."""
        if self.sampled_subgraphs is None:
            return 0
        return len(self.sampled_subgraphs)

    def edge_ids(
        self, layer_id: int
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Get the edge ids of a layer."""
        return self.sampled_subgraphs[layer_id].original_edge_ids

    def set_node_features(
        self,
        node_features: Union[
            Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]
        ],
    ) -> None:
        """Set node features."""
        self.node_features = node_features

    def set_edge_features(
        self,
        edge_features: List[
            Union[Dict[str, torch.Tensor], Dict[Tuple[str, str], torch.Tensor]]
        ],
    ) -> None:
        """Set edge features."""
        self.edge_features = edge_features

    @property
    def blocks(self):
        """Extracts DGL blocks from `MiniBatch` to construct a graphical
        structure and ID mappings.
        """
        if not self.sampled_subgraphs:
            return None

        is_heterogeneous = isinstance(
            self.sampled_subgraphs[0].sampled_csc, Dict
        )

        # Casts to minimum dtype in-place and returns self.
        def cast_to_minimum_dtype(v: CSCFormatBase):
            # Checks if number of vertices and edges fit into an int32.
            dtype = (
                torch.int32
                if max(v.indptr.size(0) - 2, v.indices.size(0))
                <= torch.iinfo(torch.int32).max
                else torch.int64
            )
            v.indptr = v.indptr.to(dtype)
            v.indices = v.indices.to(dtype)
            return v

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
                for v in subgraph.sampled_csc.values():
                    cast_to_minimum_dtype(v)
                sampled_csc = {
                    etype_str_to_tuple(etype): (
                        "csc",
                        (
                            v.indptr,
                            v.indices,
                            torch.arange(
                                0,
                                len(v.indices),
                                device=v.indptr.device,
                                dtype=v.indptr.dtype,
                            ),
                        ),
                    )
                    for etype, v in subgraph.sampled_csc.items()
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
                sampled_csc = cast_to_minimum_dtype(subgraph.sampled_csc)
                sampled_csc = (
                    "csc",
                    (
                        sampled_csc.indptr,
                        sampled_csc.indices,
                        torch.arange(
                            0,
                            len(sampled_csc.indices),
                            device=sampled_csc.indptr.device,
                            dtype=sampled_csc.indptr.dtype,
                        ),
                    ),
                )
                num_src_nodes = original_row_node_ids.size(0)
                num_dst_nodes = original_column_node_ids.size(0)
            blocks.append(
                dgl.create_block(
                    sampled_csc,
                    num_src_nodes=num_src_nodes,
                    num_dst_nodes=num_dst_nodes,
                    node_count_check=False,
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

    @property
    def positive_node_pairs(self):
        """`positive_node_pairs` is a representation of positive graphs used for
        evaluating or computing loss in link prediction tasks.
        - If `positive_node_pairs` is a tuple: It indicates a homogeneous graph
        containing two tensors representing source-destination node pairs.
        - If `positive_node_pairs` is a dictionary: The keys should be edge type,
        and the value should be a tuple of tensors representing node pairs of the
        given type.
        """
        return self.compacted_node_pairs

    @property
    def negative_node_pairs(self):
        """`negative_node_pairs` is a representation of negative graphs used for
        evaluating or computing loss in link prediction tasks.
        - If `negative_node_pairs` is a tuple: It indicates a homogeneous graph
        containing two tensors representing source-destination node pairs.
        - If `negative_node_pairs` is a dictionary: The keys should be edge type,
        and the value should be a tuple of tensors representing node pairs of the
        given type.
        """
        # Build negative graph.
        if (
            self.compacted_negative_srcs is not None
            and self.compacted_negative_dsts is not None
        ):
            # For homogeneous graph.
            if isinstance(self.compacted_negative_srcs, torch.Tensor):
                negative_node_pairs = (
                    self.compacted_negative_srcs,
                    self.compacted_negative_dsts,
                )
            # For heterogeneous graph.
            else:
                negative_node_pairs = {
                    etype: (
                        neg_src,
                        self.compacted_negative_dsts[etype],
                    )
                    for etype, neg_src in self.compacted_negative_srcs.items()
                }
        elif (
            self.compacted_negative_srcs is not None
            and self.compacted_node_pairs is not None
        ):
            # For homogeneous graph.
            if isinstance(self.compacted_negative_srcs, torch.Tensor):
                negative_ratio = self.compacted_negative_srcs.size(1)
                negative_node_pairs = (
                    self.compacted_negative_srcs,
                    self.compacted_node_pairs[1]
                    .repeat_interleave(negative_ratio)
                    .view(-1, negative_ratio),
                )
            # For heterogeneous graph.
            else:
                negative_ratio = list(self.compacted_negative_srcs.values())[
                    0
                ].size(1)
                negative_node_pairs = {
                    etype: (
                        neg_src,
                        self.compacted_node_pairs[etype][1]
                        .repeat_interleave(negative_ratio)
                        .view(-1, negative_ratio),
                    )
                    for etype, neg_src in self.compacted_negative_srcs.items()
                }
        elif (
            self.compacted_negative_dsts is not None
            and self.compacted_node_pairs is not None
        ):
            # For homogeneous graph.
            if isinstance(self.compacted_negative_dsts, torch.Tensor):
                negative_ratio = self.compacted_negative_dsts.size(1)
                negative_node_pairs = (
                    self.compacted_node_pairs[0]
                    .repeat_interleave(negative_ratio)
                    .view(-1, negative_ratio),
                    self.compacted_negative_dsts,
                )
            # For heterogeneous graph.
            else:
                negative_ratio = list(self.compacted_negative_dsts.values())[
                    0
                ].size(1)
                negative_node_pairs = {
                    etype: (
                        self.compacted_node_pairs[etype][0]
                        .repeat_interleave(negative_ratio)
                        .view(-1, negative_ratio),
                        neg_dst,
                    )
                    for etype, neg_dst in self.compacted_negative_dsts.items()
                }
        else:
            negative_node_pairs = None
        return negative_node_pairs

    @property
    def node_pairs_with_labels(self):
        """Get a node pair tensor and a label tensor from MiniBatch. They are
        used for evaluating or computing loss. For homogeneous graph, it will
        return `(node_pairs, labels)` as result; for heterogeneous graph, the
        `node_pairs` and `labels` will both be a dict with etype as the key.
        - If it's a link prediction task, `node_pairs` will contain both
        negative and positive node pairs and `labels` will consist of 0 and 1,
        indicating whether the corresponding node pair is negative or positive.
        - If it's an edge classification task, this function will directly
        return `compacted_node_pairs` for each etype and the corresponding
        `labels`.
        - Otherwise it will return None.
        """
        if self.labels is None:
            # Link prediction.
            positive_node_pairs = self.positive_node_pairs
            negative_node_pairs = self.negative_node_pairs
            if positive_node_pairs is None or negative_node_pairs is None:
                return None
            if isinstance(positive_node_pairs, Dict):
                # Heterogeneous graph.
                node_pairs_by_etype = {}
                labels_by_etype = {}
                for etype in positive_node_pairs:
                    pos_src, pos_dst = positive_node_pairs[etype]
                    neg_src, neg_dst = negative_node_pairs[etype]
                    neg_src, neg_dst = neg_src.view(-1), neg_dst.view(-1)
                    node_pairs_by_etype[etype] = (
                        torch.cat((pos_src, neg_src), dim=0),
                        torch.cat((pos_dst, neg_dst), dim=0),
                    )
                    pos_label = torch.ones_like(pos_src)
                    neg_label = torch.zeros_like(neg_src)
                    labels_by_etype[etype] = torch.cat(
                        [pos_label, neg_label], dim=0
                    )
                return (node_pairs_by_etype, labels_by_etype)
            else:
                # Homogeneous graph.
                pos_src, pos_dst = positive_node_pairs
                neg_src, neg_dst = negative_node_pairs
                neg_src, neg_dst = neg_src.view(-1), neg_dst.view(-1)
                node_pairs = (
                    torch.cat((pos_src, neg_src), dim=0),
                    torch.cat((pos_dst, neg_dst), dim=0),
                )
                pos_label = torch.ones_like(pos_src)
                neg_label = torch.zeros_like(neg_src)
                labels = torch.cat([pos_label, neg_label], dim=0)
                return (node_pairs, labels.float())
        elif self.compacted_node_pairs is not None:
            # Edge classification.
            return (self.compacted_node_pairs, self.labels)
        else:
            return None

    def to_pyg_data(self):
        """Construct a PyG Data from `MiniBatch`. This function only supports
        node classification task on a homogeneous graph and the number of
        features cannot be more than one.
        """
        from torch_geometric.data import Data

        if self.sampled_subgraphs is None:
            edge_index = None
        else:
            col_nodes = []
            row_nodes = []
            for subgraph in self.sampled_subgraphs:
                if subgraph is None:
                    continue
                sampled_csc = subgraph.sampled_csc
                indptr = sampled_csc.indptr
                indices = sampled_csc.indices
                expanded_indptr = expand_indptr(
                    indptr, dtype=indices.dtype, output_size=len(indices)
                )
                col_nodes.append(expanded_indptr)
                row_nodes.append(indices)
            col_nodes = torch.cat(col_nodes)
            row_nodes = torch.cat(row_nodes)
            edge_index = torch.unique(
                torch.stack((row_nodes, col_nodes)), dim=1
            ).long()

        if self.node_features is None:
            node_features = None
        else:
            assert (
                len(self.node_features) == 1
            ), "`to_pyg_data` only supports single feature homogeneous graph."
            node_features = next(iter(self.node_features.values()))

        if self.seed_nodes is not None:
            if isinstance(self.seed_nodes, Dict):
                batch_size = len(next(iter(self.seed_nodes.values())))
            else:
                batch_size = len(self.seed_nodes)
        elif self.node_pairs is not None:
            if isinstance(self.node_pairs, Dict):
                batch_size = len(next(iter(self.node_pairs.values()))[0])
            else:
                batch_size = len(self.node_pairs[0])
        elif self.seeds is not None:
            if isinstance(self.seeds, Dict):
                batch_size = len(next(iter(self.seeds.values())))
            else:
                batch_size = len(self.seeds)
        else:
            batch_size = None
        pyg_data = Data(
            x=node_features,
            edge_index=edge_index,
            y=self.labels,
            batch_size=batch_size,
            n_id=self.node_ids(),
        )
        return pyg_data

    def to(self, device: torch.device):  # pylint: disable=invalid-name
        """Copy `MiniBatch` to the specified device using reflection."""

        def _to(x, device):
            return x.to(device) if hasattr(x, "to") else x

        def apply_to(x, device):
            return recursive_apply(x, lambda x: _to(x, device))

        if self.seed_nodes is not None and self.compacted_node_pairs is None:
            # Node related tasks.
            transfer_attrs = [
                "labels",
                "sampled_subgraphs",
                "node_features",
                "edge_features",
            ]
            if self.labels is None:
                # Layerwise inference
                transfer_attrs.append("seed_nodes")
        elif self.seed_nodes is None and self.compacted_node_pairs is not None:
            # Link/edge related tasks.
            transfer_attrs = [
                "labels",
                "compacted_node_pairs",
                "compacted_negative_srcs",
                "compacted_negative_dsts",
                "sampled_subgraphs",
                "node_features",
                "edge_features",
            ]
        elif self.seeds is not None:
            # Node/link/edge related tasks.
            transfer_attrs = [
                "labels",
                "sampled_subgraphs",
                "node_features",
                "edge_features",
                "compacted_seeds",
                "indexes",
                "seeds",
            ]
        else:
            # Otherwise copy all the attributes to the device.
            transfer_attrs = get_attributes(self)

        for attr in transfer_attrs:
            # Only copy member variables.
            try:
                # For read-only attributes such as blocks and
                # node_pairs_with_labels, setattr will throw an AttributeError.
                # We catch these exceptions and skip those attributes.
                setattr(self, attr, apply_to(getattr(self, attr), device))
            except AttributeError:
                continue

        return self


def _minibatch_str(minibatch: MiniBatch) -> str:
    final_str = ""
    # Get all attributes in the class except methods.
    attributes = get_attributes(minibatch)
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
            val = [str(val_str) for val_str in val]
            val = "[" + ",\n".join(val) + "]"
        elif isinstance(val, tuple):
            val = [str(val_str) for val_str in val]
            val = "(" + ",\n".join(val) + ")"
        else:
            val = str(val)
        final_str = (
            final_str + f"{name}={_add_indent(val, len(name)+1)},\n" + " " * 10
        )
    return "MiniBatch(" + final_str[:-3] + ")"
