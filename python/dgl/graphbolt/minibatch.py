"""Unified data structure for input and ouput of all the stages in loading process."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

from .base import (
    apply_to,
    CSCFormatBase,
    etype_str_to_tuple,
    expand_indptr,
    is_object_pinned,
)
from .internal_utils import (
    get_attributes,
    get_nonproperty_attributes,
    recursive_apply,
)
from .sampled_subgraph import SampledSubgraph

__all__ = ["MiniBatch"]


@dataclass
class MiniBatch:
    r"""A composite data class for data structure in the graphbolt.

    It is designed to facilitate the exchange of data among different components
    involved in processing data. The purpose of this class is to unify the
    representation of input and output data across different stages, ensuring
    consistency and ease of use throughout the loading process."""

    labels: Union[torch.Tensor, Dict[str, torch.Tensor]] = None
    """
    Labels associated with seeds in the graph.
    - If `labels` is a tensor: It indicates the graph is homogeneous. The value
      should be corresponding labels to given 'seeds'.
    - If `labels` is a dictionary: The keys should be node or edge type and the
      value should be corresponding labels to given 'seeds'.
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
    Indexes associated with seeds in the graph, which
    indicates to which query a seeds belongs.
    - If `indexes` is a tensor: It indicates the graph is homogeneous. The
      value should be corresponding query to given 'seeds'.
    - If `indexes` is a dictionary: It indicates the graph is heterogeneous.
      The keys should be node or edge type and the value should be
      corresponding query to given 'seeds'. For each key, indexes are
      consecutive integers starting from zero.
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

    compacted_seeds: Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ] = None
    """
    Representation of compacted seeds corresponding to 'seeds', where
    all node ids inside are compacted.
    """

    _blocks: list = None
    """
    A list of `DGLBlock`s.
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
    def blocks(self) -> list:
        """DGL blocks extracted from `MiniBatch` containing graphical structures
        and ID mappings.
        """
        if not self.sampled_subgraphs:
            return None

        if self._blocks is None:
            self._blocks = self.compute_blocks()
        return self._blocks

    def compute_blocks(self) -> list:
        """Extracts DGL blocks from `MiniBatch` to construct graphical
        structures and ID mappings.
        """
        from dgl.convert import create_block, EID, NID

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
                node_types = set()
                sampled_csc = {}
                for v in subgraph.sampled_csc.values():
                    cast_to_minimum_dtype(v)
                for etype, v in subgraph.sampled_csc.items():
                    etype_tuple = etype_str_to_tuple(etype)
                    node_types.add(etype_tuple[0])
                    node_types.add(etype_tuple[2])
                    sampled_csc[etype_tuple] = (
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
                num_src_nodes = {
                    ntype: (
                        original_row_node_ids[ntype].size(0)
                        if original_row_node_ids.get(ntype) is not None
                        else 0
                    )
                    for ntype in node_types
                }
                num_dst_nodes = {
                    ntype: (
                        original_column_node_ids[ntype].size(0)
                        if original_column_node_ids.get(ntype) is not None
                        else 0
                    )
                    for ntype in node_types
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
                create_block(
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
                blocks[0].srcnodes[node_type].data[NID] = reverse_ids
            # Assign reverse edges ids.
            for block, subgraph in zip(blocks, self.sampled_subgraphs):
                if subgraph.original_edge_ids is not None:
                    for (
                        edge_type,
                        reverse_ids,
                    ) in subgraph.original_edge_ids.items():
                        block.edges[etype_str_to_tuple(edge_type)].data[
                            EID
                        ] = reverse_ids
        else:
            blocks[0].srcdata[NID] = self.sampled_subgraphs[
                0
            ].original_row_node_ids
            # Assign reverse edges ids.
            for block, subgraph in zip(blocks, self.sampled_subgraphs):
                if subgraph.original_edge_ids is not None:
                    block.edata[EID] = subgraph.original_edge_ids
        return blocks

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

        if self.seeds is not None:
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

    def to(
        self, device: torch.device, non_blocking=False
    ):  # pylint: disable=invalid-name
        """Copy `MiniBatch` to the specified device using reflection."""

        copy_fn = lambda x: apply_to(x, device, non_blocking=non_blocking)

        transfer_attrs = get_nonproperty_attributes(self)

        for attr in transfer_attrs:
            # Only copy member variables.
            setattr(self, attr, recursive_apply(getattr(self, attr), copy_fn))

        return self

    def pin_memory(self):
        """Copy `MiniBatch` to the pinned memory using reflection."""

        return self.to("pinned")

    def is_pinned(self) -> bool:
        """Check whether `SampledSubgraph` is pinned using reflection."""

        return is_object_pinned(self)


def _minibatch_str(minibatch: MiniBatch) -> str:
    final_str = ""
    # Get all attributes in the class except methods.
    attributes = get_attributes(minibatch)
    attributes.reverse()
    # Insert key with its value into the string.
    for name in attributes:
        if name[0] == "_":
            continue
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
