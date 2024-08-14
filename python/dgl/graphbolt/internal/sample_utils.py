"""Utility functions for sampling."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..base import CSCFormatBase, etype_str_to_tuple, expand_indptr


def unique_and_compact(
    nodes: Union[
        List[torch.Tensor],
        Dict[str, List[torch.Tensor]],
    ],
):
    """
    Compact a list of nodes tensor.

    Parameters
    ----------
    nodes : List[torch.Tensor] or Dict[str, List[torch.Tensor]]
        List of nodes for compacting.
        the unique_and_compact will be done per type
        - If `nodes` is a list of tensor: All the tensors will do unique and
        compact together, usually it is used for homogeneous graph.
        - If `nodes` is a list of dictionary: The keys should be node type and
        the values should be corresponding nodes, the unique and compact will
        be done per type, usually it is used for heterogeneous graph.

    Returns
    -------
    Tuple[unique_nodes, compacted_node_list]
        The Unique nodes (per type) of all nodes in the input. And the compacted
        nodes list, where IDs inside are replaced with compacted node IDs.
        "Compacted node list" indicates that the node IDs in the input node
        list are replaced with mapped node IDs, where each type of node is
        mapped to a contiguous space of IDs ranging from 0 to N.
    """
    is_heterogeneous = isinstance(nodes, dict)

    def unique_and_compact_per_type(nodes):
        nums = [node.size(0) for node in nodes]
        nodes = torch.cat(nodes)
        empty_tensor = nodes.new_empty(0)
        unique, compacted, _ = torch.ops.graphbolt.unique_and_compact(
            nodes, empty_tensor, empty_tensor
        )
        compacted = compacted.split(nums)
        return unique, list(compacted)

    if is_heterogeneous:
        unique, compacted = {}, {}
        for ntype, nodes_of_type in nodes.items():
            unique[ntype], compacted[ntype] = unique_and_compact_per_type(
                nodes_of_type
            )
        return unique, compacted
    else:
        return unique_and_compact_per_type(nodes)


def compact_temporal_nodes(nodes, nodes_timestamp):
    """Compact a list of temporal nodes without unique.

    Note that since there is no unique, the nodes and nodes_timestamp are simply
    concatenated. And the compacted nodes are consecutive numbers starting from
    0.

    Parameters
    ----------
    nodes : List[torch.Tensor] or Dict[str, List[torch.Tensor]]
        List of nodes for compacting.
        the compact operator will be done per type
        - If `nodes` is a list of tensor: All the tensors will compact together,
        usually it is used for homogeneous graph.
        - If `nodes` is a list of dictionary: The keys should be node type and
        the values should be corresponding nodes, the compact will be done per
        type, usually it is used for heterogeneous graph.

    nodes_timestamp : List[torch.Tensor] or Dict[str, List[torch.Tensor]]
        List of timestamps for compacting.

    Returns
    -------
    Tuple[nodes, nodes_timestamp, compacted_node_list]

    The concatenated nodes and nodes_timestamp, and the compacted nodes list,
    where IDs inside are replaced with compacted node IDs.
    """

    def _compact_per_type(per_type_nodes, per_type_nodes_timestamp):
        nums = [node.size(0) for node in per_type_nodes]
        per_type_nodes = torch.cat(per_type_nodes)
        per_type_nodes_timestamp = torch.cat(per_type_nodes_timestamp)
        compacted_nodes = torch.arange(
            0,
            per_type_nodes.numel(),
            dtype=per_type_nodes.dtype,
            device=per_type_nodes.device,
        )
        compacted_nodes = list(compacted_nodes.split(nums))
        return per_type_nodes, per_type_nodes_timestamp, compacted_nodes

    if isinstance(nodes, dict):
        ret_nodes, ret_timestamp, compacted = {}, {}, {}
        for ntype, nodes_of_type in nodes.items():
            (
                ret_nodes[ntype],
                ret_timestamp[ntype],
                compacted[ntype],
            ) = _compact_per_type(nodes_of_type, nodes_timestamp[ntype])
        return ret_nodes, ret_timestamp, compacted
    else:
        return _compact_per_type(nodes, nodes_timestamp)


def unique_and_compact_csc_formats(
    csc_formats: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ],
    unique_dst_nodes: Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ],
    async_op: bool = False,
):
    """
    Compact csc formats and return unique nodes (per type).

    Parameters
    ----------
    csc_formats : Union[CSCFormatBase, Dict(str, CSCFormatBase)]
        CSC formats representing source-destination edges.
        - If `csc_formats` is a CSCFormatBase: It means the graph is
        homogeneous. Also, indptr and indice in it should be torch.tensor
        representing source and destination pairs in csc format. And IDs inside
        are homogeneous ids.
        - If `csc_formats` is a Dict[str, CSCFormatBase]: The keys
        should be edge type and the values should be csc format node pairs.
        And IDs inside are heterogeneous ids.
    unique_dst_nodes: torch.Tensor or Dict[str, torch.Tensor]
        Unique nodes of all destination nodes in the node pairs.
        - If `unique_dst_nodes` is a tensor: It means the graph is homogeneous.
        - If `csc_formats` is a dictionary: The keys are node type and the
        values are corresponding nodes. And IDs inside are heterogeneous ids.
    async_op: bool
        Boolean indicating whether the call is asynchronous. If so, the result
        can be obtained by calling wait on the returned future.

    Returns
    -------
    Tuple[csc_formats, unique_nodes]
        The compacted csc formats, where node IDs are replaced with mapped node
        IDs, and the unique nodes (per type).
        "Compacted csc formats" indicates that the node IDs in the input node
        pairs are replaced with mapped node IDs, where each type of node is
        mapped to a contiguous space of IDs ranging from 0 to N.

    Examples
    --------
    >>> import dgl.graphbolt as gb
    >>> N1 = torch.LongTensor([1, 2, 2])
    >>> N2 = torch.LongTensor([5, 5, 6])
    >>> unique_dst = {
    ...     "n1": torch.LongTensor([1, 2]),
    ...     "n2": torch.LongTensor([5, 6])}
    >>> csc_formats = {
    ...     "n1:e1:n2": gb.CSCFormatBase(indptr=torch.tensor([0, 2, 3]),indices=N1),
    ...     "n2:e2:n1": gb.CSCFormatBase(indptr=torch.tensor([0, 1, 3]),indices=N2)}
    >>> unique_nodes, compacted_csc_formats = gb.unique_and_compact_csc_formats(
    ...     csc_formats, unique_dst
    ... )
    >>> print(unique_nodes)
    {'n1': tensor([1, 2]), 'n2': tensor([5, 6])}
    >>> print(compacted_csc_formats)
    {"n1:e1:n2": CSCFormatBase(indptr=torch.tensor([0, 2, 3]),
                               indices=torch.tensor([0, 1, 1])),
     "n2:e2:n1": CSCFormatBase(indptr=torch.tensor([0, 1, 3]),
                               indices=torch.Longtensor([0, 0, 1]))}
    """
    is_homogeneous = not isinstance(csc_formats, dict)
    if is_homogeneous:
        csc_formats = {"_N:_E:_N": csc_formats}
        if unique_dst_nodes is not None:
            assert isinstance(
                unique_dst_nodes, torch.Tensor
            ), "Edge type not supported in homogeneous graph."
            unique_dst_nodes = {"_N": unique_dst_nodes}

    # Collect all source and destination nodes for each node type.
    indices = defaultdict(list)
    device = None
    for etype, csc_format in csc_formats.items():
        if device is None:
            device = csc_format.indices.device
        src_type, _, dst_type = etype_str_to_tuple(etype)
        assert len(unique_dst_nodes.get(dst_type, [])) + 1 == len(
            csc_format.indptr
        ), "The seed nodes should correspond to indptr."
        indices[src_type].append(csc_format.indices)
    indices = {ntype: torch.cat(nodes) for ntype, nodes in indices.items()}

    ntypes = set(indices.keys())
    dtype = list(indices.values())[0].dtype
    default_tensor = torch.tensor([], dtype=dtype, device=device)
    indice_list = []
    unique_dst_list = []
    for ntype in ntypes:
        indice_list.append(indices.get(ntype, default_tensor))
        unique_dst_list.append(unique_dst_nodes.get(ntype, default_tensor))
    dst_list = [torch.tensor([], dtype=dtype, device=device)] * len(
        unique_dst_list
    )
    unique_fn = (
        torch.ops.graphbolt.unique_and_compact_batched_async
        if async_op
        else torch.ops.graphbolt.unique_and_compact_batched
    )
    results = unique_fn(indice_list, dst_list, unique_dst_list)

    class _Waiter:
        def __init__(self, future, csc_formats):
            self.future = future
            self.csc_formats = csc_formats

        def wait(self):
            """Returns the stored value when invoked."""
            results = self.future.wait() if async_op else self.future
            csc_formats = self.csc_formats
            # Ensure there is no memory leak.
            self.future = self.csc_formats = None

            unique_nodes = {}
            compacted_indices = {}
            for i, ntype in enumerate(ntypes):
                unique_nodes[ntype], compacted_indices[ntype], _ = results[i]

            compacted_csc_formats = {}
            # Map back with the same order.
            for etype, csc_format in csc_formats.items():
                num_elem = csc_format.indices.size(0)
                src_type, _, _ = etype_str_to_tuple(etype)
                indice = compacted_indices[src_type][:num_elem]
                indptr = csc_format.indptr
                compacted_csc_formats[etype] = CSCFormatBase(
                    indptr=indptr, indices=indice
                )
                compacted_indices[src_type] = compacted_indices[src_type][
                    num_elem:
                ]

            # Return singleton for a homogeneous graph.
            if is_homogeneous:
                compacted_csc_formats = list(compacted_csc_formats.values())[0]
                unique_nodes = list(unique_nodes.values())[0]

            return unique_nodes, compacted_csc_formats

    post_processer = _Waiter(results, csc_formats)
    if async_op:
        return post_processer
    else:
        return post_processer.wait()


def _broadcast_timestamps(csc, dst_timestamps):
    """Broadcast the timestamp of each destination node to its corresponding
    source nodes."""
    return expand_indptr(
        csc.indptr, node_ids=dst_timestamps, output_size=len(csc.indices)
    )


def compact_csc_format(
    csc_formats: Union[CSCFormatBase, Dict[str, CSCFormatBase]],
    dst_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
    dst_timestamps: Optional[
        Union[torch.Tensor, Dict[str, torch.Tensor]]
    ] = None,
):
    """
    Relabel the row (source) IDs in the csc formats into a contiguous range from
    0 and return the original row node IDs per type.

    Note that
    1. The column (destination) IDs are included in the relabeled row IDs.
    2. If there are repeated row IDs, they would not be uniqued and will be
    treated as different nodes.
    3. If `dst_timestamps` is given, the timestamp of each destination node will
    be broadcasted to its corresponding source nodes.

    Parameters
    ----------
    csc_formats: Union[CSCFormatBase, Dict[str, CSCFormatBase]]
        CSC formats representing source-destination edges.
        - If `csc_formats` is a CSCFormatBase: It means the graph is
        homogeneous. Also, indptr and indice in it should be torch.tensor
        representing source and destination pairs in csc format. And IDs inside
        are homogeneous ids.
        - If `csc_formats` is a Dict[str, CSCFormatBase]: The keys
        should be edge type and the values should be csc format node pairs.
        And IDs inside are heterogeneous ids.
    dst_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]]
        Nodes of all destination nodes in the node pairs.
        - If `dst_nodes` is a tensor: It means the graph is homogeneous.
        - If `dst_nodes` is a dictionary: The keys are node type and the
        values are corresponding nodes. And IDs inside are heterogeneous ids.

    dst_timestamps: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]
        Timestamps of all destination nodes in the csc formats.
        If given, the timestamp of each destination node will be broadcasted
        to its corresponding source nodes.

    Returns
    -------
    Tuple[original_row_node_ids, compacted_csc_formats, ...]
        A tensor of original row node IDs (per type) of all nodes in the input.
        The compacted CSC formats, where node IDs are replaced with mapped node
        IDs ranging from 0 to N.
        The source timestamps (per type) of all nodes in the input if
        `dst_timestamps` is given.

    Examples
    --------
    >>> import dgl.graphbolt as gb
    >>> csc_formats = {
    ...     "n2:e2:n1": gb.CSCFormatBase(
    ...         indptr=torch.tensor([0, 1, 3]), indices=torch.tensor([5, 4, 6])
    ...     ),
    ...     "n1:e1:n1": gb.CSCFormatBase(
    ...         indptr=torch.tensor([0, 1, 3]), indices=torch.tensor([1, 2, 3])
    ...     ),
    ... }
    >>> dst_nodes = {"n1": torch.LongTensor([2, 4])}
    >>> original_row_node_ids, compacted_csc_formats = gb.compact_csc_format(
    ...     csc_formats, dst_nodes
    ... )
    >>> original_row_node_ids
    {'n1': tensor([2, 4, 1, 2, 3]), 'n2': tensor([5, 4, 6])}
    >>> compacted_csc_formats
    {'n2:e2:n1': CSCFormatBase(indptr=tensor([0, 1, 3]),
                indices=tensor([0, 1, 2]),
    ), 'n1:e1:n1': CSCFormatBase(indptr=tensor([0, 1, 3]),
                indices=tensor([2, 3, 4]),
    )}

    >>> csc_formats = {
    ...     "n2:e2:n1": gb.CSCFormatBase(
    ...         indptr=torch.tensor([0, 1, 3]), indices=torch.tensor([5, 4, 6])
    ...     ),
    ...     "n1:e1:n1": gb.CSCFormatBase(
    ...         indptr=torch.tensor([0, 1, 3]), indices=torch.tensor([1, 2, 3])
    ...     ),
    ... }
    >>> dst_nodes = {"n1": torch.LongTensor([2, 4])}
    >>> original_row_node_ids, compacted_csc_formats = gb.compact_csc_format(
    ...     csc_formats, dst_nodes
    ... )
    >>> original_row_node_ids
    {'n1': tensor([2, 4, 1, 2, 3]), 'n2': tensor([5, 4, 6])}
    >>> compacted_csc_formats
    {'n2:e2:n1': CSCFormatBase(indptr=tensor([0, 1, 3]),
                indices=tensor([0, 1, 2]),
    ), 'n1:e1:n1': CSCFormatBase(indptr=tensor([0, 1, 3]),
                indices=tensor([2, 3, 4]),
    )}

    >>> dst_timestamps = {"n1": torch.LongTensor([10, 20])}
    >>> (
    ...     original_row_node_ids,
    ...     compacted_csc_formats,
    ...     src_timestamps,
    ... ) = gb.compact_csc_format(csc_formats, dst_nodes, dst_timestamps)
    >>> src_timestamps
    {'n1': tensor([10, 20, 10, 20, 20]), 'n2': tensor([10, 20, 20])}
    """
    is_homogeneous = not isinstance(csc_formats, dict)
    has_timestamp = dst_timestamps is not None
    if is_homogeneous:
        if dst_nodes is not None:
            assert isinstance(
                dst_nodes, torch.Tensor
            ), "Edge type not supported in homogeneous graph."
            assert len(dst_nodes) + 1 == len(
                csc_formats.indptr
            ), "The seed nodes should correspond to indptr."
        offset = dst_nodes.size(0)
        original_row_ids = torch.cat((dst_nodes, csc_formats.indices))
        compacted_csc_formats = CSCFormatBase(
            indptr=csc_formats.indptr,
            indices=(
                torch.arange(
                    0,
                    csc_formats.indices.size(0),
                    device=csc_formats.indices.device,
                )
                + offset
            ),
        )

        src_timestamps = None
        if has_timestamp:
            src_timestamps = torch.cat(
                [
                    dst_timestamps,
                    _broadcast_timestamps(
                        compacted_csc_formats, dst_timestamps
                    ),
                ]
            )
    else:
        compacted_csc_formats = {}
        src_timestamps = None
        original_row_ids = {key: val.clone() for key, val in dst_nodes.items()}
        if has_timestamp:
            src_timestamps = {
                key: val.clone() for key, val in dst_timestamps.items()
            }
        for etype, csc_format in csc_formats.items():
            src_type, _, dst_type = etype_str_to_tuple(etype)
            assert len(dst_nodes.get(dst_type, [])) + 1 == len(
                csc_format.indptr
            ), "The seed nodes should correspond to indptr."
            device = csc_format.indices.device
            offset = original_row_ids.get(
                src_type, torch.tensor([], device=device)
            ).size(0)
            original_row_ids[src_type] = torch.cat(
                (
                    original_row_ids.get(
                        src_type,
                        torch.tensor(
                            [], dtype=csc_format.indices.dtype, device=device
                        ),
                    ),
                    csc_format.indices,
                )
            )
            compacted_csc_formats[etype] = CSCFormatBase(
                indptr=csc_format.indptr,
                indices=(
                    torch.arange(
                        0,
                        csc_format.indices.size(0),
                        dtype=csc_format.indices.dtype,
                        device=device,
                    )
                    + offset
                ),
            )
            if has_timestamp:
                # If destination timestamps are given, broadcast them to the
                # corresponding source nodes.
                src_timestamps[src_type] = torch.cat(
                    (
                        src_timestamps.get(
                            src_type,
                            torch.tensor(
                                [],
                                dtype=dst_timestamps[dst_type].dtype,
                                device=device,
                            ),
                        ),
                        _broadcast_timestamps(
                            csc_format, dst_timestamps[dst_type]
                        ),
                    )
                )
    if has_timestamp:
        return original_row_ids, compacted_csc_formats, src_timestamps
    return original_row_ids, compacted_csc_formats
