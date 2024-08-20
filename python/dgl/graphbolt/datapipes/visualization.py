# pylint: disable=W,C,R
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# Original source:
# https://github.com/pytorch/data/blob/v0.7.1/torchdata/datapipes/utils/_visualization.py

import itertools
from collections import defaultdict

from typing import Optional, Set, TYPE_CHECKING

from torch.utils.data.datapipes.iter.combining import _ChildDataPipe

from .utils import IterDataPipe, traverse_dps

if TYPE_CHECKING:
    import graphviz


__all__ = [
    "to_graph",
]


class Node:
    def __init__(self, dp, *, name=None):
        self.dp = dp
        self.name = name or type(dp).__name__.replace("IterDataPipe", "")
        self.childs = set()
        self.parents = set()

    def add_child(self, child):
        self.childs.add(child)
        child.parents.add(self)

    def remove_child(self, child):
        self.childs.remove(child)
        child.parents.remove(self)

    def add_parent(self, parent):
        self.parents.add(parent)
        parent.childs.add(self)

    def remove_parent(self, parent):
        self.parents.remove(parent)
        parent.childs.remove(self)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.dp)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self}-{hash(self)}"


def to_nodes(dp, *, debug: bool) -> Set[Node]:
    def recurse(dp_graph, child=None):
        for _dp_id, (dp_node, dp_parents) in dp_graph.items():
            node = Node(dp_node)
            if child is not None:
                node.add_child(child)
            yield node
            yield from recurse(dp_parents, child=node)

    def aggregate(nodes):
        groups = defaultdict(list)
        for node in nodes:
            groups[node].append(node)

        nodes = set()
        for node, group in groups.items():
            if len(group) == 1:
                nodes.add(node)
                continue

            aggregated_node = Node(node.dp)

            for duplicate_node in group:
                for child in duplicate_node.childs.copy():
                    duplicate_node.remove_child(child)
                    aggregated_node.add_child(child)

                for parent in duplicate_node.parents.copy():
                    duplicate_node.remove_parent(parent)
                    aggregated_node.add_parent(parent)

            nodes.add(aggregated_node)

        if debug:
            return nodes

        child_dp_nodes = set(
            itertools.chain.from_iterable(
                node.parents
                for node in nodes
                if isinstance(node.dp, _ChildDataPipe)
            )
        )

        if not child_dp_nodes:
            return nodes

        for node in child_dp_nodes:
            fixed_parent_node = Node(
                type(
                    str(node).lstrip("_"),
                    (IterDataPipe,),
                    dict(dp=node.dp, childs=node.childs),
                )()
            )
            nodes.remove(node)
            nodes.add(fixed_parent_node)

            for parent in node.parents.copy():
                node.remove_parent(parent)
                fixed_parent_node.add_parent(parent)

            for child in node.childs:
                nodes.remove(child)
                for actual_child in child.childs.copy():
                    actual_child.remove_parent(child)
                    actual_child.add_parent(fixed_parent_node)

        return nodes

    return aggregate(recurse(traverse_dps(dp)))


def to_graph(dp, *, debug: bool = False) -> "graphviz.Digraph":
    """Visualizes a DataPipe by returning a :class:`graphviz.Digraph`, which is a graph of the data pipeline.
    This allows you to visually inspect all the transformation that takes place in your DataPipes.

    .. note::

        The package :mod:`graphviz` is required to use this function.

    .. note::

        The most common interfaces for the returned graph object are:

        - :meth:`~graphviz.Digraph.render`: Save the graph to a file.
        - :meth:`~graphviz.Digraph.view`: Open the graph in a viewer.

    Args:
        dp: DataPipe that you would like to visualize (generally the last one in a chain of DataPipes).
        debug (bool): If ``True``, renders internal datapipes that are usually hidden from the user
            (such as ``ChildDataPipe`` of `demux` and `fork`). Defaults to ``False``.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> from torchdata.datapipes.utils import to_graph
        >>> dp = IterableWrapper(range(10))
        >>> dp1, dp2 = dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        >>> dp1 = dp1.map(lambda x: x + 1)
        >>> dp2 = dp2.filter(lambda _: True)
        >>> dp3 = dp1.zip(dp2).map(lambda t: t[0] + t[1])
        >>> g = to_graph(dp3)
        >>> g.view()  # This will open the graph in a viewer
    """
    try:
        import graphviz
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The package `graphviz` is required to be installed to use this function. "
            "Please `pip install graphviz` or `conda install -c conda-forge graphviz`."
        ) from None

    # The graph style as well as the color scheme below was copied from https://github.com/szagoruyko/pytorchviz/
    # https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py#L78-L85
    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="10",
        ranksep="0.1",
        height="0.2",
        fontname="monospace",
    )
    graph = graphviz.Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in to_nodes(dp, debug=debug):
        fillcolor: Optional[str]
        if not node.parents:
            fillcolor = "lightblue"
        elif not node.childs:
            fillcolor = "darkolivegreen1"
        else:
            fillcolor = None

        graph.node(name=repr(node), label=str(node), fillcolor=fillcolor)

        for child in node.childs:
            graph.edge(repr(node), repr(child))

    return graph
