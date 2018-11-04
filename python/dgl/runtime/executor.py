"""Executors used by runtime."""
from __future__ import absolute_import

from collections import namedtuple
from abc import abstractmethod

FrameData = namedtuple('FrameData', ['frame', 'fields', 'ids'])

class Executor(object):
    """Executor is the basic unit of computation.

    An executor transforms the node and edge features into new ones. An executor
    can operate only on a part of the feature space of node/edge space.

    Parameters
    ----------
    graph_data : GraphData
        A key-value store of graph structure data (e.g. adjmat index, incmat index)
    """
    def __init__(self, graph_data): # XXX: udf does not need graph_data
        self._graph_data = graph_data
        self._graph_key = None
        self._node_input = None
        self._edge_input = None
        self._node_output = None
        self._edge_output = None

    @property
    def graph_data(self):
        """Return the graph data."""
        return self._graph_data

    def set_graph_key(self, key):
        """Set the key to the structure data this executor requires.

        For example, the subclass can get the adjmat of the graph as follows:
        >>> def run(self):
        >>>     adjmat = self.graph_data[self.graph_key]
        >>>     ...

        Parameters
        ----------
        """
        self._graph_key = key

    @property
    def graph_key(self):
        """Return the graph key."""
        return self._graph_key

    def set_node_input(self, node_frame, fields=None, ids=None):
        """Set the input node features.

        Parameters
        ----------
        node_frame : frame.FrameRef
            The frame containing node features.
        fields : list of str, optional
            The fields to be read. If none, all fields are required.
        ids : utils.Index, optional
            The node ids. If none, all the nodes are required.
        """
        self._node_input = FrameData(node_frame, fields, ids)

    def set_edge_input(self, edge_frame, fields=None, ids=None):
        """Set the input edge features.

        Parameters
        ----------
        edge_frame : frame.FrameRef
            The frame containing edge features.
        fields : list of str, optional
            The fields to be read. If none, all fields are required.
        ids : utils.Index, optional
            The edge ids. If none, all the edges are required.
        """
        self._edge_input = FrameData(edge_frame, fields, ids)

    def set_node_output(self, node_frame, fields=None, ids=None):
        """Set the output node features.

        Parameters
        ----------
        node_frame : frame.FrameRef
            The frame to write to.
        fields : list of str, optional
            The output field. If none, the output fields cannot be determined
            before running (e.g. UDFs).
        ids : utils.Index
            The node ids. If none, all the nodes are updated.
        """
        self._node_output = FrameData(node_frame, fields, ids)

    def set_edge_output(self, edge_frame, edge_field=None, edge_ids=None):
        """Set the output edge features.

        Parameters
        ----------
        edge_frame : frame.FrameRef
            The frame to write to.
        fields : list of str, optional
            The output field. If none, the output fields cannot be determined
            before running (e.g. UDFs).
        ids : utils.Index
            The node ids. If none, all the edges are updated.
        """
        self._edge_output = FrameData(edge_frame, fields, ids)

    def read_node_input(self):
        """Read the input and return the feature data.

        Returns
        -------
        dict
            The node feature data.
        """
        fd = self._node_input
        if fd.fields is None and fd.ids is None:
            return fd.frame
        elif fd.fields is None and fd.ids is not None:
            return fd.frame[fd.ids]
        elif fd.fields is not None and fd.ids is None:
            return {fld : fd.frame[fld] for fld in fd.fields}
        else:
            sub = fd.frame[fd.ids]
            return {fld : sub[fld] for fld in fd.fields}

    def read_edge_input(self):
        # TODO(minjie)
        pass

    def write_node_output(self, data):
        """Write the data to the output.
        """
        # TODO(minjie)
        pass

    def write_edge_output(self, data):
        # TODO(minjie)
        pass

    @abstractmethod
    def run(self):
        """Run the executor and compute new features.

        This method should be inherited by subclass. A normal procedure looks like this:
        (1) Use read_[node|edge]_input to get the feature data.
        (2) [optional] Use self.graph_data[self.graph_key] to get the graph tensor.
        (3) Call the pre-bound function.
        (4) Use write_[node|edge]_output to write the output.
        """
        raise RuntimeError('The "run" method is not implemented by subclass.')

class SPMVExecutor(Executor):
    def __init__(self, graph_data, use_edge_feat=False):
        pass

    def run(self):
        # check context of node/edge data
        # adjmat = self.data_store[self.key].get(ctx)
        if self.use_edge_feat:
            # create adjmat using edge frame and edge field
            pass

# Two UDF executors
class NodeExecutor(Executor):
    pass

class EdgeExecutor(Executor):
    pass

class GraphData(object):
    """A component that stores the sparse matrices computed from graphs."""
    def __getitem__(self, key):
        # TODO(minjie):
        pass

class ExecutionPlan(object):
    """The class to represent execution plan.

    The execution plan contains multiple stages. Each stage contains multiple executors
    that computes different feature data. The output of these executors will be merged
    by a merge-n-apply operation before passed to the next stage.
    """
    def __init__(self):
        self._executors = []

    def add_stage(self, execs, apply_node_func, apply_edge_func):
        # XXX: apply_edge_func not needed
        """Add one stage to the plan.

        Parameters
        ----------
        execs : list of Executor
            The executors in this stage.
        apply_node_func : callable
            The apply_node_func. This can be None.
        apply_edge_func : callable
            The apply_edge_func. This can be None.
        """
        # TODO(minjie):
        pass

class MergeAndApplyExecutor(Executor):
    # TODO(minjie)

    # Things to be covered in merge
    # 1. Merge node dim: (Does runtime has enough info to find out how to merge?)
    #   a) if deg bucketing, check zero degree
    #   b) for both spmv and deg bucketing, if destination is a subset of nodes, get
    #      other nodes not in recv nodes for outputting a full frame
    # 2. Merge field dimension

    # If apply_func is removed from recv, then ignore the following:
    # Otherwise, apply_nodes should be happening half way during merge
    # You should:
    # 1. Merge all received nodes (send_and_recv and recv case)
    # 2. perform apply
    # 3. Incorporate nodes that not received to form a full frame
    pass
