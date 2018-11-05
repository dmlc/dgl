"""Built-in readout function."""

from abc import abstractmethod
from .. import backend as F
from .base import create_bundled_function_class

__all__ = ['sum_nodes', 'sum_edges', 'mean_nodes', 'mean_edges',
           'weighted_sum_nodes', 'weighted_sum_edges']

class ReadoutFunction(object):
    """Base builtin readout function class."""

    def __call__(self, batched_graph):
        """Regular computation of this builtin.

        This will be used when optimization is not available.
        """
        raise NotImplementedError

    def name(self):
        """Return the name of this builtin function."""
        raise NotImplementedError

    def is_spmv_supported(self, batched_graph):
        """Return whether the SPMV optimization is supported."""
        raise NotImplementedError


BundledReadoutFunction = create_bundled_function_class(
        'BundledReadoutFunction', ReadoutFunction)


def unbatch_node_field(batched_graph, field_name):
    bn = batched_graph.batch_num_nodes
    col = batched_graph._node_frame[field_name]
    return F.unpack(col, bn)


def unbatch_edge_field(batched_graph, field_name):
    be = batched_graph.batch_num_edges
    col = batched_graph._edge_frame[field_name]
    return F.unpack(col, be)

unbatch_funcs = {
        'node': unbatch_node_field,
        'edge': unbatch_edge_field,
        }

class SimpleReadoutFunction(ReadoutFunction):
    """Builtin readout function that aggregates either node or edge
    information on a single field.

    Parameters
    ----------
    name: str
        Name of aggregate function.
    op: callable
        The aggregate function.  Should have a single argument being the
        input field.
    in_field: str
        The input field name to be aggregated
    out_field:
        The output field name
    on: "node" or "edge"
        Whether the aggregation is on node or edge
    """

    def __init__(self, name, op, in_field, out_field, on):
        self._name = name
        self.op = op
        self.in_field = in_field
        self.out_field = out_field
        self.on = on

        self.unbatch_func = unbatch_funcs[on]
        # always enable SPMV for sum and mean.
        # NOTE: I'm using scatter_add to do the job.  Should we change the
        # terminology in general, like from "is_spmv_supported" to something
        # like "can_optimize"?
        self._is_spmv_supported = self._name in ('sum', 'mean')

    def is_spmv_supported(self, batched_graph):
        return self._is_spmv_supported

    def __call__(self, batched_graph):
        vs = self.unbatch_func(batched_graph, self.in_field)
        return {self.out_field: F.stack([self.op(v) for v in vs], 0)}

    def name(self):
        return self._name


class WeightedReadoutFunction(ReadoutFunction):
    """Builtin readout function that weights the node or edge information
    on a single field first, then aggregates the weighted information into
    a single field.

    Parameters
    ----------
    mul_op: callable
        The multiply/weighting function.  It should have two arguments,
        the first being the input field and the second being the weight
        field.
    agg_op: callable
        The aggregate function.  Should have a single argument being the
        weighted input field.
    in_field: str
        The input field name to be weighted
    weight_field: str
        The weight field name
    out_field:
        The output field name
    on: "node" or "edge"
        Whether the aggregation is on node or edge
    """

    def __init__(self, mul_op, agg_op, in_field, weight_field, out_field, on):
        self.mul_op = mul_op
        self.agg_op = agg_op
        self.in_field = in_field
        self.weight_field = weight_field
        self.out_field = out_field
        self.on = on

        self.unbatch_func = unbatch_funcs[on]
        # always enable SPMV for sum and mean.
        self._is_spmv_supported = self.agg_op in (F.sum, F.mean)

    def is_spmv_supported(self, batched_graph):
        return self._is_spmv_supported

    def __call__(self, batched_graph):
        vs = self.unbatch_func(batched_graph, in_field)
        ws = self.unbatch_func(batched_graph, weight_field)
        wvs = [self.mul_op(v, w) for v, w in zip(vs, ws)]
        return {self.out_field: F.stack([self.agg_op(wv) for wv in wvs], 0)}

    def name(self):
        return "weighted"


def simple_readout_factory(name, verb, op, on):
    docstring = \
            f"""Builtin readout function that {verb} a single {on} field.

            Parameters
            ----------
            in_: str
                The input feature name
            out: str
                The output feature name
            """
    def _func(in_, out):
        return SimpleReadoutFunction(name, op, in_, out, on)

    _func.__name__ = name
    _func.__qualname__ = name
    _func.__doc__ = docstring

    return _func


def weighted_readout_factory(name, mul_clause, mul_op, agg_noun, agg_op, on):
    docstring = \
            f"""Builtin readout function that {mul_clause},
            followed by {agg_noun}.

            Parameters
            ----------
            in_: str
                The input feature name
            weight: str
                The weight feature name
            out_: str
                The output feature name
            """
    def _func(in_, weight, out):
        return WeightedReadoutFunction(mul_op, agg_op, in_, weight, out, on)

    _func.__name__ = name
    _func.__qualname__ = name
    _func.__doc__ = docstring

    return _func


sum_nodes = simple_readout_factory('sum_nodes', 'sums', F.sum, 'node')
sum_edges = simple_readout_factory('sum_edges', 'sums', F.sum, 'edge')
mean_nodes = simple_readout_factory('mean_nodes', 'averages', F.mean, 'node')
mean_edges = simple_readout_factory('mean_edges', 'averages', F.mean, 'edge')
weighted_sum_nodes = weighted_readout_factory(
        'weighted_sum_nodes', 'multiplies the node weight to the node input',
        operator.mul, 'a sum', F.sum, 'node')
weighted_sum_edges = weighted_readout_factory(
        'weighted_sum_nodes', 'multiplies the edge weight to the edge input',
        operator.mul, 'a sum', F.sum, 'edge')
