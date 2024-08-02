"""DataPipe utilities"""

from typing import List, Set, Type

from torch.utils.data.graph import DataPipe, DataPipeGraph

__all__ = ["datapipe_graph_to_adjlist", "find_dps", "replace_dp"]


def _get_parents(result_dict, datapipe_graph):
    for k, (v, parents) in datapipe_graph.items():
        if k not in result_dict:
            result_dict[k] = (v, list(parents.keys()))
            _get_parents(result_dict, parents)


def datapipe_graph_to_adjlist(datapipe_graph):
    """Given a DataPipe graph returned by
    :func:`torch.utils.data.graph.traverse_dps` in DAG form, convert it into
    adjacency list form.

    Namely, :func:`torch.utils.data.graph.traverse_dps` returns the following
    data structure:

    .. code::

       {
           id(datapipe): (
               datapipe,
               {
                   id(parent1_of_datapipe): (parent1_of_datapipe, {...}),
                   id(parent2_of_datapipe): (parent2_of_datapipe, {...}),
                   ...
               }
           )
       }

    We convert it into the following for easier access:

    .. code::

       {
           id(datapipe1): (
               datapipe1,
               [id(parent1_of_datapipe1), id(parent2_of_datapipe1), ...]
           ),
           id(datapipe2): (
               datapipe2,
               [id(parent1_of_datapipe2), id(parent2_of_datapipe2), ...]
           ),
           ...
       }
    """

    result_dict = {}
    _get_parents(result_dict, datapipe_graph)
    return result_dict


# Copied from:
# https://github.com/pytorch/data/blob/88c8bdc6662f37649b7ea5df0bd90a4b24a56876/torchdata/dataloader2/graph/utils.py#L16
def find_dps(graph: DataPipeGraph, dp_type: Type[DataPipe]) -> List[DataPipe]:
    r"""
    Given the graph of DataPipe generated by ``traverse_dps`` function, return DataPipe
    instances with the provided DataPipe type.
    """
    dps: List[DataPipe] = []
    cache: Set[int] = set()

    def helper(g) -> None:  # pyre-ignore
        for dp_id, (dp, src_graph) in g.items():
            if dp_id in cache:
                continue
            cache.add(dp_id)
            if (
                type(dp) is dp_type
            ):  # Please not use `isinstance`, there is a bug.
                dps.append(dp)
            helper(src_graph)

    helper(graph)

    return dps


# Copied from:
# https://github.com/pytorch/data/blob/88c8bdc6662f37649b7ea5df0bd90a4b24a56876/torchdata/dataloader2/graph/utils.py#L82
# Given the DataPipe needs to be replaced and the expected DataPipe, return a new graph
def replace_dp(
    graph: DataPipeGraph, old_datapipe: DataPipe, new_datapipe: DataPipe
) -> DataPipeGraph:
    r"""
    Given the graph of DataPipe generated by ``traverse_dps`` function and the DataPipe to be replaced and
    the new DataPipe, return the new graph of DataPipe.
    """
    assert len(graph) == 1

    if id(old_datapipe) in graph:
        graph = traverse_dps(new_datapipe)

    final_datapipe = list(graph.values())[0][0]

    for recv_dp, send_graph in graph.values():
        _replace_dp(recv_dp, send_graph, old_datapipe, new_datapipe)

    return traverse_dps(final_datapipe)
