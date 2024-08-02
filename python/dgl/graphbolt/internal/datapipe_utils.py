"""DataPipe utilities"""

from typing import List, Set, Type

from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps

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
            # Please not use `isinstance`, there is a bug.
            if type(dp) is dp_type:  # pylint: disable=unidiomatic-typecheck
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
    Given the graph of DataPipe generated by ``traverse_dps`` function and the
    DataPipe to be replaced and the new DataPipe, return the new graph of
    DataPipe.
    """
    assert len(graph) == 1

    if id(old_datapipe) in graph:
        graph = traverse_dps(new_datapipe)

    final_datapipe = list(graph.values())[0][0]

    for recv_dp, send_graph in graph.values():
        _replace_dp(recv_dp, send_graph, old_datapipe, new_datapipe)

    return traverse_dps(final_datapipe)


# For each `recv_dp`, find if the source_datapipe needs to be replaced by the new one.
# If found, find where the `old_dp` is located in `recv_dp` and switch it to the `new_dp`
def _replace_dp(
    recv_dp, send_graph: DataPipeGraph, old_dp: DataPipe, new_dp: DataPipe
) -> None:
    old_dp_id = id(old_dp)
    for send_id in send_graph:
        if send_id == old_dp_id:
            _assign_attr(recv_dp, old_dp, new_dp, inner_dp=True)
        else:
            send_dp, sub_send_graph = send_graph[send_id]
            _replace_dp(send_dp, sub_send_graph, old_dp, new_dp)


# Recursively re-assign datapipe for the sake of nested data structure
# `inner_dp` is used to prevent recursive call if we have already met a `DataPipe`
def _assign_attr(obj, old_dp, new_dp, inner_dp: bool = False):
    if obj is old_dp:
        return new_dp
    elif isinstance(obj, (IterDataPipe, MapDataPipe)):
        # Prevent recursive call for DataPipe
        if not inner_dp:
            return None
        for k in list(obj.__dict__.keys()):
            new_obj = _assign_attr(obj.__dict__[k], old_dp, new_dp)
            if new_obj is not None:
                obj.__dict__[k] = new_obj
                break
        return None
    elif isinstance(obj, dict):
        for k in list(obj.keys()):
            new_obj = _assign_attr(obj[k], old_dp, new_dp)
            if new_obj is not None:
                obj[k] = new_obj
                break
        return None
    # Tuple is immutable, has to re-create a tuple
    elif isinstance(obj, tuple):
        temp_list = []
        flag = False
        for item in obj:
            new_obj = _assign_attr(item, old_dp, new_dp, inner_dp)
            if new_obj is not None:
                flag = True
                temp_list.append(new_dp)
            else:
                temp_list.append(item)
        if flag:
            return tuple(temp_list)  # Special case
        else:
            return None
    elif isinstance(obj, list):
        for i in range(len(obj)):  # pylint: disable=consider-using-enumerate
            new_obj = _assign_attr(obj[i], old_dp, new_dp, inner_dp)
            if new_obj is not None:
                obj[i] = new_obj
                break
        return None
    elif isinstance(obj, set):
        new_obj = None
        for item in obj:
            if _assign_attr(item, old_dp, new_dp, inner_dp) is not None:
                new_obj = new_dp
                break
        if new_obj is not None:
            obj.remove(old_dp)
            obj.add(new_dp)
        return None
    else:
        return None
