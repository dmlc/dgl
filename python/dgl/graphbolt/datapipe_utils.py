"""DataPipe utilities"""

# pylint: disable=unused-import
try:
    from torchdata.dataloader2.graph import traverse_dps

    DATAPIPE_COMPATIBILITY = "0.5"
except ImportError:
    # PyTorch 1.12-
    from torchdata.dataloader2.graph import traverse

    DATAPIPE_COMPATIBILITY = "0.4"

    def traverse_dps(datapipe):
        """Wrapper of PyTorch 1.12 ``traverse`` function to PyTorch 1.13
        ``traverse_dps`` interface.
        """
        return traverse(datapipe, True)


if DATAPIPE_COMPATIBILITY == "0.4":

    def _get_parents(result_dict, datapipe_graph):
        for k, parents in datapipe_graph.items():
            if k not in result_dict:
                result_dict[id(k)] = (
                    k,
                    [id(datapipe) for datapipe in parents.keys()],
                )
                _get_parents(result_dict, parents)

else:

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
