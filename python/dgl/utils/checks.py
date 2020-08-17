"""Checking and logging utilities."""
# pylint: disable=invalid-name
from __future__ import absolute_import, division

from ..base import DGLError
from .. import backend as F

def prepare_tensor(g, data, name):
    """Convert the data to ID tensor and check its ID type and context.

    If the data is already in tensor type, raise error if its ID type
    and context does not match the graph's.
    Otherwise, convert it to tensor type of the graph's ID type and
    ctx and return.

    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : int, iterable of int, tensor
        Data.
    name : str
        Name of the data.

    Returns
    -------
    Tensor
        Data in tensor object.
    """
    if F.is_tensor(data):
        if F.dtype(data) != g.idtype or F.context(data) != g.device:
            raise DGLError('Expect argument "{}" to have data type {} and device '
                           'context {}. But got {} and {}.'.format(
                               name, g.idtype, g.device, F.dtype(data), F.context(data)))
        ret = data
    else:
        data = F.tensor(data)
        if F.dtype(data) not in (F.int32, F.int64):
            raise DGLError('Expect argument "{}" to have data type int32 or int64,'
                           ' but got {}.'.format(name, F.dtype(data)))
        ret = F.copy_to(F.astype(data, g.idtype), g.device)

    if F.ndim(ret) == 0:
        ret = F.unsqueeze(ret, 0)
    if F.ndim(ret) > 1:
        raise DGLError('Expect a 1-D tensor for argument "{}". But got {}.'.format(
            name, ret))
    return ret

def prepare_tensor_dict(g, data, name):
    """Convert a dictionary of data to a dictionary of ID tensors.

    If calls ``prepare_tensor`` on each key-value pair.

    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : dict[str, (int, iterable of int, tensor)]
        Data dict.
    name : str
        Name of the data.

    Returns
    -------
    dict[str, tensor]
    """
    return {key : prepare_tensor(g, val, '{}["{}"]'.format(name, key))
            for key, val in data.items()}

def parse_edges_arg_to_eid(g, edges, etid, argname='edges'):
    """Parse the :attr:`edges` argument and return an edge ID tensor.

    The resulting edge ID tensor has the same ID type and device of :attr:`g`.

    Parameters
    ----------
    g : DGLGraph
        Graph
    edges : pair of Tensor, Tensor, iterable[int]
        Argument for specifying edges.
    etid : int
        Edge type ID.
    argname : str, optional
        Argument name.

    Returns
    -------
    Tensor
        Edge ID tensor
    """
    if isinstance(edges, tuple):
        u, v = edges
        u = prepare_tensor(g, u, '{}[0]'.format(argname))
        v = prepare_tensor(g, v, '{}[1]'.format(argname))
        eid = g.edge_ids(u, v, etype=g.canonical_etypes[etid])
    else:
        eid = prepare_tensor(g, edges, argname)
    return eid

def check_all_same_idtype(glist, name):
    """Check all the graphs have the same idtype."""
    if len(glist) == 0:
        return
    idtype = glist[0].idtype
    for i, g in enumerate(glist):
        if g.idtype != idtype:
            raise DGLError('Expect {}[{}] to have {} type ID, but got {}.'.format(
                name, i, idtype, g.idtype))

def check_all_same_device(glist, name):
    """Check all the graphs have the same device."""
    if len(glist) == 0:
        return
    device = glist[0].device
    for i, g in enumerate(glist):
        if g.device != device:
            raise DGLError('Expect {}[{}] to be on device {}, but got {}.'.format(
                name, i, device, g.device))

def check_all_same_keys(dict_list, name):
    """Check all the dictionaries have the same set of keys."""
    if len(dict_list) == 0:
        return
    keys = dict_list[0].keys()
    for dct in dict_list:
        if keys != dct.keys():
            raise DGLError('Expect all {} to have the same set of keys, but got'
                           ' {} and {}.'.format(name, keys, dct.keys()))

def check_all_have_keys(dict_list, keys, name):
    """Check the dictionaries all have the given keys."""
    if len(dict_list) == 0:
        return
    keys = set(keys)
    for dct in dict_list:
        if not keys.issubset(dct.keys()):
            raise DGLError('Expect all {} to include keys {}, but got {}.'.format(
                name, keys, dct.keys()))

def check_all_same_schema(feat_dict_list, keys, name):
    """Check the features of the given keys all have the same schema.

    Suggest calling ``check_all_have_keys`` first.

    Parameters
    ----------
    feat_dict_list : list[dict[str, Tensor]]
        Feature dictionaries.
    keys : list[str]
        Keys
    name : str
        Name of this feature dict.
    """
    if len(feat_dict_list) == 0:
        return
    for fdict in feat_dict_list:
        for k in keys:
            t1 = feat_dict_list[0][k]
            t2 = fdict[k]
            if F.dtype(t1) != F.dtype(t2) or F.shape(t1)[1:] != F.shape(t2)[1:]:
                raise DGLError('Expect all features {}["{}"] to have the same data type'
                               ' and feature size, but got\n\t{} {}\nand\n\t{} {}.'.format(
                                   name, k, F.dtype(t1), F.shape(t1)[1:],
                                   F.dtype(t2), F.shape(t2)[1:]))

def check_valid_idtype(idtype):
    """Check whether the value of the idtype argument is valid (int32/int64)

    Parameters
    ----------
    idtype : data type
        The framework object of a data type.
    """
    if idtype not in [None, F.int32, F.int64]:
        raise DGLError('Expect idtype to be a framework object of int32/int64, '
                       'got {}'.format(idtype))
