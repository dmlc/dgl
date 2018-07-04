import dgl.backend as F
from dgl.backend import Tensor

def node_iter(n):
    n_is_container = isinstance(n, list)
    n_is_tensor = isinstance(n, Tensor)
    if n_is_tensor:
        n = F.asnumpy(n)
        n_is_tensor = False
        n_is_container = True
    if n_is_container:
        for nn in n:
            yield nn
    else:
        yield n

def edge_iter(u, v):
    u_is_container = isinstance(u, list)
    v_is_container = isinstance(v, list)
    u_is_tensor = isinstance(u, Tensor)
    v_is_tensor = isinstance(v, Tensor)
    if u_is_tensor:
        u = F.asnumpy(u)
        u_is_tensor = False
        u_is_container = True
    if v_is_tensor:
        v = F.asnumpy(v)
        v_is_tensor = False
        v_is_container = True
    if u_is_container and v_is_container:
        # many-many
        for uu, vv in zip(u, v):
            yield uu, vv
    elif u_is_container and not v_is_container:
        # many-one
        for uu in u:
            yield uu, v
    elif not u_is_container and v_is_container:
        # one-many
        for vv in v:
            yield u, vv
    else:
        yield u, v

def batch(x_list):
    x_dict = x_list[0].copy()
    for attr in x_dict:
        # TODO(gaiyu): place guards
        # TODO(gaiyu): platform-agnostic
        x_dict[attr] = F.cat([x[attr] for x in x_list], 0)
    return x_dict
