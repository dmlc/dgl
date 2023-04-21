import torch as th

"""Compute x,y coordinate for nodes in the graph"""
eps = 1e-8


def get_coordinates(graphs, grid_side, coarsening_levels, perm):
    rst = []
    for l in range(coarsening_levels + 1):
        xs, ys = [], []
        for i in range(graphs[l].num_nodes()):
            cnt = eps
            x_accum = 0
            y_accum = 0
            for j in range(i * 2**l, (i + 1) * 2**l):
                if perm[j] < grid_side**2:
                    x_accum += perm[j] // grid_side
                    y_accum += perm[j] % grid_side
                    cnt += 1
            xs.append(x_accum / cnt)
            ys.append(y_accum / cnt)
        rst.append(
            th.cat([th.tensor(xs).view(-1, 1), th.tensor(ys).view(-1, 1)], -1)
        )
    return rst


"""Cartesian coordinate to polar coordinate"""


def z2polar(edges):
    z = edges.dst["xy"] - edges.src["xy"]
    rho = th.norm(z, dim=-1, p=2)
    x, y = z.unbind(dim=-1)
    phi = th.atan2(y, x)
    return {"u": th.cat([rho.unsqueeze(-1), phi.unsqueeze(-1)], -1)}
