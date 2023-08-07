import io
import pickle

import dgl

import networkx as nx
import torch


def _reconstruct_pickle(obj):
    f = io.BytesIO()
    pickle.dump(obj, f)
    f.seek(0)
    obj = pickle.load(f)
    f.close()
    return obj


def test_pickling_batched_graph():
    # NOTE: this is a test for a wierd bug mentioned in
    #   https://github.com/dmlc/dgl/issues/438
    glist = [nx.path_graph(i + 5) for i in range(5)]
    glist = [dgl.from_networkx(g) for g in glist]
    bg = dgl.batch(glist)
    bg.ndata["x"] = torch.randn((35, 5))
    bg.edata["y"] = torch.randn((60, 3))
    new_bg = _reconstruct_pickle(bg)


if __name__ == "__main__":
    test_pickling_batched_graph()
