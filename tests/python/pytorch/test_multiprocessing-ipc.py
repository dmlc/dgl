import os
import unittest

import dgl

import torch as th
import torch.multiprocessing as mp


def sub_ipc(g):
    print(g)
    return g


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_torch_ipc():
    g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=sub_ipc, args=(g,))

    p.start()
    p.join()


if __name__ == "__main__":
    test_torch_ipc()
