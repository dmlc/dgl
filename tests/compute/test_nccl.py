from dgl.cuda import nccl
import unittest
import backend as F


def gen_test_id():
    return '{:0256x}'.format(78236728318467363)

@unittest.skipIf(F._default_context_str == 'cpu', reason="NCCL only runs on GPU.")
def test_nccl_id():
    nccl_id = nccl.UniqueId()

    text = str(nccl_id)

    nccl_id2 = nccl.UniqueId(id_str=text)

    assert nccl_id == nccl_id2

    nccl_id2 = nccl.UniqueId(gen_test_id())

    assert nccl_id2 != nccl_id

    nccl_id3 = nccl.UniqueId(str(nccl_id2))

    assert nccl_id2 == nccl_id3


if __name__ == '__main__':
    test_nccl_id()
