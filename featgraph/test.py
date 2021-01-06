import torch
from operators import gsddmm


def test_gsddmm():
    print(gsddmm('add', 3, 'int32', 'float32', schedule_type='general'))
    print(gsddmm('dot', 2, 'int32', 'float32', schedule_type='tree'))
    pass

if __name__ == "__main__":
    test_gsddmm()
