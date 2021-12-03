.. _guide_ko-graph-gpu:

1.6 GPU에서 DGLGraph 사용하기
--------------------------

:ref:`(English Version)<guide-graph-gpu>`

그래프 생성시, 두 GPU 텐서를 전달해서 GPU에 위치한 :class:`~dgl.DGLGraph` 를 만들 수 있다. 다른 방법으로는 :func:`~dgl.DGLGraph.to` API를 사용해서 :class:`~dgl.DGLGraph` 를 GPU로 복사할 수 있다. 이는 그래프 구조와 피처 데이터를 함께 복사한다.

.. code::

    >>> import dgl
    >>> import torch as th
    >>> u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
    >>> g = dgl.graph((u, v))
    >>> g.ndata['x'] = th.randn(5, 3)  # original feature is on CPU
    >>> g.device
    device(type='cpu')
    >>> cuda_g = g.to('cuda:0')  # accepts any device objects from backend framework
    >>> cuda_g.device
    device(type='cuda', index=0)
    >>> cuda_g.ndata['x'].device       # feature data is copied to GPU too
    device(type='cuda', index=0)

    >>> # A graph constructed from GPU tensors is also on GPU
    >>> u, v = u.to('cuda:0'), v.to('cuda:0')
    >>> g = dgl.graph((u, v))
    >>> g.device
    device(type='cuda', index=0)

GPU 그래프에 대한 모든 연산은 GPU에서 수행된다. 따라서, 모든 텐서 인자들이 GPU에 이미 존재해야하며, 연산 결과(그래프 또는 텐서) 역시 GPU에 저장된다. 더 나아가, GPU 그래프는 GPU에 있는 피쳐 데이터만 받아들인다.

.. code::

    >>> cuda_g.in_degrees()
    tensor([0, 0, 1, 1, 1], device='cuda:0')
    >>> cuda_g.in_edges([2, 3, 4])   # ok for non-tensor type arguments
    (tensor([0, 1, 2], device='cuda:0'), tensor([2, 3, 4], device='cuda:0'))
    >>> cuda_g.in_edges(th.tensor([2, 3, 4]).to('cuda:0'))  # tensor type must be on GPU
    (tensor([0, 1, 2], device='cuda:0'), tensor([2, 3, 4], device='cuda:0'))
    >>> cuda_g.ndata['h'] = th.randn(5, 4)  # ERROR! feature must be on GPU too!
    DGLError: Cannot assign node feature "h" on device cpu to a graph on device
    cuda:0. Call DGLGraph.to() to copy the graph to the same device.
