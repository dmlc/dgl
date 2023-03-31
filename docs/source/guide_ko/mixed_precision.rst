.. _guide_ko-mixed_precision:

8장: Mixed Precision 학습
=======================

:ref:`(English Version) <guide-mixed_precision>`

DGL은 mixed precision 학습을 위해서 `PyTorch's automatic mixed precision package <https://pytorch.org/docs/stable/amp.html>`_ 와 호환된다. 따라서, 학습 시간 및 GPU 메모리 사용량을 절약할 수 있다. 

Half precision을 사용한 메시지 전달
------------------------------

fp16을 지원하는 DGL은 UDF(User Defined Function)이나 빌트인 함수(예, ``dgl.function.sum``,
``dgl.function.copy_u``)를 사용해서 ``float16`` 피쳐에 대한 메시지 전달을 허용한다.


다음 예제는 DGL 메시지 전달 API를 half-precision 피쳐들에 사용하는 방법을 보여준다.

    >>> import torch
    >>> import dgl
    >>> import dgl.function as fn
    >>> g = dgl.rand_graph(30, 100).to(0)  # Create a graph on GPU w/ 30 nodes and 100 edges.
    >>> g.ndata['h'] = torch.rand(30, 16).to(0).half()  # Create fp16 node features.
    >>> g.edata['w'] = torch.rand(100, 1).to(0).half()  # Create fp16 edge features.
    >>> # Use DGL's built-in functions for message passing on fp16 features.
    >>> g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'x'))
    >>> g.ndata['x'][0]
    tensor([0.3391, 0.2208, 0.7163, 0.6655, 0.7031, 0.5854, 0.9404, 0.7720, 0.6562,
            0.4028, 0.6943, 0.5908, 0.9307, 0.5962, 0.7827, 0.5034],
           device='cuda:0', dtype=torch.float16)
    >>> g.apply_edges(fn.u_dot_v('h', 'x', 'hx'))
    >>> g.edata['hx'][0]
    tensor([5.4570], device='cuda:0', dtype=torch.float16)
    >>> # Use UDF(User Defined Functions) for message passing on fp16 features.
    >>> def message(edges):
    ...     return {'m': edges.src['h'] * edges.data['w']}
    ...
    >>> def reduce(nodes):
    ...     return {'y': torch.sum(nodes.mailbox['m'], 1)}
    ...
    >>> def dot(edges):
    ...     return {'hy': (edges.src['h'] * edges.dst['y']).sum(-1, keepdims=True)}
    ...
    >>> g.update_all(message, reduce)
    >>> g.ndata['y'][0]
    tensor([0.3394, 0.2209, 0.7168, 0.6655, 0.7026, 0.5854, 0.9404, 0.7720, 0.6562,
            0.4028, 0.6943, 0.5908, 0.9307, 0.5967, 0.7827, 0.5039],
           device='cuda:0', dtype=torch.float16)
    >>> g.apply_edges(dot)
    >>> g.edata['hy'][0]
    tensor([5.4609], device='cuda:0', dtype=torch.float16)


End-to-End Mixed Precision 학습
------------------------------

DGL은 PyTorch의 AMP package를 사용해서 mixed precision 학습을 구현하고 있어서, 사용 방법은 `PyTorch의 것 <https://pytorch.org/docs/stable/notes/amp_examples.html>`_ 과 동일하다.

GNN 모델의 forward 패스(loss 계산 포함)를 ``torch.cuda.amp.autocast()`` 로 래핑하면 PyTorch는 각 op 및 텐서에 대해서 적절한 데이터 타입을 자동으로 선택한다. Half precision 텐서는 메모리 효율적이고, half precision 텐서에 대한 대부분 연산들은 GPU tensorcore들을 활용하기 때문에 더 빠르다.

``float16`` 포멧의 작은 graident들은 언더플로우(underflow) 문제를 갖는데 (0이 되버림), PyTorch는 이를 해결하기 위해서 ``GradScaler`` 모듈을 제공한다. ``GradScaler`` 는 loss 값에 factor를 곱하고, 이 scaled loss에 backward pass를 수행한다. 그리고 파라메터들을 업데이트하는 optimizer를 수행하기 전에 unscale 한다.

다음은 3-레이어 GAT를 Reddit 데이터셋(1140억개의 에지를 갖는)에 학습을 하는 스크립트이다. ``use_fp16`` 가 활성화/비활성화되었을 때의 코드 차이를 살펴보자.

.. code::

    import torch 
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    import dgl
    from dgl.data import RedditDataset
    from dgl.nn import GATConv

    use_fp16 = True


    class GAT(nn.Module):
        def __init__(self,
                     in_feats,
                     n_hidden,
                     n_classes,
                     heads):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GATConv(in_feats, n_hidden, heads[0], activation=F.elu))
            self.layers.append(GATConv(n_hidden * heads[0], n_hidden, heads[1], activation=F.elu))
            self.layers.append(GATConv(n_hidden * heads[1], n_classes, heads[2], activation=F.elu))

        def forward(self, g, h):
            for l, layer in enumerate(self.layers):
                h = layer(g, h)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                else:
                    h = h.mean(1)
            return h

    # Data loading
    data = RedditDataset()
    device = torch.device(0)
    g = data[0]
    g = dgl.add_self_loop(g)
    g = g.int().to(device)
    train_mask = g.ndata['train_mask']
    features = g.ndata['feat']
    labels = g.ndata['label']
    in_feats = features.shape[1]
    n_hidden = 256
    n_classes = data.num_classes
    n_edges = g.num_edges()
    heads = [1, 1, 1]
    model = GAT(in_feats, n_hidden, n_classes, heads)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # Create gradient scaler
    scaler = GradScaler()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # Wrap forward pass with autocast
        with autocast(enabled=use_fp16):
            logits = model(g, features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        if use_fp16:
            # Backprop w/ gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print('Epoch {} | Loss {}'.format(epoch, loss.item()))

NVIDIA V100 (16GB) 한개를 갖는 컴퓨터에서, 이 모델을 fp16을 사용하지 않고 학습할 때는 15.2GB GPU 메모리가 사용되는데, fp16을 활성화하면, 학습에 12.8G GPU 메모리가 사용된며, 두 경우 loss가 비슷한 값으로 수렴한다. 만약 head의 갯수를 ``[2, 2, 2]`` 로 바꾸면, fp16를 사용하지 않는 학습은 GPU OOM(out-of-memory) 이슈가 생길 것이지만, fp16를 사용한 학습은 15.7G GPU 메모리를 사용하면서 수행된다.

DGL은 half-precision 지원을 계속 향상하고 있고, 연산 커널의 성능은 아직 최적은 아니다. 앞으로의 업데이트를 계속 지켜보자.