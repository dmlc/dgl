.. _guide_ko-minibatch-custom-gnn-module:

6.5 미니-배치 학습을 위한 커스텀 GNN 모듈 구현하기
----------------------------------------

:ref:`(English Version) <guide-minibatch-custom-gnn-module>`

Homogeneous 그래프나 heterogeneous 그래프를 대상으로 전체 그래프를 업데이트하는 커스텀 GNN 모듈을 만드는 것에 익숙하다면, MFG에 대한 연산을 구현하는 코드도 비슷하다는 것을 알 수 있다. 차이점은 노드들이 입력 노드와 출력 노드로 나뉜다는 것 뿐이다.

커스텀 graph convolution 모듈을 예로 들자. 이 코드는 단지 커스텀 GNN 모듈이 어떻게 동작하는지 보여주기 위함이지, 가장 효율적인 구현이 아님을 주의하자. 

.. code:: python

    class CustomGraphConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        def forward(self, g, h):
            with g.local_scope():
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))

전체 그래프에 대한 커스텀 메시지 전달 NN 모듈이 있고, 이를 MFG에서 작동하도록 만들고 싶다면, 다음과 같이 forward 함수를 다시 작성하는 것만이 필요하다. 전체 그래프에 대한 구현은 주석 처리를 했으니, 새로운 코드들과 비교해 보자.

.. code:: python

    class CustomGraphConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        # h is now a pair of feature tensors for input and output nodes, instead of
        # a single feature tensor.
        # def forward(self, g, h):
        def forward(self, block, h):
            # with g.local_scope():
            with block.local_scope():
                # g.ndata['h'] = h
                h_src = h
                h_dst = h[:block.number_of_dst_nodes()]
                block.srcdata['h'] = h_src
                block.dstdata['h'] = h_dst
    
                # g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
    
                # return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))
                return self.W(torch.cat(
                    [block.dstdata['h'], block.dstdata['h_neigh']], 1))

일반적으로, 직접 구현한 NN 모듈이 MFG에서 동작하게 만들기 위해서는 다음과 같은 것을 해야한다.

- 첫 몇 행들(row)을 잘라서 입력 피쳐들로부터 출력 노드의 피처를 얻는다. 행의 개수는 :meth:`block.number_of_dst_nodes <dgl.DGLGraph.number_of_dst_nodes>` 로 얻는다.
- 원본 그래프가 한 하나의 노드 타입을 갖는 경우, :attr:`g.ndata <dgl.DGLGraph.ndata>` 를 입력 노드의 피쳐의 경우 :attr:`block.srcdata <dgl.DGLGraph.srcdata>` 로 또는 출력 노드의 피쳐의 경우 :attr:`block.dstdata <dgl.DGLGraph.dstdata>` 로 교체한다.
- 원본 그래프가 여러 종류의 노드 타입을 갖는 경우, :attr:`g.nodes <dgl.DGLGraph.nodes>` 를 입력 노드의 피쳐의 경우 :attr:`block.srcnodes <dgl.DGLGraph.srcnodes>` 로 또는 출력 노드의 피처의 경우 :attr:`block.dstnodes <dgl.DGLGraph.dstnodes>` 로 교체한다.
- :meth:`g.num_nodes <dgl.DGLGraph.num_nodes>` 를 입력 노드의 개수는 :meth:`block.number_of_src_nodes <dgl.DGLGraph.number_of_src_nodes>` 로 출력 노드의 개수는 :meth:`block.number_of_dst_nodes <dgl.DGLGraph.number_of_dst_nodes>` 로 각각 교체한다.

Heterogeneous 그래프들
~~~~~~~~~~~~~~~~~~~~

Heterogeneous 그래프의 경우도 커스텀 GNN 모듈을 만드는 것은 비슷하다. 예를 들어, 전체 그래프에 적용되는 다음 모듈을 예로 들어보자.

.. code:: python

    class CustomHeteroGraphConv(nn.Module):
        def __init__(self, g, in_feats, out_feats):
            super().__init__()
            self.Ws = nn.ModuleDict()
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
            for ntype in g.ntypes:
                self.Vs[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])
    
        def forward(self, g, h):
            with g.local_scope():
                for ntype in g.ntypes:
                    g.nodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                    g.nodes[ntype].data['h_src'] = h[ntype]
                for etype in g.canonical_etypes:
                    utype, _, vtype = etype
                    g.update_all(
                        fn.copy_u('h_src', 'm'), fn.mean('m', 'h_neigh'),
                        etype=etype)
                    g.nodes[vtype].data['h_dst'] = g.nodes[vtype].data['h_dst'] + \
                        self.Ws[etype](g.nodes[vtype].data['h_neigh'])
                return {ntype: g.nodes[ntype].data['h_dst'] for ntype in g.ntypes}

``CustomHeteroGraphConv`` 에서의 원칙은 ``g.nodes`` 를 대상 피쳐가 입력 노드의 것인지 출력 노드의 것인지에 따라서 ``g.srcnodes`` 또는 ``g.dstnodes`` 바꾸는 것이다.

.. code:: python

    class CustomHeteroGraphConv(nn.Module):
        def __init__(self, g, in_feats, out_feats):
            super().__init__()
            self.Ws = nn.ModuleDict()
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
            for ntype in g.ntypes:
                self.Vs[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])
    
        def forward(self, g, h):
            with g.local_scope():
                for ntype in g.ntypes:
                    h_src, h_dst = h[ntype]
                    g.dstnodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                    g.srcnodes[ntype].data['h_src'] = h[ntype]
                for etype in g.canonical_etypes:
                    utype, _, vtype = etype
                    g.update_all(
                        fn.copy_u('h_src', 'm'), fn.mean('m', 'h_neigh'),
                        etype=etype)
                    g.dstnodes[vtype].data['h_dst'] = \
                        g.dstnodes[vtype].data['h_dst'] + \
                        self.Ws[etype](g.dstnodes[vtype].data['h_neigh'])
                return {ntype: g.dstnodes[ntype].data['h_dst']
                        for ntype in g.ntypes}

Homogeneous 그래프, 이분 그래프(bipartite graph), 그리고 MFG를 위한 모듈 작성하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL의 모든 메시지 전달 모듈들은 homogeneous 그래프, 단방향 이분 그래프 (unidirectional bipartite graphs, 두개 노드 타입을 갖고, 하나의 에지 타입을 갖음), 그리고 하나의 에지 타입을 갖는 MFG에서 동작한다. 기본적으로 DGL 빌트인 뉴럴 네트워크 모듈의 입력 그래프와 피쳐는 아래 경우들 중에 하나를 만족해야 한다.

- 입력 피쳐가 텐서들의 쌍인 경우, 입력 그래프는 단방향 이분(unidirectional bipartite) 그래프이어야 한다.
- 입력 피쳐가 단일 텐서이고 입력 그래프가 MFG인 경우, DGL은 자동으로 출력 노드의 피쳐를 입력 노드 피처의 첫 몇개의 행으로 정의한다.
- 입력 피쳐가 단일 텐서이고 입력 그래프가 MGF가 아닌 경우, 입력 그래프는 반드시 homogeneous여야 한다.

다음 코드는 :class:`dgl.nn.pytorch.SAGEConv` 을 PyTorch로 단순하게 구현한 것이다. (MXNet이나 TensorFlow 버전도 제공함. (이 코드는 normalization이 제거되어 있고, mean aggregation만 사용한다.)

.. code:: python

    import dgl.function as fn
    class SAGEConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        def forward(self, g, h):
            if isinstance(h, tuple):
                h_src, h_dst = h
            elif g.is_block:
                h_src = h
                h_dst = h[:g.number_of_dst_nodes()]
            else:
                h_src = h_dst = h
                 
            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
            return F.relu(
                self.W(torch.cat([g.dstdata['h'], g.dstdata['h_neigh']], 1)))

:ref:`guide_ko-nn` 은 단방향 이분 그래프, homogeneous 그래프와 MFG에 적용되는 :class:`dgl.nn.pytorch.SAGEConv` 를 자세히 다루고 있다.


