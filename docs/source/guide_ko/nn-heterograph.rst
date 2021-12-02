.. _guide_ko-nn-heterograph:

3.3 Heterogeneous GraphConv 모듈
-------------------------------

:ref:`(English Version) <guide-nn-heterograph>`

:class:`~dgl.nn.pytorch.HeteroGraphConv` 는 heterogeneous 그래프들에 DGL NN 모듈을 적용하기 위한 모듈 수준의 인캡슐레이션이다. 메시지 전달 API :meth:`~dgl.DGLGraph.multi_update_all` 와 같은 로직으로 구현되어 있고, 이는 다음을 포함한다.

- :math:`r` 관계에 대한 DGL NN 모듈
- 한 노드에 연결된 여러 관계로부터 얻은 결과를 통합하는 축약(reduction)

이는 다음과 같이 공식으로 표현된다:

.. math::  h_{dst}^{(l+1)} = \underset{r\in\mathcal{R}, r_{dst}=dst}{AGG} (f_r(g_r, h_{r_{src}}^l, h_{r_{dst}}^l))

, 여기서 :math:`f_r` 는 각 :math:`r` 관계에 대한 NN 모듈이고, :math:`AGG` 는 aggregation 함수이다.

HeteroGraphConv 구현 로직:
~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    import torch.nn as nn

    class HeteroGraphConv(nn.Module):
        def __init__(self, mods, aggregate='sum'):
            super(HeteroGraphConv, self).__init__()
            self.mods = nn.ModuleDict(mods)
            if isinstance(aggregate, str):
                # An internal function to get common aggregation functions
                self.agg_fn = get_aggregate_fn(aggregate)
            else:
                self.agg_fn = aggregate

Heterograph convolution은 각 관계를 NN 모듈에 매핑하는 ``mods`` 사전을 인자로 받고, 한 노드에 대한 여러 관계들의 결과를 집계하는 함수를 설정한다.

.. code::

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}

입력 그래프와 입력 텐서들과 더불어, ``forward()`` 함수는 두가지 추가적인 파라메터들, ``mod_args`` 와 ``mod_kwargs`` 을 받는다. 이것들은 ``self.mods`` 안에서, 다른 종류의 관계에 연관된 NN 모듈을 수행할 때, 커스터마이즈된 파라메터들로써 사용된다.

각 목적지 타입 ``nty`` 에 대한 결과 텐서를 저장하기 위해서 결과 사전(output dictionary)가 생성된다. 각 ``nty`` 에 대한 값은 리스트이다. 이는 ``nty`` 를 목적 타입으로 갖을 관계가 여러개가 있는 경우, 단일 노드 타입이 여러 아웃풋들을 갖을 수 있음을 의미한다. ``HeteroGraphConv`` 는 이 리스트들에 대해서 추가적인 aggregation을 수행할 것이다.

.. code::

          if g.is_block:
              src_inputs = inputs
              dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
          else:
              src_inputs = dst_inputs = inputs

          for stype, etype, dtype in g.canonical_etypes:
              rel_graph = g[stype, etype, dtype]
              if rel_graph.num_edges() == 0:
                  continue
              if stype not in src_inputs or dtype not in dst_inputs:
                  continue
              dstdata = self.mods[etype](
                  rel_graph,
                  (src_inputs[stype], dst_inputs[dtype]),
                  *mod_args.get(etype, ()),
                  **mod_kwargs.get(etype, {}))
              outputs[dtype].append(dstdata)

입력 그래프 ``g`` 는 heterogeneous 그래프 또는 heterogeneous 그래프의 서브그래프 블록일 수 있다. 보통의 NN 모듈처럼, ``forward()`` 함수는 다양한 입력 그래프 타입들을 별로도 다룰 수 있어야 한다.

각 관계는 ``(stype, etype, dtype)`` 인 ``canonical_etype`` 으로 표현된다. ``canonical_etype`` 을 키로 사용해서, 이분 그래프(bipartite graph)인 ``rel_graph`` 를 추출할 수 있다. 이분 그래프에서 입력 피쳐는 ``(src_inputs[stype], dst_inputs[dtype])`` 로 구성된다. 각 관계에 대한 NN 모듈이 호출되고, 결과는 저장된다. 

.. code::

        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)

마지막으로 한 목적 노드 타입에 대해 여러 관계로 부터 얻어진 결과들은 ``self.agg_fn`` 를 통해서 집계된다. :class:`~dgl.nn.pytorch.HeteroGraphConv` 의 API DOC에서 관련 예제들이 있다.
