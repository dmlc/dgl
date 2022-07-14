.. _guide_ko-nn-construction:

3.1 DGL NN 모듈 생성 함수
---------------------

:ref:`(English Version) <guide-nn-construction>`

생성 함수는 다음 단계들을 수행한다:

1. 옵션 설정
2. 학습할 파라메터 또는 서브모듈 등록
3. 파라메터 리셋

.. code::

    import torch.nn as nn

    from dgl.utils import expand_as_pair

    class SAGEConv(nn.Module):
        def __init__(self,
                     in_feats,
                     out_feats,
                     aggregator_type,
                     bias=True,
                     norm=None,
                     activation=None):
            super(SAGEConv, self).__init__()

            self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
            self._out_feats = out_feats
            self._aggre_type = aggregator_type
            self.norm = norm
            self.activation = activation

생성 함수를 만들 때 데이터 차원을 지정해야 한다. 일반적인 PyTorch 모듈의 경우에는 차원이란 보통은 입력 차원, 출력 차원, 그리고 은닉(hidden) 치원을 의미하는데, 그래프 뉴럴 네트워크의 경우 입력 차원은 소스 노드의 차원과 목적지 노드의 차원으로 나뉜다.

데이터 차원들 이외의 전형적인 그래프 뉴럴 네트워크의 옵션으로 aggregation 타입( ``self._aggre_type`` )이 있다. Aggregation 타입은 특정 목적지 노드에 대해서 관련된 여러 에지의 메시지들이 어떻게 집합되어야 하는지를 결정한다. 흔히 사용되는 aggregation 타입으로는 ``mean`` , ``sum`` , ``max`` , ``min`` 이 있으며, 어떤 모듈은 ``lstm`` 과 같이 좀더 복잡한 aggregation을 적용하기도 한다.

여기서 ``norm`` 은 피처 normalization을 위해서 호출될 수 있는 함수이다. SAGEConv 페이퍼에서는 l2 normlization, :math:`h_v = h_v / \lVert h_v \rVert_2` 이 normalization으로 사용되고 있다.

.. code::

            # aggregator type: mean, pool, lstm, gcn
            if aggregator_type not in ['mean', 'pool', 'lstm', 'gcn']:
                raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
            if aggregator_type == 'pool':
                self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
            if aggregator_type == 'lstm':
                self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
            if aggregator_type in ['mean', 'pool', 'lstm']:
                self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
            self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
            self.reset_parameters()

다음으로는 파라메터들과 서브모듈들을 등록한다. SAGEConv의 경우에는 서브모듈은 aggregation 타입에 따라 달라진다. 그 모듈들은 ``nn.Linear`` , ``nn.LSTM`` 등과 같은 순수한 PyTorch nn 모듈이다. 생성 함수의 마지막에는 ``reset_parameters()`` 호출로 가중치들을 초기화한다.

.. code::

        def reset_parameters(self):
            """Reinitialize learnable parameters."""
            gain = nn.init.calculate_gain('relu')
            if self._aggre_type == 'pool':
                nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
            if self._aggre_type == 'lstm':
                self.lstm.reset_parameters()
            if self._aggre_type != 'gcn':
                nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
