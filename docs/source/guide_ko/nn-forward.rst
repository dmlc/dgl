.. _guide_ko-nn-forward:

3.2 DGL NN 모듈의 Forward 함수
---------------------------

:ref:`(English Versin) <guide-nn-forward>`

NN 모듈에서 ``forward()`` 함수는 실제 메시지 전달과 연산을 수행한다. 일반적으로 텐서들을 파라메터로 받는 PyTorch의 NN 모듈과 비교하면, DGL NN 모듈은 :class:`dgl.DGLGraph` 를 추가 파라메터로 받는다. ``forward()`` 함수는 3단계로 수행된다.

- 그래프 체크 및 그래프 타입 명세화
- 메시지 전달
- 피쳐 업데이트

이 절에서는 SAGEConv에서 사용되는 ``forward()`` 함수를 자세하게 살펴보겠다.

그래프 체크와 그래프 타입 명세화(graph type specification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

        def forward(self, graph, feat):
            with graph.local_scope():
                # Specify graph type then expand input feature according to graph type
                feat_src, feat_dst = expand_as_pair(feat, graph)

``forward()`` 는 계산 및 메시지 전달 과정에서 유효하지 않은 값을 만들 수 있는 여러 특별한 케이스들을 다룰 수 있어야 한다. :class:`~dgl.nn.pytorch.conv.GraphConv` 와 같은 그래프 conv 모듈에서 수행하는 가장 전형적인 점검은 입력 그래프가 in-degree가 0인 노드를 갖지 않는지 확인하는 것이다. in-degree가 0인 경우에, ``mailbox`` 에 아무것도 없게 되고, 축약 함수는 모두 0인 값을 만들어낼 것이다. 이는 잠재적인 모델 성능 문제를 일이킬 수도 있다. 하지만, :class:`~dgl.nn.pytorch.conv.SAGEConv` 모듈의 경우, aggregated representation은 원래의 노드 피쳐와 연결(concatenated)되기 때문에, ``forward()`` 의 결과는 항상 0이 아니기 때문에, 이런 체크가 필요 없다.

DGL NN 모듈은 여러 종류의 그래프, 단종 그래프, 이종 그래프(:ref:`guide_ko-graph-heterogeneous`), 서브그래프 블록(:ref:`guide_ko-minibatch` ), 입력에 걸쳐서 재사용될 수 있다. 

SAGEConv의 수학 공식은 다음과 같다:

.. math::

   h_{\mathcal{N}(dst)}^{(l+1)}  = \mathrm{aggregate}
           \left(\{h_{src}^{l}, \forall src \in \mathcal{N}(dst) \}\right)

.. math::

    h_{dst}^{(l+1)} = \sigma \left(W \cdot \mathrm{concat}
           (h_{dst}^{l}, h_{\mathcal{N}(dst)}^{l+1}) + b \right)

.. math::

    h_{dst}^{(l+1)} = \mathrm{norm}(h_{dst}^{l+1})

그래프 타입에 따라서 소스 노드 피쳐(``feat_src``)와 목적지 노드 피쳐(``feat_dst``)를 명시해야 한다. :meth:`~dgl.utils.expand_as_pair` 는 명시된 그래프 타입에 따라 ``feat`` 를 ``feat_src`` 와 ``feat_dst`` 로 확장하는 함수이다. 이 함수의 동작은 다음과 같다.

.. code::

    def expand_as_pair(input_, g=None):
        if isinstance(input_, tuple):
            # Bipartite graph case
            return input_
        elif g is not None and g.is_block:
            # Subgraph block case
            if isinstance(input_, Mapping):
                input_dst = {
                    k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                    for k, v in input_.items()}
            else:
                input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
            return input_, input_dst
        else:
            # Homogeneous graph case
            return input_, input_

homogeneous 그래프 전체를 학습시키는 경우, 소스 노드와 목적지 노드들의 타입이 같다. 이것들은 그래프의 전체 노드들이다.

Heterogeneous 그래프의 경우, 그래프는 여러 이분 그래프로 나뉠 수 있다. 즉, 각 관계당 하나의 그래프로. 관계는 ``(src_type, edge_type, dst_dtype)`` 로 표현된다. 입력 피쳐 ``feat`` 가 tuple 이라고 확인되면, 이 함수는 그 그래프는 이분 그래프로 취급한다. Tuple의 첫번째 요소는 소스 노드 피처이고, 두번째는 목적지 노드의 피처이다.

미니-배치 학습의 경우, 연산이 여러 목적지 노드들을 기반으로 샘플된 서브 그래프에 적용된다. DGL에서 서브 그래프는 ``block`` 이라고 한다. 블록이 생성되는 단계에서, ``dst_nodes`` 가 노드 리스트의 앞에 놓이게 된다. ``[0:g.number_of_dst_nodes()]`` 인덱스를 이용해서 ``feat_dst`` 를 찾아낼 수 있다.

``feat_src`` 와 ``feat_dst`` 가 정해진 후에는, 세가지 그래프 타입들에 대한 연산은 모두 동일하다.

메시지 전달과 축약
~~~~~~~~~~~~~~

.. code::

                import dgl.function as fn
                import torch.nn.functional as F
                from dgl.utils import check_eq_shape

                if self._aggre_type == 'mean':
                    graph.srcdata['h'] = feat_src
                    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                elif self._aggre_type == 'gcn':
                    check_eq_shape(feat)
                    graph.srcdata['h'] = feat_src
                    graph.dstdata['h'] = feat_dst
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                    # divide in_degrees
                    degs = graph.in_degrees().to(feat_dst)
                    h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                elif self._aggre_type == 'pool':
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                    graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                else:
                    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

                # GraphSAGE GCN does not require fc_self.
                if self._aggre_type == 'gcn':
                    rst = self.fc_neigh(h_neigh)
                else:
                    rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

이 코드는 실제로 메시지 전달과 축약 연산을 실행하고 있다. 이 부분의 코드는 모듈에 따라 다르게 구현된다. 이 코드의 모든 메시지 전달은 :meth:`~dgl.DGLGraph.update_all` API와 ``built-in``  메시지/축약 함수들로 구현되어 있는데, 이는 :ref:`guide_ko-message-passing-efficient` 에서 설명된 DGL의 성능 최적화를 모두 활용하기 위해서이다.

출력값을 위한 축약 후 피쳐 업데이트
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

                # activation
                if self.activation is not None:
                    rst = self.activation(rst)
                # normalization
                if self.norm is not None:
                    rst = self.norm(rst)
                return rst

``forward()`` 함수의 마지막 부분은 ``reduce function`` 다음에 피쳐를 업데이트하는 것이다. 일반적인 업데이트 연산들은 활성화 함수를 적용하고, 객체 생성 단계에서 설정된 옵션에 따라 normalization을 수행한다.

