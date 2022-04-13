.. _guide_ko-minibatch-gpu-sampling:

6.7 이웃 샘플링에 GPU 사용하기
------------------------

:ref:`(English Version) <guide-minibatch-gpu-sampling>`

DGL 0.7부터 GPU 기반의 이웃 샘플링을 지원하는데, 이는 CPU 기반의 이웃 샘플링에 비해서 상당한 속도 향상을 가져다 준다. 만약 다루는 그래프와 피쳐들이 GPU에 들어갈 수 있는 크기이고, 모델이 너무 많은 GPU 메모리를 차지하지 않는다면, GPU 메모리에 올려서 GPU 기반의 이웃 샘플링을 하는 것이 최선의 방법이다.

예를 들어, `OGB Products <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`__ 는 2.4M 노드들과 61M 에지들을 갖고, 각 노드는 100 차원의 피쳐를 갖는다. 노트 피쳐들을 모두 합해서 1GB 미만의 메모리를 차지하고, 그래프는 약 1GB 보다 적은 메모리를 사용한다. 그래프의 메모리 요구량은 에지의 개수에 관련이 있다. 따라서, 전체 그래프를 GPU에 로딩하는 것이 가능하다.

.. note::

   이 기능은 실험적인 것으로 개발이 진행 중이다. 추가 업데이트를 지켜보자.

DGL 데이터 로더에서 GPU 기반의 이웃 샘플링 사용하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DGL 데이터 로더에서 GPU 기반의 이웃 샘플링은 다음 방법으로 사용할 수 있다.

* 그래프를 GPU에 넣기
* ``num_workers`` 인자를 0으로 설정하기. CUDA는 같은 context를 사용하는 멀티 프로세스를 지원하지 않기 때문이다.
* ``device`` 인자를 GPU 디바이스로 설정하기

:class:`~dgl.dataloading.pytorch.NodeDataLoader` 의 다른 모든 인자들은 다른 가이드와 튜토리얼에서 사용한 것돠 같다.

.. code:: python

   g = g.to('cuda:0')
   dataloader = dgl.dataloading.NodeDataLoader(
       g,                                # The graph must be on GPU.
       train_nid,
       sampler,
       device=torch.device('cuda:0'),    # The device argument must be GPU.
       num_workers=0,                    # Number of workers must be 0.
       batch_size=1000,
       drop_last=False,
       shuffle=True)

GPU 기반의 이웃 샘플링은 커스텀 이웃 샘플러가 두가지 조건을 충족하면 동작한다. (1) 커스텀 샘플러가 :class:`~dgl.dataloading.BlockSampler` 의 서브 클래스이고, (2) 샘플러가 GPU에서 완전하게 동작한다.

.. note::

   현재는 :class:`~dgl.dataloading.pytorch.EdgeDataLoader` 와 heterogeneous 그래프는 지원하지 않는다.

GPU 기반의 이웃 샘플러를 DGL 함수와 함께 사용하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

다음 함수들은 GPU에서 작동을 지원한다.

* :func:`dgl.sampling.sample_neighbors`

  * 균일 샘플링(uniform sampling)만 지원함. non-uniform샘플링은 CPU에서만 동작함.

위 함수들 이외의 GPU에서 동작하는 함수들은 :func:`dgl.to_block` 를 참고하자.