.. _guide_ko-distributed-apis:

7.2 분산 APIs
--------------------

:ref:`(English Version) <guide-distributed-apis>`

이 절은 학습 스크립트에 사용할 분산 API들을 다룬다. DGL은 초기화, 분산 샘플링, 그리고 워크로드 분할(split)을 위한 세가지 분산 데이터 구조와 다양한 API들을 제공한다. 분산 학습/추론에 사용되는 세가지 분산 자료 구조는 분산 그래프를 위한 :class:`~dgl.distributed.DistGraph` , 분산 텐서를 위한 :class:`~dgl.distributed.DistTensor` , 그리고 분산 learnable 임베딩을 위한 :class:`~dgl.distributed.DistEmbedding` 이다.

DGL 분산 모듈 초기화
~~~~~~~~~~~~~~~~

:func:`~dgl.distributed.initialize` 은 분산 모듈을 초기화한다. 학습 스크립트가 학습 모드로 수행되면, 이 API는 DGL 서버들간의 연결을 만들고, 샘플러 프로세스들을 생성한다; 스크립트가 서버 모드로 실행되면, 이 API는 서버 코드를 실행하고 절대로 리턴되지 않는다. 이 API는 어떤 DGL 분산 API들 보다 먼저 호출되어야 한다. PyTorch와 함께 사용될 때, :func:`~dgl.distributed.initialize` 는 ``torch.distributed.init_process_group`` 전에 호출되어야 한다. 일반적으로 초기화 API들은 다음 순서로 실행된다.

.. code:: python

    dgl.distributed.initialize('ip_config.txt')
    th.distributed.init_process_group(backend='gloo')

Distributed 그래프
~~~~~~~~~~~~~~~~~

:class:`~dgl.distributed.DistGraph` 는 클러스터에서 그래프 구조와 노드/에지 피쳐들을 접근하기 위한 Python 클래스이다. 각 컴퓨터는 단 하나의 파티션을 담당한다. 이 클래스는 파티션 데이터(그 파티션의 그래프 구조, 노드 데이터와 에지 데이터)를 로드하고, 클러스터의 모든 트레이너들이 접근할 수 있도록 만들어 준다. :class:`~dgl.distributed.DistGraph` 는 데이터 접근을 위한 :class:`~dgl.DGLGraph` API들의 작은 서브셋을 지원한다.

**Note**: :class:`~dgl.distributed.DistGraph` 는 현재 한 개의 노드 타입과 한 개의 에지 타입만을 지원한다.

분산 모드 vs. 단독(standalone) 모드
^^^^^^^^^^^^^^^^^^

:class:`~dgl.distributed.DistGraph` 는 두가지 모드로 실행된다: 분산 모드와 단독 모드. 사용자가 학습 스크립트를 Python 명령행이나 Jupyter notebook에서 실행하면, 단독 모드로 수행된다. 즉, 모든 계산이 단일 프로세스에서 수행되고, 다른 어떤 프로세스들과의 통신이 없다. 따라서, 단독 모드에서는 입력 그래프가 한 개의 파티션이다. 이 모드는 주로 개발 및 테스트를 위해서 사용된다 (즉, Jupyter notebook에서 코드를 개발하고 수행할 때). 학습 스크립트가 launch 스크립트를 사용해서 실행되면 (launch 스크립트 섹션 참조), :class:`~dgl.distributed.DistGraph` 가 분산 모드로 동작한다. Launch 툴은 자동으로 (노드/에지 피쳐 접근 및 그래프 샘플링을 하는) 서버들을 구동하고, 클러스터의 각 컴퓨터에 파티션 데이터를 자동으로 로드한다. :class:`~dgl.distributed.DistGraph` 는 클러스터의 서버들과 네트워크를 통해서 연결한다.

DistGraph 생성
^^^^^^^^^^^^^

분산 모드에서는, :class:`~dgl.distributed.DistGraph` 를 생성할 때 파티션에서 사용된 그래프 이름이 필요하다. 그래프 이름은 클러스터에서 로드될 그래프를 지정한다.

.. code:: python

    import dgl
    g = dgl.distributed.DistGraph('graph_name')

단독 모드로 수행될 때, 로컬 머신의 그래프 데이터를 로드한다. 따라서, 사용자는 입력 그래프에 대한 모든 정보를 담고 있는 파티션 설정 파일을 제공해야 한다.

.. code:: python

    import dgl
    g = dgl.distributed.DistGraph('graph_name', part_config='data/graph_name.json')

**Note**: DGL의 현재 구현은 `DistGraph` 객체를 한 개만 만들 수 있다. `DistGraph` 를 없애고 새로운 것을 다시 만드는 것은 정의되어 있지 않다.

그래프 구조 접근
^^^^^^^^^^^^

:class:`~dgl.distributed.DistGraph` 는 그래프 구조 접근을 위한 적은 수의 API들을 갖고 있다. 현재 대부분 API들은 노드 및 에지 수와 같은 그래프 정보를 제공한다. DistGraph의 주요 사용 케이스는 미니-배치 학습을 지원하기 위한 샘플링 API를 수행하는 것이다. (분산 그래프 샘플링은 섹션 참조)

.. code:: python

    print(g.num_nodes())

노드/에지 데이터 접근
^^^^^^^^^^^^^^^^

:class:`~dgl.DGLGraph` 처럼 :class:`~dgl.distributed.DistGraph` 는 노드와 에지의 데이터 접근을 위해서 ``ndata`` 와 ``edata`` 를 제공한다. 차이점은 :class:`~dgl.distributed.DistGraph` 의 ``ndata`` / ``edata`` 는 사용되는 프레임워크의 텐서 대신 :class:`~dgl.distributed.DistTensor` 를 리턴한다는 것이다. 사용자는 새로운 :class:`~dgl.distributed.DistTensor` 를 :class:`~dgl.distributed.DistGraph` 노드 데이터 또는 에지 데이터로서 할당할 수 있다.

.. code:: python

    g.ndata['train_mask']  # <dgl.distributed.dist_graph.DistTensor at 0x7fec820937b8>
    g.ndata['train_mask'][0]  # tensor([1], dtype=torch.uint8)

분산 텐서(Distributed Tensor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

앞에서 언급했듯이, DGL은 노드/에치 피쳐들을 샤드(shard)해서, 머신들의 클러스터에 이것들을 저장한다. DGL은 클러스터에서 파티션된 노드/에지 피쳐들을 접근하기 위해서 tensor-like 인터패이스를 갖는 분산 텐서를 제공한다. 분산 세팅에서 DGL은 덴스 노드/에지 피쳐들만 지원한다.

:class:`~dgl.distributed.DistTensor` 는 파티션되어 여러 머신들에 저장되어 있는 덴스 텐서들을 관리한다. 지금은 부산 텐서는 그래프의 노드 또는 에지와 연결되어 있어야만 한다. 다르게 말하자면, `DistTensor` 의 행 개수는 그래프의 노드 개수 또는 에지의 개수과 같아야만 한다. 아래 코드는 분산 텐서를 생성하고 있다. `shape` 과 `dtype` 뿐만아니라, 유일한 텐서 이름을 지정할 수 있다. 사용자가 영속적인 분산 텐서를 참고하고자 할 경우 이 이름은 유용하다 (즉, :class:`~dgl.distributed.DistTensor` 객체가 사라져도 클러스터에 존재하는 텐서).

.. code:: python

    tensor = dgl.distributed.DistTensor((g.num_nodes(), 10), th.float32, name='test')

**Note**: :class:`~dgl.distributed.DistTensor` 생성은 동기화 수행이다. 모든 트레이너들은 생성을 실행해야하고, 모든 트레이너가 이를 호출한 경우에만 생성이 완료된다.

사용자는 :class:`~dgl.distributed.DistTensor` 를 노드 데이터 또는 에지 데이터의 하나로서 :class:`~dgl.distributed.DistGraph`  객체에 추가할 수 있다.

.. code:: python

    g.ndata['feat'] = tensor

**Note**: 노드 데이터 이름과 텐서 이름이 같을 필요는 없다. 전자는 :class:`~dgl.distributed.DistGraph` 로부터 노드 데이터를 구별하고(트레이너 프로세스에서), 후자는 DGL 서버들에서 분산 텐서를 구별하는데 사용된다. 

:class:`~dgl.distributed.DistTensor` 는 적은 수의 함수들을 제공한다. 이는 일반 텐서가 `shape` 또는 `dtype` 과 같은 메타데이터를 접근하는 것과 같은 API들이다. :class:`~dgl.distributed.DistTensor` 는 인덱스를 사용한 읽기와 쓰기를 지원하지만, `sum` 또는 `mean` 과 같은 연산 오퍼레이터는 지원하지 않는다.

.. code:: python

    data = g.ndata['feat'][[1, 2, 3]]
    print(data)
    g.ndata['feat'][[3, 4, 5]] = data

**Note**: 현재 DGL은 한 머신이 여러 서버들을 수행할 때, 다중의 서버들이 동시에 쓰기를 동시에 수행하는 경우에 대한 보호를 지원하지 않는다. 이 경우 데이터 깨짐(data corruption)이 발생할 수 있다. 같은 행의 데이터에 동시 쓰기를 방지하는 방법 중에 하나로 한 머신에서 한 개의 서버 프로세스만 실행하는 것이다.

분산 DistEmbedding
~~~~~~~~~~~~~~~~~

DGL은 노드 임베딩들을 필요로 하는 변환 모델(transductive models)을 지원하기 위해서 :class:`~dgl.distributed.DistEmbedding` 를 제공한다. 분산 임베딩을 생성하는 것은 분산 텐서를 생성하는 것과 비슷하다.

.. code:: python

    def initializer(shape, dtype):
        arr = th.zeros(shape, dtype=dtype)
        arr.uniform_(-1, 1)
        return arr
    emb = dgl.distributed.DistEmbedding(g.num_nodes(), 10, init_func=initializer)

내부적으로는 분산 임배딩은 분산 텐서를 사용해서 만들어진다. 따라서, 분산 텐서와 비슷하게 동작한다. 예를 들어, 임베딩이 만들어지면, 그것들은 클러스터의 여러 머신들에 나눠져서(shard) 저장된다. 이는 이름을 통해서 고유하게 식별될 수 있다.

**Note**: 초기화 함수가 서버 프로세스에서 호출된다. 따라서, :class:`~dgl.distributed.initialize` 전에 선언되야 한다.

임배딩은 모델의 일부이기 때문에, 미니배치 학습을 위해서 이를 optimizer에 붙여줘야 한다. 현재는, DGL은 sparse Adagrad optimizer, :class:`~dgl.distributed.SparseAdagrad` 를 지원한다 (DGL은 sparse 임베딩을 위핸 더 많은 optimizer들을 추가할 예정이다). 사용자는 모델로 부터 모든 분산 임베딩을 수집하고, 이를 sparse optimizer에 전달해야 한다. 만약 모델이 노드 임베딩과 정상적인 dense 모델 파라메터들을 갖고, 사용자가 임베딩들에 sparse 업데이트를 수행하고 싶은 경우, optimizer 두 개를 만들어야 한다. 하나는 노드 임베딩을 위한 것이고, 다른 하나는 dense model 파라메터들을 위한 것이다. 다음 코드를 보자.

.. code:: python

    sparse_optimizer = dgl.distributed.SparseAdagrad([emb], lr=lr1)
    optimizer = th.optim.Adam(model.parameters(), lr=lr2)
    feats = emb(nids)
    loss = model(feats)
    loss.backward()
    optimizer.step()
    sparse_optimizer.step()

**Note**: :class:`~dgl.distributed.DistEmbedding` 는 PyTorch nn 모듈이 아니다. 따라서, PyTorch nn 모듈의 파라메터들을 통해서 접근할 수 없다.

분산 샘플링
~~~~~~~~

DGL은 미니-배치를 생성하기 위해 노드 및 에지 샘플링을 하는 두 수준의 API를 제공한다 (미니-배치 학습 섹션 참조). Low-level API는 노드들의 레이어가 어떻게 샘플링될지를 명시적으로 정의하는 코드를 직접 작성해야한다 (예를 들면, :func:`dgl.sampling.sample_neighbors` 사용해서). High-level API는 노드 분류 및 링크 예측(예, :class:`~dgl.dataloading.pytorch.NodeDataLoader` 와
:class:`~dgl.dataloading.pytorch.EdgeDataLoader`) 에 사용되는 몇 가지 유명한 샘플링 알고리즘을 구현하고 있다.

분산 샘플링 모듈도 같은 디자인을 따르고 있고, 두 level의 샘플링 API를 제공한다. Low-level 샘플링 API의 경우, :class:`~dgl.distributed.DistGraph` 에 대한 분산 이웃 샘플링을 위해 :func:`~dgl.distributed.sample_neighbors` 가 있다. 또한, DGL은 분산 샘플링을 위해 분산 데이터 로더, :class:`~dgl.distributed.DistDataLoader` 를 제공한다. 분산 DataLoader는 PyTorch DataLoader와 같은 인터페이스를 갖는데, 다른 점은 사용자가 데이터 로더를 생성할 때 worker 프로세스의 개수를 지정할 수 없다는 점이다. Worker 프로세스들은 :func:`dgl.distributed.initialize` 에서 만들어진다.

**Note**: :class:`~dgl.distributed.DistGraph` 에 :func:`dgl.distributed.sample_neighbors` 를 실행할 때, 샘플러는 다중의 worker 프로세스를 갖는 PyTorch DataLoader에서 실행될 수 없다. 주요 이유는 PyTorch DataLoader는 매 epoch 마다 새로운 샘플링 worker 프로세스는 생성하는데, 이는 :class:`~dgl.distributed.DistGraph` 객체들을 여러번 생성하고 삭제하게하기 때문이다.

Low-level API를 사용할 때, 샘플링 코드는 단일 프로세스 샘플링과 비슷하다. 유일한 차이점은 사용자가 :func:`dgl.distributed.sample_neighbors` 와 :class:`~dgl.distributed.DistDataLoader` 를 사용한다는 것이다.

.. code:: python

    def sample_blocks(seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in [10, 25]:
            frontier = dgl.distributed.sample_neighbors(g, seeds, fanout, replace=True)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
            return blocks
        dataloader = dgl.distributed.DistDataLoader(dataset=train_nid,
                                                    batch_size=batch_size,
                                                    collate_fn=sample_blocks,
                                                    shuffle=True)
        for batch in dataloader:
            ...

동일한 high-level 샘플링 API들(:class:`~dgl.dataloading.pytorch.NodeDataLoader` 와 :class:`~dgl.dataloading.pytorch.EdgeDataLoader` )이 :class:`~dgl.DGLGraph` 와 :class:`~dgl.distributed.DistGraph` 에 대해서 동작한다. :class:`~dgl.dataloading.pytorch.NodeDataLoader` 과 :class:`~dgl.dataloading.pytorch.EdgeDataLoader` 를 사용할 때, 분산 샘플링 코드는 싱글-프로세스 샘플링 코드와 정확하게 같다.

.. code:: python

    sampler = dgl.sampling.MultiLayerNeighborSampler([10, 25])
    dataloader = dgl.sampling.DistNodeDataLoader(g, train_nid, sampler,
                                                 batch_size=batch_size, shuffle=True)
    for batch in dataloader:
        ...


워크로드 나누기(Split workloads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

모델을 학습하기 위해서, 사용자는 우선 데이터를 학습, 검증 그리고 테스트 셋으로 나눠야한다. 분산 학습에서는, 이 단계가 보통은 그래프를 파터션하기 위해 :func:`dgl.distributed.partition_graph` 를 호출하기 전에 일어난다. 우리는 데이터 split를 노드 데이 또는 에지 데이터로서 boolean array들에 저장하는 것을 권장한다. 노드 분류 테스크의 경우에 이 boolean array들의 길이는 그래프의 노드의 개수와 같고, 각 원소들은 노드가 학습/검증/테스트 셋에 속하는지를 지정한다. 링크 예측 테스크에도 비슷한 boolean array들을 사용해야 한다. :func:`dgl.distributed.partition_graph` 는 그래프 파티션 결과에 따라서 이 boolean array들을 나누고, 이를 그래프 파타션과 함께 저장한다.

분산 학습을 수행하는 동안에 사용자는 학습 노드들/에지들을 각 트레이너에게 할당해야 한다. 비슷하게, 검증 및 테스트 셋도 같은 방법으로 나눠야만 한다. DGL은 분산학습이 수행될 때 학습, 검증, 테스트 셋을 나누는 :func:`~dgl.distributed.node_split` 와 :func:`~dgl.distributed.edge_split` 를 제공한다. 이 두 함수는 그래프 파티셔닝 전에 생성된 boolean array들을 입력으로 받고, 그것들을 나누고 나눠진 부분을 로컬 트레이너에게 리턴한다. 기본 설정으로는 모든 부분들이 같은 개수의 노드와 에지를 갖도록 해준다. 이는 각 트레이너가 같은 크기의 미니-배치들을 갖는다고 가정하는 synchronous SDG에서 중요하다.

아래 예제는 학습 셋을 나누고, 노들의 서브셋을 로컬 프로세스에 리턴한다.

.. code:: python

    train_nids = dgl.distributed.node_split(g.ndata['train_mask'])

