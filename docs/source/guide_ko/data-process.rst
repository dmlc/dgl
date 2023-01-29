.. _guide_ko-data-pipeline-process:

4.3 데이터 프로세싱
---------------

:ref:`(English Version) <guide-data-pipeline-process>`

데이터 프로세싱 코드를 ``process()`` 함수에 구현할 수 있으며, 이때 처리되지 않은 데이터는 ``self.raw_dir`` 디렉토리에 있어야 한다. 그래프 머신러닝에는 일반적으로 3가지 종류의 일이 있다: 그래프 분류, 노드 분류, 그리고 링크 예측. 이 절에서는 이 일들에 관련된 데이터셋 처리 방법을 설명한다.

이 절에서 그래프들, 피쳐들, 그리고 마스크들을 처리하는 표준 방법에 집중해서 알아본다. 빌트인 데이터셋을 예제로 사용할 것이고, 파일로 부터 그래프를 만드는 방법은 생략한다. 하지만, 이와 관련된 구현에 대한 링크를 제공할 것이다. 외부 소스들로 부터 그래프를 만드는 방법에 대한 완벽한 가이드는 :ref:`guide_ko-graph-external` 를 참고하자.

그래프 분류 데이터셋 프로세싱
~~~~~~~~~~~~~~~~~~~~~~

그래프 분류 데이터셋은 미니-배치 학습이 사용되는 전형적인 머신러닝 테스크에서 사용되는 데이터셋과 거의 동일하다. 즉, 처리되지 않은 데이터는 :class:`dgl.DGLGraph` 객체들의 리스트와 레이블 텐서들의 리스트로 변환하면 된다. 또한, 만약 처리되지 않은 데이터가 여러 파일들로 나눠져 있을 경우에는, 데이터의 특정 부분을 로드하기 위해서 ``split``  파라메터를 더할 수 있다.

:class:`~dgl.data.QM7bDataset` 를 예로 살펴보자:

.. code::

    from dgl.data import DGLDataset

    class QM7bDataset(DGLDataset):
        _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/qm7b.mat'
        _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'

        def __init__(self, raw_dir=None, force_reload=False, verbose=False):
            super(QM7bDataset, self).__init__(name='qm7b',
                                              url=self._url,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

        def process(self):
            mat_path = self.raw_path + '.mat'
            # process data to a list of graphs and a list of labels
            self.graphs, self.label = self._load_graph(mat_path)

        def __getitem__(self, idx):
            """ Get graph and label by index

            Parameters
            ----------
            idx : int
                Item index

            Returns
            -------
            (dgl.DGLGraph, Tensor)
            """
            return self.graphs[idx], self.label[idx]

        def __len__(self):
            """Number of graphs in the dataset"""
            return len(self.graphs)

``process()`` 함수에서 처리되지 않은 데이터는 그래프들의 리스트와 레이블들의 리스트로 변환된다. Iteration을 위해서 ``__getitem__(idx)`` 와 ``__len__()`` 를 구현해야 한다. 위의 예제에서와 같이, DGL에서는 ``__getitem__(idx)`` 가 ``(graph, label)`` tuple을 리턴하도록 권장한다. ``self._load_graph()`` 와 ``__getitem__`` 함수의 구체적인 구현은 `QM7bDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/qm7b.html#QM7bDataset>`__ 를 확인하자.

데이터셋의 유용한 정보들을 지정하기 위해서 클래스에 프로퍼티들을 추가하는 것이 가능하다. :class:`~dgl.data.QM7bDataset` 에 이 멀티 테스크 데이터셋의 예측 테스트의 총 개숫를 지정하기 위해 ``num_tasks`` 라는 프로퍼티를 추가할 수 있다.

.. code::

    @property
    def num_tasks(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 14

구현 코드를 마친 후에, :class:`~dgl.data.QM7bDataset` 를 다음과 같이 사용한다.

.. code::

    import dgl
    import torch

    from dgl.dataloading import GraphDataLoader

    # load data
    dataset = QM7bDataset()
    num_tasks = dataset.num_tasks

    # create dataloaders
    dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True)

    # training
    for epoch in range(100):
        for g, labels in dataloader:
            # your training code here
            pass

그래프 분류 모델 학습에 대한 전체 가이드는 :ref:`guide_ko-training-graph-classification` 를 참고하자.

DGL의 빌트인 그래프 분류 데이터셋을 참고하면 그래프 분류 데이터셋의 더 많은 예들을 확인할 수 있다.

* :ref:`gindataset`
* :ref:`minigcdataset`
* :ref:`qm7bdata`
* :ref:`tudata`

노드 분류 데이터셋 프로세싱
~~~~~~~~~~~~~~~~~~~~

그래프 분류와는 다르게 노드 분류는 일번적으로 단일 그래프에서 이뤄진다. 따라서, 데이터셋의 분할(split)은 그래프 노드에서 일어난다. DGL은 노드 마스크를 사용해서 분할을 지정하는 것을 권장한다. 이 절에서는 빌트인 데이터셋 `CitationGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__ 을 예로 들겠다.

추가로, DGL은 노드들와 에지들이 서로 가까운 ID값들이 서로 가까운 범위에 있도록 재배열하는 것을 권장한다. 이 절차는 노드의 neighbor들에 대한 접근성을 향상시켜서, 이 후의 연산 및 그래프에 대한 분석을 빠르게 하기 위함이다. 이를 위해서 DGL은 :func:`dgl.reorder_graph` API를 제공한다. 더 자세한 내용은 다음 예제의 ``process()`` 를 참고하자.

.. code::

    from dgl.data import DGLBuiltinDataset
    from dgl.data.utils import _get_dgl_url

    class CitationGraphDataset(DGLBuiltinDataset):
        _urls = {
            'cora_v2' : 'dataset/cora_v2.zip',
            'citeseer' : 'dataset/citeseer.zip',
            'pubmed' : 'dataset/pubmed.zip',
        }

        def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
            assert name.lower() in ['cora', 'citeseer', 'pubmed']
            if name.lower() == 'cora':
                name = 'cora_v2'
            url = _get_dgl_url(self._urls[name])
            super(CitationGraphDataset, self).__init__(name,
                                                       url=url,
                                                       raw_dir=raw_dir,
                                                       force_reload=force_reload,
                                                       verbose=verbose)

        def process(self):
            # Skip some processing code
            # === data processing skipped ===

            # build graph
            g = dgl.graph(graph)
            # splitting masks
            g.ndata['train_mask'] = train_mask
            g.ndata['val_mask'] = val_mask
            g.ndata['test_mask'] = test_mask
            # node labels
            g.ndata['label'] = torch.tensor(labels)
            # node features
            g.ndata['feat'] = torch.tensor(_preprocess_features(features),
                                           dtype=F.data_type_dict['float32'])
            self._num_tasks = onehot_labels.shape[1]
            self._labels = labels
            # reorder graph to obtain better locality.
            self._g = dgl.reorder_graph(g)

        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g

        def __len__(self):
            return 1

분류 데이터셋 프로세싱 코드의 중요한 부분(마스크 분할하기)을 강조하기 위해서 ``process()`` 함수의 코드 일부는 생략해서 간략하게 만들었다.

일반적으로 노드 분류 테스크에서 하나의 그래프만 사용되기 때문에, ``__getitem__(idx)`` 와 ``__len__()`` 함수 구현이 바뀐 점을 알아두자. 마스크는 PyTorch와 TensorFlow에서는 ``bool tensors`` 이고 MXNet에서는 ``float tensors`` 이다.

다음 예는  ``CitationGraphDataset`` 의 서브 클래스인 :class:`dgl.data.CiteseerGraphDataset` 를 사용하는 방법이다.

.. code::

    # load data
    dataset = CiteseerGraphDataset(raw_dir='')
    graph = dataset[0]

    # get split masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    # get node features
    feats = graph.ndata['feat']

    # get labels
    labels = graph.ndata['label']

노드 분류 모델에 대한 전체 가이드는 :ref:`guide_ko-training-node-classification` 를 참고하자.

DGL의 빌트인 데이터셋들은 노드 분류 데이터셋의 여러 예제들을 포함하고 있다.

* :ref:`citationdata`

* :ref:`corafulldata`

* :ref:`amazoncobuydata`

* :ref:`coauthordata`

* :ref:`karateclubdata`

* :ref:`ppidata`

* :ref:`redditdata`

* :ref:`sbmdata`

* :ref:`sstdata`

* :ref:`rdfdata`

링크 예측 데이터셋 프로세싱
~~~~~~~~~~~~~~~~~~~~

링크 예측 데이테셋을 프로세싱하는 것은 주로 데이터셋에 하나의 그래프만 있기 때문에, 노드 분류의 경우와 비슷하다.

예제로 `KnowledgeGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__ 빌트인 데이터셋을 사용하는데, 링크 예측 데이터셋 프로세싱의 주요 부분을 강조하기 위해서 자세한 데이터 프로세싱 코드는 생략했다.

.. code::

    # Example for creating Link Prediction datasets
    class KnowledgeGraphDataset(DGLBuiltinDataset):
        def __init__(self, name, reverse=True, raw_dir=None, force_reload=False, verbose=True):
            self._name = name
            self.reverse = reverse
            url = _get_dgl_url('dataset/') + '{}.tgz'.format(name)
            super(KnowledgeGraphDataset, self).__init__(name,
                                                        url=url,
                                                        raw_dir=raw_dir,
                                                        force_reload=force_reload,
                                                        verbose=verbose)

        def process(self):
            # Skip some processing code
            # === data processing skipped ===

            # splitting mask
            g.edata['train_mask'] = train_mask
            g.edata['val_mask'] = val_mask
            g.edata['test_mask'] = test_mask
            # edge type
            g.edata['etype'] = etype
            # node type
            g.ndata['ntype'] = ntype
            self._g = g

        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g

        def __len__(self):
            return 1


위 코드에서 볼 수 있듯이 분할 마스크들을 그래프의 ``edata`` 필드에 추가한다. 전체 구현은  `KnowledgeGraphDataset 소스 코드 <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__ 를 참고하자.

.. code::

    from dgl.data import FB15k237Dataset

    # load data
    dataset = FB15k237Dataset()
    graph = dataset[0]

    # get training mask
    train_mask = graph.edata['train_mask']
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    src, dst = graph.edges(train_idx)
    # get edge types in training set
    rel = graph.edata['etype'][train_idx]

링크 예측 모델에 대한 전체 가이드는 :ref:`guide_ko-training-link-prediction` 에 있다.

DGL의 빌트인 데이터셋들은 링크 예측 데이터셋의 여러 예제들을 포함하고 있다.

* :ref:`kgdata`

* :ref:`bitcoinotcdata`
