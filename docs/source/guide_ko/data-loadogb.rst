.. _guide_ko-data-pipeline-loadogb:

4.5 ``ogb`` 패키지를 사용해서 OGB 데이터셋들 로드하기
-------------------------------------------

:ref:`(English Version) <guide-data-pipeline-loadogb>`

`Open Graph Benchmark (OGB) <https://ogb.stanford.edu/docs/home/>`__ 은 벤치마킹 데이터셋의 모음이다. 공식 OGB 패키지 `ogb <https://github.com/snap-stanford/ogb>`__ 는 OBG 데이터셋들을 다운로드해서 :class:`dgl.data.DGLGraph` 객체로 프로세싱하는 API들을 제공한다. 이 절은 기본적인 사용법을 설명한다.

우선 obg 패키지를 pip 명령으로 설치한다.

.. code:: 

    pip install ogb

다음 코드는 *Graph Property Prediction* 테스크를 위한 데이터셋 로딩 방법을 보여준다.

.. code:: 

    # Load Graph Property Prediction datasets in OGB
    import dgl
    import torch
    from ogb.graphproppred import DglGraphPropPredDataset
    from dgl.dataloading import GraphDataLoader
    
    
    def _collate_fn(batch):
        # batch is a list of tuple (graph, label)
        graphs = [e[0] for e in batch]
        g = dgl.batch(graphs)
        labels = [e[1] for e in batch]
        labels = torch.stack(labels, 0)
        return g, labels
    
    # load dataset
    dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()
    # dataloader
    train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, collate_fn=_collate_fn)
    valid_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, collate_fn=_collate_fn)

*Node Property Prediction* 데이터셋을 로딩하는 것이 비슷하지만, 이런 종류의 데이터셋은 오직 한 개의 그래프 객체만 존재한다는 것이 다름을 유의하자.

.. code:: 

    # Load Node Property Prediction datasets in OGB
    from ogb.nodeproppred import DglNodePropPredDataset
    
    dataset = DglNodePropPredDataset(name='ogbn-proteins')
    split_idx = dataset.get_idx_split()
    
    # there is only one graph in Node Property Prediction datasets
    g, labels = dataset[0]
    # get split labels
    train_label = dataset.labels[split_idx['train']]
    valid_label = dataset.labels[split_idx['valid']]
    test_label = dataset.labels[split_idx['test']]

*Link Property Prediction* 데이터셋 역시 데이터셋에 한개의 그래프를 갖고 있다.

.. code:: 

    # Load Link Property Prediction datasets in OGB
    from ogb.linkproppred import DglLinkPropPredDataset
    
    dataset = DglLinkPropPredDataset(name='ogbl-ppa')
    split_edge = dataset.get_edge_split()
    
    graph = dataset[0]
    print(split_edge['train'].keys())
    print(split_edge['valid'].keys())
    print(split_edge['test'].keys())
