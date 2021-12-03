.. _guide_ko-data-pipeline-savenload:

4.4 데이터 저장과 로딩
------------------

:ref:`(English Version) <guide-data-pipeline-savenload>`

DGL에서는 프로세싱된 데이터를 로컬 디스크에 임시로 저장하기 위해 저장 및 로딩 함수를 구현할 것을 권장한다. 이는 대부분의 경우에 데이터 프로세싱 시간을 상당히 절약할 수 있게한다. DGL은 이를 간단하게 구현하기 위한 4가지 함수를 제공한다:

- :func:`dgl.save_graphs` 와 :func:`dgl.load_graphs` : DGLGraph 객체와 레이블을 로컬 디스크로 저장/로딩함
- :func:`dgl.data.utils.save_info` 와 :func:`dgl.data.utils.load_info` : 데이터셋에 대한 유용한 정보(python의 ``dict`` 객체)를 로컬 디스크로 저장/로딩함

다음 예는 그래프들의 리스트와 데이터셋 정보를 저장하는 것을 보여준다.

.. code:: 

    import os
    from dgl import save_graphs, load_graphs
    from dgl.data.utils import makedirs, save_info, load_info
    
    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']
    
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

단, 프로세싱된 데이터를 저장하는 것이 적합하지 않은 경우도 있다. 예를 들어, 빌트인 데이터셋 중 :class:`~dgl.data.GDELTDataset` 의 경우 프로세스된 데이터가 굉장히 크기 때문에 ``__getitem__(idx)`` 에서 각 데이터 예제들을 처리하는 것이 더 효율적이다.
