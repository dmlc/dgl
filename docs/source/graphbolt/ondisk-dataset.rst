.. _graphbolt-ondisk-dataset:

Prepare dataset
===============

**GraphBolt** provides the ``OnDiskDataset`` class to help user organize graph
strucutre, feature data and tasks. ``OnDiskDataset`` is designed to efficiently
handle large graphs that do not fit into memory by storing them on disk.

To create an ``OnDiskDataset`` object, you need to organize all the data including
graph structure, feature data and tasks into a directory. The directory should
contain a ``metadata.yaml`` file that describes the metadata of the dataset.

Then just pass the directory path to the ``OnDiskDataset`` constructor to create
the dataset object.

.. code:: python

    from dgl.graphbolt import OnDiskDataset

    dataset = OnDiskDataset('/path/to/dataset')

The returned ``dataset`` object just loads the yaml file and does not load any
data. To load the graph structure, feature data and tasks, you need to call
the ``load`` method.

.. code:: python

    dataset.load()

The reason why we separate the ``OnDiskDataset`` object creation and data loading
is that users may want to change some fields in the ``metadata.yaml`` file before
loading the data. For example, users may want to change the path of the feature
data files to point to a different directory. In this case, users can just
modify the path via ``dataset.yaml_data`` directly. Then call the ``load`` method
again to load the data.

After loading the data, you can access the graph structure, feature data and
tasks through the ``graph``, ``feature`` and ``tasks`` attributes respectively.

.. code:: python

    graph = dataset.graph
    feature = dataset.feature
    tasks = dataset.tasks

The returned ``graph`` is a ``FusedCSCSamplingGraph`` object, which will be used
for sampling. The returned ``feature`` is a ``TorchBasedFeatureStore`` object,
which will be used for feature lookup. The returned ``tasks`` is a list of
``Task`` objects, which will be used for training and evaluation.

Now let's show data folder structure and ``metadata.yaml`` file for homogeneous
graphs and heterogeneous graphs respectively. If you want to know the full YAML
specification, please refer to the `Full YAML specification`_ section.

Homogeneous graph
-----------------

Data folder structure:
^^^^^^^^^^^^^^^^^^^^^

.. code::

    data/
      node_feat.npy
      edge_feat.npy
    edges/
      edges.csv
    set_nc/
      train_seed_nodes.npy
      train_labels.npy
      val_seed_nodes.npy
      val_labels.npy
      test_seed_nodes.npy
      test_labels.npy
    set_lp/
      train_node_pairs.npy
      val_node_pairs.npy
      val_negative_dsts.npy
      test_node_pairs.npy
      test_negative_dsts.npy
    metadata.yaml


``metadata.yaml`` file:
^^^^^^^^^^^^^^^^^^^^^

.. code:: yaml

    dataset_name: homogeneous_graph_nc_lp
    graph:
      nodes:
        - num: 10
      edges:
        - format: csv
          path: edges/edges.csv
    feature_data:
      - domain: node
        name: feat
        format: numpy
        in_memory: true
        path: data/node_feat.npy
      - domain: edge
        name: feat
        format: numpy
        in_memory: true
        path: data/edge_feat.npy
    tasks:
      - name: node_classification
        num_classes: 2
        train_set:
          - data:
              - name: seed_nodes
                format: numpy
                in_memory: true
                path: set_nc/train_seed_nodes.npy
              - name: labels
                format: numpy
                in_memory: true
                path: set_nc/train_labels.npy
        validation_set:
          - data:
              - name: seed_nodes
                format: numpy
                in_memory: true
                path: set_nc/val_seed_nodes.npy
              - name: labels
                format: numpy
                in_memory: true
                path: set_nc/val_labels.npy
        test_set:
          - data:
              - name: seed_nodes
                format: numpy
                in_memory: true
                path: set_nc/test_seed_nodes.npy
              - name: labels
                format: numpy
                in_memory: true
                path: set_nc/test_labels.npy
      - name: link_prediction
        num_classes: 2
        train_set:
          - data:
              - name: node_pairs
                format: numpy
                in_memory: true
                path: set_lp/train_node_pairs.npy
        validation_set:
          - data:
              - name: node_pairs
                format: numpy
                in_memory: true
                path: set_lp/val_node_pairs.npy
              - name: negative_dsts
                format: numpy
                in_memory: true
                path: set_lp/val_negative_dsts.npy
        test_set:
          - data:
              - name: node_pairs
                format: numpy
                in_memory: true
                path: set_lp/test_node_pairs.npy
              - name: negative_dsts
                format: numpy
                in_memory: true
                path: set_lp/test_negative_dsts.npy


For the graph structure, number of nodes is specified by the ``num`` field and
edges are stored in a csv file in format of ``<src, dst>`` like below.

.. code:: csv

    edges.csv

    0,1
    1,2
    2,3
    3,4
    4,5
    5,6
    6,7
    7,8
    8,9


For the feature data, we have feature data named as ``feat`` for nodes and
edges. The feature data are stored in numpy files in shape of ``[num_nodes, 10]``
and ``[num_edges, 10]`` respectively like below.

.. code:: python

    node_feat.npy

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
     [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
     [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
     [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
     [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
     [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
     [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]]

    edge_feat.npy

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
     [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
     [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
     [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
     [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
     [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
     [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]]

For the ``tasks`` field, we have two tasks: ``node_classification`` and
``link_prediction``. For each task, we have three sets: ``train_set``,
``validation_set`` and ``test_set``.

For ``node_classification`` task, we have two fields: ``seed_nodes`` and
``labels``. The ``seed_nodes`` field is used to specify the node IDs for
training and evaluation. The ``labels`` field is used to specify the
labels. Both of them are stored in numpy files with shape of ``[num_nodes]``
like below.

.. code:: python

    train_seed_nodes.npy

    [0 1 2 3 4 5]

    train_labels.npy

    [0 1 0 1 0 1]

    val_seed_nodes.npy

    [6 7]

    val_labels.npy

    [0 1]

    test_seed_nodes.npy

    [8 9]

    test_labels.npy

    [0 1]


For ``link_prediction`` task, we have two fields: ``node_pairs``,
``negative_dsts``.  The ``node_pairs`` field is used to specify the node pairs.
The ``negative_dsts`` field is used to specify the negative destination nodes.
They are stored in numpy file with shape of ``[num_edges, 2]`` and
``[num_edges, num_neg_dsts]`` respectively like below.

.. code:: python

    train_node_pairs.npy

    [[0 1]
     [1 2]
     [2 3]
     [3 4]
     [4 5]
     [5 6]]

    val_node_pairs.npy

    [[6 7]
     [7 8]]

    val_negative_dsts.npy

    [[8 9]
     [8 9]]

    test_node_pairs.npy

    [[8 9]
     [9 0]]

    test_negative_dsts.npy

    [[0 1]
     [0 1]]


.. note::

    The values of ``name`` fields in the ``task`` such as ``seed_nodes``,
    ``labels``, ``node_pairs`` and ``negative_dsts`` are mandatory. They are
    used to specify the data fields of ``MiniBatch`` for sampling. The values
    of ``name`` fields in the ``feature_data`` such as ``feat`` are user-defined.


Heterogeneous graph
-----------------

Data folder structure:
^^^^^^^^^^^^^^^^^^^^^

.. code::

    data/
      user_feat.npy
      item_feat.npy
      user_follow_user_feat.npy
      user_click_item_feat.npy
    edges/
      user_follow_user.csv
      user_click_item.csv
    set_nc/
      user_train_seed_nodes.npy
      user_train_labels.npy
      user_val_seed_nodes.npy
      user_val_labels.npy
      user_test_seed_nodes.npy
      user_test_labels.npy
    set_lp/
      follow_train_node_pairs.npy
      follow_val_node_pairs.npy
      follow_val_negative_dsts.npy
      follow_test_node_pairs.npy
      follow_test_negative_dsts.npy
    metadata.yaml


``metadata.yaml`` file:
^^^^^^^^^^^^^^^^^^^^^

.. code:: yaml

    dataset_name: heterogeneous_graph_nc_lp
    graph:
      nodes:
        - type: user
          num: 10
        - type: item
          num: 10
      edges:
        - type: "user:follow:user"
          format: csv
          path: edges/user_follow_user.csv
        - type: "user:click:item"
          format: csv
          path: edges/user_click_item.csv
    feature_data:
      - domain: node
        type: user
        name: feat
        format: numpy
        in_memory: true
        path: data/user_feat.npy
      - domain: node
        type: item
        name: feat
        format: numpy
        in_memory: true
        path: data/item_feat.npy
      - domain: edge
        type: "user:follow:user"
        name: feat
        format: numpy
        in_memory: true
        path: data/user_follow_user_feat.npy
      - domain: edge
        type: "user:click:item"
        name: feat
        format: numpy
        in_memory: true
        path: data/user_click_item_feat.npy
    tasks:
      - name: node_classification
        num_classes: 2
        train_set:
          - type: user
            data:
              - name: seed_nodes
                format: numpy
                in_memory: true
                path: set/user_train_seed_nodes.npy
              - name: labels
                format: numpy
                in_memory: true
                path: set/user_train_labels.npy
        validation_set:
          - type: user
            data:
              - name: seed_nodes
                format: numpy
                in_memory: true
                path: set/user_val_seed_nodes.npy
              - name: labels
                format: numpy
                in_memory: true
                path: set/user_val_labels.npy
        test_set:
          - type: user
            data:
              - name: seed_nodes
                format: numpy
                in_memory: true
                path: set/user_test_seed_nodes.npy
              - name: labels
                format: numpy
                in_memory: true
                path: set/user_test_labels.npy
      - name: link_prediction
        num_classes: 2
        train_set:
          - type: "user:follow:user"
            data:
              - name: node_pairs
                format: numpy
                in_memory: true
                path: set/follow_train_node_pairs.npy
        validation_set:
          - type: "user:follow:user"
            data:
              - name: node_pairs
                format: numpy
                in_memory: true
                path: set/follow_val_node_pairs.npy
              - name: negative_dsts
                format: numpy
                in_memory: true
                path: set/follow_val_negative_dsts.npy
        test_set:
          - type: "user:follow:user"
            data:
              - name: node_pairs
                format: numpy
                in_memory: true
                path: set/follow_test_node_pairs.npy
              - name: negative_dsts
                format: numpy
                in_memory: true
                path: set/follow_test_negative_dsts.npy

For the graph structure, we have two types of nodes: ``user`` and ``item``
in above example. Number of each node type is specified by the ``num`` field.
We have two types of edges: ``user:follow:user`` and ``user:click:item``.
The edges are stored in two columns of csv files like below.

.. code:: csv

    user_follow_user.csv

    0,1
    1,2
    2,3
    3,4
    4,5
    5,6
    6,7
    7,8
    8,9

    user_click_item.csv

    0,0
    1,1
    2,2
    3,3
    4,4
    5,5
    6,6
    7,7
    8,8
    9,9

For the feature data, we have feature data named as ``feat`` for nodes and
edges. The feature data are stored in numpy files in shape of ``[num_nodes, 10]``
and ``[num_edges, 10]`` respectively like below.

.. code:: python

    user_feat.npy

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
     [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
     [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
     [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
     [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
     [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
     [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]]

    item_feat.npy

    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
     [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
     [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
     [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
     [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
     [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
     [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
     [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]]

    user_follow_user_feat.npy
      
      [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
      [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
      [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
      [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
      [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
      [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
      [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
      [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
      [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]]

    user_click_item_feat.npy
      
      [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
      [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
      [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
      [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
      [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
      [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
      [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
      [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
      [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
      [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]]

For the ``tasks`` field, we have two tasks: ``node_classification`` and
``link_prediction``. For each task, we have three sets: ``train_set``,
``validation_set`` and ``test_set``.

For ``node_classification`` task, we have two fields: ``seed_nodes`` and
``labels``. The ``seed_nodes`` field is used to specify the node IDs for
training and evaluation. The ``labels`` field is used to specify the
labels. Both of them are stored in numpy files with shape of ``[num_nodes]``
like below.

.. code:: python

    user_train_seed_nodes.npy

    [0 1 2 3 4 5]

    user_train_labels.npy

    [0 1 0 1 0 1]

    user_val_seed_nodes.npy

    [6 7]

    user_val_labels.npy

    [0 1]

    user_test_seed_nodes.npy

    [8 9]

    user_test_labels.npy

    [0 1]


For ``link_prediction`` task, we have two fields: ``node_pairs``,
``negative_dsts``.  The ``node_pairs`` field is used to specify the node pairs.
The ``negative_dsts`` field is used to specify the negative destination nodes.
They are stored in numpy file with shape of ``[num_edges, 2]`` and
``[num_edges, num_neg_dsts]`` respectively like below.

.. code:: python

    follow_train_node_pairs.npy

    [[0 1]
     [1 2]
     [2 3]
     [3 4]
     [4 5]
     [5 6]]

    follow_val_node_pairs.npy

    [[6 7]
     [7 8]]

    follow_val_negative_dsts.npy

    [[8 9]
     [8 9]]

    follow_test_node_pairs.npy

    [[8 9]
     [9 0]]

    follow_test_negative_dsts.npy

    [[0 1]
     [0 1]]


Full YAML specification
-----------------------

The full YAML specification of ``metadata.yaml`` file is shown below.

.. code:: yaml

    dataset_name: <string>
    graph:
      nodes:
        - type: <string>
          num: <int>
        - type: <string>
          num: <int>
      edges:
        - type: <string>
          format: <string>
          path: <string>
        - type: <string>
          format: <string>
          path: <string>
    feature_data:
      - domain: node
        type: <string>
        name: <string>
        format: <string>
        in_memory: <bool>
        path: <string>
      - domain: node
        type: <string>
        name: <string>
        format: <string>
        in_memory: <bool>
        path: <string>
      - domain: edge
        type: <string>
        name: <string>
        format: <string>
        in_memory: <bool>
        path: <string>
      - domain: edge
        type: <string>
        name: <string>
        format: <string>
        in_memory: <bool>
        path: <string>
    tasks:
      - name: <string>
        num_classes: <int>
        train_set:
          - type: <string>
            data:
              - name: <string>
                format: <string>
                in_memory: <bool>
                path: <string>
              - name: <string>
                format: <string>
                in_memory: <bool>
                path: <string>
        validation_set:
          - type: <string>
            data:
              - name: <string>
                format: <string>
                in_memory: <bool>
                path: <string>
              - name: <string>
                format: <string>
                in_memory: <bool>
                path: <string>
        test_set:
          - type: <string>
            data:
              - name: <string>
                format: <string>
                in_memory: <bool>
                path: <string>
              - name: <string>
                format: <string>
                in_memory: <bool>
                path: <string>

``dataset_name``
^^^^^^^^^^^^^^^

The ``dataset_name`` field is used to specify the name of the dataset. It is
user-defined.

``graph``
^^^^^^^^

The ``graph`` field is used to specify the graph structure. It has two fields:
``nodes`` and ``edges``.

 - ``nodes``: ``list``

   The ``nodes`` field is used to specify the number of nodes for each node type.
   It is a list of ``node`` objects. Each ``node`` object has two fields: ``type``
   and ``num``.
    - ``type``: ``string``, optional

      The ``type`` field is used to specify the node type. It is ``null`` for
      homogeneous graphs. For heterogeneous graphs, it is the node type.
    - ``num``: ``int``

      The ``num`` field is used to specify the number of nodes for the node type.
      It is mandatory for both homogeneous graphs and heterogeneous graphs.

  - ``edges``: ``list``

    The ``edges`` field is used to specify the edges. It is a list of ``edge``
    objects. Each ``edge`` object has three fields: ``type``, ``format`` and
    ``path``.
    - ``type``: ``string``, optional

      The ``type`` field is used to specify the edge type. It is ``null`` for
      homogeneous graphs. For heterogeneous graphs, it is the edge type.
    - ``format``: ``string``

      The ``format`` field is used to specify the format of the edge data. It can
      only be ``csv`` for now.
    - ``path``: ``string``

      The ``path`` field is used to specify the path of the edge data. It is
      relative to the directory of ``metadata.yaml`` file.


``feature_data``
^^^^^^^^^^^^^^^

The ``feature_data`` field is used to specify the feature data. It is a list of
``feature`` objects. Each ``feature`` object has five fields: ``domain``, ``type``,
``name``, ``format`` and ``path``.

 - ``domain``: ``string``

   The ``domain`` field is used to specify the domain of the feature data. It can
   be either ``node`` or ``edge``.
 - ``type``: ``string``, optional

   The ``type`` field is used to specify the type of the feature data. It is
   ``null`` for homogeneous graphs. For heterogeneous graphs, it is the node or
   edge type.
  - ``name``: ``string``

    The ``name`` field is used to specify the name of the feature data. It is
    user-defined.
  - ``format``: ``string``

    The ``format`` field is used to specify the format of the feature data. It can
    be either ``numpy`` or ``torch``.
  - ``in_memory``: ``bool``, optional

    The ``in_memory`` field is used to specify whether the feature data is loaded
    into memory. It can be either ``true`` or ``false``. Default is ``true``.
  - ``path``: ``string``

    The ``path`` field is used to specify the path of the feature data. It is
    relative to the directory of ``metadata.yaml`` file.


``tasks``
^^^^^^^^

The ``tasks`` field is used to specify the tasks. It is a list of ``task``
objects. Each ``task`` object has at least three fields: ``train_set``,
``validation_set``, ``test_set``. And users are free to add other fields
such as ``num_classes``.

 - ``name``: ``string``, optional

   The ``name`` field is used to specify the name of the task. It is user-defined.
  - ``num_classes``: ``int``, optional

    The ``num_classes`` field is used to specify the number of classes of the task.
  - ``train_set``: ``list``

    The ``train_set`` field is used to specify the training set. It is a list of
    ``set`` objects. Each ``set`` object has two fields: ``type`` and ``data``.
    - ``type``: ``string``, optional

      The ``type`` field is used to specify the node/edge type of the set. It is
      ``null`` for homogeneous graphs. For heterogeneous graphs, it is the node
      or edge type.
    - ``data``: ``list``

      The ``data`` field is used to load ``train_set``. It is a list of ``data``
      objects. Each ``data`` object has four fields: ``name``, ``format``,
      ``in_memory`` and ``path``.

      - ``name``: ``string``

        The ``name`` field is used to specify the name of the data. It is mandatory
        and used to specify the data fields of ``MiniBatch`` for sampling. It can
        be either ``seed_nodes``, ``labels``, ``node_pairs``, ``negative_srcs`` or 
        ``negative_dsts``. If any other name is used, it will be added into the
        ``MiniBatch`` data fields.
      - ``format``: ``string``

        The ``format`` field is used to specify the format of the data. It can be
        either ``numpy`` or ``torch``.
      - ``in_memory``: ``bool``, optional

        The ``in_memory`` field is used to specify whether the data is loaded into
        memory. It can be either ``true`` or ``false``. Default is ``true``.
      - ``path``: ``string``

        The ``path`` field is used to specify the path of the data. It is relative
        to the directory of ``metadata.yaml`` file.
  - ``validation_set``: ``list``
  - ``test_set``: ``list``

    The ``validation_set`` and ``test_set`` fields are used to specify the
    validation set and test set respectively. They are similar to the
    ``train_set`` field.

