.. _stochastic_training-ondisk-dataset-specification:

YAML specification
==================

This document describes the YAML specification of ``metadata.yaml`` file for
``OnDiskDataset``. ``metadata.yaml`` file is used to specify the dataset
information, including the graph structure, feature data and tasks.

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
---------------

The ``dataset_name`` field is used to specify the name of the dataset. It is
user-defined.

``graph``
---------

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

      The ``format`` field is used to specify the format of the edge data. It
      can be ``csv`` or ``numpy``. If it is ``csv``, no ``index`` and ``header``
      fields are needed. If it is ``numpy``, the array requires to be in shape
      of ``(2, num_edges)``. ``numpy`` format is recommended for large graphs.
    - ``path``: ``string``

      The ``path`` field is used to specify the path of the edge data. It is
      relative to the directory of ``metadata.yaml`` file.


``feature_data``
----------------

The ``feature_data`` field is used to specify the feature data. It is a list of
``feature`` objects. Each ``feature`` object has five canonical fields: ``domain``,
``type``, ``name``, ``format`` and ``path``. Any other fields will be passed to
the ``Feature.metadata`` object.

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
---------

The ``tasks`` field is used to specify the tasks. It is a list of ``task``
objects. Each ``task`` object has at least three fields: ``train_set``,
``validation_set``, ``test_set``. And you are free to add other fields
such as ``num_classes`` and all these fields will be passed to the
``Task.metadata`` object.

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
        be either ``seeds``, ``labels`` or ``indexes``. If any other name is used,
        it will be added into the ``MiniBatch`` data fields.
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

