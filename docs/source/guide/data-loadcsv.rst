.. _guide-data-pipeline-loadcsv:

4.6 Loading datasets from CSV files
----------------------------------------------

Problem & Motivation
~~~~~~~~~~~~~~~~~~~~

With the growing interests in graph deep learning, many ML researchers or data scientists
wish to try GNN models on custom datasets. Although DGL has a recommended practice on how
a dataset object should behave (see :ref:`guide-data-pipeline-dataset`) once loaded into
RAM, the on-disk storage format is still largely arbitrary. This proposal is to define an
on-disk graph storage format based on Comma Separated Value (CSV) as well as to add a new
dataset class called :class:`~dgl.data.DGLCSVDataset` for loading and processing it to
accord with the current data pipeline practice. We choose CSV format due to its wide
acceptance, good readability and rich set of toolkits for loading, creating and manipulating
it (e.g., ``pandas``).

Use :class:`~dgl.data.DGLCSVDataset` in DGL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a DGLCSVDataset object:

.. code:: python

    import dgl
    ds = dgl.data.DGLCSVDataset('/path/to/dataset')

The returned ``ds`` object is as standard :class:`~dgl.data.DGLDataset`. For example, if the
dataset is for single-graph node classification, you can use it as:

.. code:: python

    g = ds[0] # get the graph
    label = g.ndata['label']
    feat = g.ndata['feat']

Data folder structure
~~~~~~~~~~~~~~~~~~~~~

.. code::

    /path/to/dataset/
    |-- meta.yaml     # metadata of the dataset
    |-- edges_0.csv   # edge data including src_id, dst_id, feature, label and so on
    |-- ...   # you can have as many CSVs for edge data as you want
    |-- nodes_0.csv   # node data including node_id, feature, label and so on
    |-- ...   # you can have as many CSVs for node data as you want
    |-- graphs.csv    # graph-level features

Node/edge/graph-level data are stored in CSV files. ``meta.yaml`` is a metadata file specifying
where to read nodes/edges/graphs data and how to parse them in order to construct the dataset
object. A minimal data folder contains one ``meta.yaml`` and two CSVs, one for node data and one
for edge data, in which case the dataset only contains a single graph with no graph-level data.

Examples
~~~~~~~~

Dataset of a single feature-less graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the dataset contains only one graph with no node or edge features, there need only three
files in the data folder: ``meta.yaml``, one CSV for node IDs and one CSV for edges:

.. code::

    ./mini_featureless_dataset/
    |-- meta.yaml
    |-- nodes.csv
    |-- edges.csv

``meta.yaml`` contains the following information:

.. code:: yaml

    dataset_name: mini_featureless_dataset
    edge_data:
    - file_name: edges.csv
    node_data:
    - file_name: nodes.csv

``nodes.csv`` lists the node IDs under the ``node_id`` field:

.. code::

    node_id
    0
    1
    2
    3
    4

``edges.csv`` lists all the edges in two columns (``src_id`` and ``dst_id``) specifying the
source and destination node ID of each edge:

.. code::

    src_id,dst_id
    4,4
    4,1
    3,0
    4,1
    4,0
    1,2
    1,3
    3,3
    1,1
    4,1

After loaded, the dataset has one graph without any features:

.. code:: python

    import dgl
    dataset = dgl.data.DGLCSVDataset('./mini_featureless_dataset')
    g = dataset[0]  # only one graph
    print(g)
    #Graph(num_nodes=5, num_edges=10,
    #    ndata_schemes={}
    #    edata_schemes={})


A graph without any feature is often of less interest. In the next example, we will show
how node or edge features are stored.

.. note::
    Graph generated here is always directed. If you need reverse edges, please specify manually.

Dataset of a single graph with features and labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the dataset contains only one graph with node or edge features and labels, there still
need only three files in the data folder: ``meta.yaml``, one CSV for node IDs and one CSV
for edges:

.. code::

    ./mini_feature_dataset/
    |-- meta.yaml
    |-- nodes.csv
    |-- edges.csv

``meta.yaml``:

.. code:: yaml

    dataset_name: mini_feature_dataset
    edge_data:
    - file_name: edges.csv
    node_data:
    - file_name: nodes.csv

``edges.csv``:

.. code::

    src_id,dst_id,label,train_mask,val_mask,test_mask,feat
    4,0,2,False,True,True,"[0.5477868606453535, 0.4470617033458436, 0.936706701616337]"
    4,0,0,False,False,True,"[0.9794634290792008, 0.23682038840665198, 0.049629338970987646]"
    0,3,1,True,True,True,"[0.8586722047523594, 0.5746912787380253, 0.6462162561249654]"
    0,1,2,True,False,False,"[0.2730008213674695, 0.5937484188166621, 0.765544096939567]"
    0,2,1,True,True,True,"[0.45441619816038514, 0.1681403185591509, 0.9952376085297715]"
    0,0,0,False,False,False,"[0.4197669213305396, 0.849983324532477, 0.16974127573016262]"
    2,2,1,False,True,True,"[0.5495035052928215, 0.21394654203489705, 0.7174910641836348]"
    1,0,2,False,True,False,"[0.008790817766266334, 0.4216530595907526, 0.529195480661293]"
    3,0,0,True,True,True,"[0.6598715708878852, 0.1932390907048961, 0.9774471538377553]"
    4,0,1,False,False,False,"[0.16846068931179736, 0.41516080644186737, 0.002158116134429955]"


``nodes.csv``:

.. code::

    node_id,label,train_mask,val_mask,test_mask,feat
    0,1,False,True,True,"[0.07816474278491703, 0.9137336384979067, 0.4654086994009452]"
    1,1,True,True,True,"[0.05354099924658973, 0.8753101998792645, 0.33929432608774135]"
    2,1,True,False,True,"[0.33234211884156384, 0.9370522452510665, 0.6694943496824788]"
    3,0,False,True,False,"[0.9784264442230887, 0.22131880861864428, 0.3161154827254189]"
    4,1,True,True,False,"[0.23142237259162102, 0.8715767748481147, 0.19117861103555467]"

After loaded, the dataset has one graph with features and labels:

.. code:: python

    import dgl
    dataset = dgl.data.DGLCSVDataset('./mini_feature_dataset')
    g = dataset[0]  # only one graph
    print(g)
    #Graph(num_nodes=5, num_edges=10,
    #    ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'train_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool), 'feat': Scheme(shape=(3,), dtype=torch.float64)}
    #    edata_schemes={'label': Scheme(shape=(), dtype=torch.int64), 'train_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool), 'feat': Scheme(shape=(3,), dtype=torch.float64)})

.. note::
    All columns will be read, parsed and set as edge/node attributes except ``node_id`` in ``nodes.csv``,
    ``src_id`` and ``dst_id`` in ``edges.csv``. User is able to access directly like: ``g.ndata[‘label’]``.
    The keys in ``g.ndata`` and ``g.edata`` are the same as original column names. Data format is
    infered automatically during parse.

Dataset of a single heterogeneous graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the dataset contains only one heterograph with 2 node/edge types respectively, there need
only 5 files in the data folder: ``meta.yaml``, 2 CSV for nodes and 2 CSV for edges:

.. code::

    ./mini_hetero_dataset/
    |-- meta.yaml
    |-- nodes_0.csv
    |-- nodes_1.csv
    |-- edges_0.csv
    |-- edges_1.csv

``meta.yaml``
For heterogeneous graph, ``etype`` and ``ntype`` are MUST HAVE and UNIQUE in ``edge_data`` and
``node_data`` respectively, or only the last etype/ntype is kept when generating graph as all
of them use the same default etype/ntype name. What's more, each node/edge csv file should
contains single and unique ntype/etype. If there exist several ntype/etypes, multiple node/edge
csv files are required.

.. code:: yaml

    dataset_name: mini_hetero_dataset
    edge_data:
    - file_name: edges_0.csv
      etype:
      - user
      - follow
      - user
    - file_name: edges_1.csv
      etype:
      - user
      - like
      - item
    node_data:
    - file_name: nodes_0.csv
      ntype: user
    - file_name: nodes_1.csv
      ntype: item

``edges_0.csv``, ``edges_1.csv`` (Both are the same, just for example only.)

.. code::

    src_id,dst_id,label,feat
    4,4,1,"0.736833152378035,0.10522806046048205,0.9418796835016118"
    3,4,2,"0.5749339182767451,0.20181320245665535,0.490938012147181"
    1,4,2,"0.7697294432580938,0.49397782380750765,0.10864079337442234"
    0,4,0,"0.1364240150959487,0.1393107840629273,0.7901988878812207"
    2,3,1,"0.42988138237505735,0.18389137408509248,0.18431292077750894"
    0,4,2,"0.8613368738351794,0.67985810014162,0.6580438064356824"
    2,4,1,"0.6594951663841697,0.26499036865016423,0.7891429392727503"
    4,1,0,"0.36649684241348557,0.9511783938523962,0.8494919263589972"
    1,1,2,"0.698592283371875,0.038622249776255946,0.5563827995742111"
    0,4,1,"0.5227112950269823,0.3148264185956532,0.47562693094002173"

``nodes_0.csv``, ``nodes_1.csv`` (Both are the same, just for example only.)

.. code::

    node_id,label,feat
    0,2,"0.5400687466285844,0.7588441197954202,0.4268254673041745"
    1,1,"0.08680051341900807,0.11446843700743892,0.7196969604886617"
    2,2,"0.8964389655603473,0.23368113896545695,0.8813472954005022"
    3,1,"0.5454703921677284,0.7819383771535038,0.3027939452162367"
    4,1,"0.5365210052235699,0.8975240205792763,0.7613943085507672"

After loaded, the dataset has one heterograph with features and labels:

.. code:: python

    import dgl
    dataset = dgl.data.DGLCSVDataset('./mini_hetero_dataset')
    g = dataset[0]  # only one graph
    print(g)
    #Graph(num_nodes={'item': 5, 'user': 5},
    #    num_edges={('user', 'follow', 'user'): 10, ('user', 'like', 'item'): 10},
    #    metagraph=[('user', 'user', 'follow'), ('user', 'item', 'like')])
    g.nodes['user'].data
    #{'label': tensor([2, 1, 2, 1, 1]), 'feat': tensor([[0.5401, 0.7588, 0.4268],
    #        [0.0868, 0.1145, 0.7197],
    #        [0.8964, 0.2337, 0.8813],
    #        [0.5455, 0.7819, 0.3028],
    #        [0.5365, 0.8975, 0.7614]], dtype=torch.float64)}
    g.edges['like'].data
    #{'label': tensor([1, 2, 2, 0, 1, 2, 1, 0, 2, 1]), 'feat': tensor([[0.7368, 0.1052, 0.9419],
    #        [0.5749, 0.2018, 0.4909],
    #        [0.7697, 0.4940, 0.1086],
    #        [0.1364, 0.1393, 0.7902],
    #        [0.4299, 0.1839, 0.1843],
    #        [0.8613, 0.6799, 0.6580],
    #        [0.6595, 0.2650, 0.7891],
    #        [0.3665, 0.9512, 0.8495],
    #        [0.6986, 0.0386, 0.5564],
    #        [0.5227, 0.3148, 0.4756]], dtype=torch.float64)}

Dataset of multiple graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the dataset contains multiple graphs(for now, only homograph is supported) with node/edge/graph
level features, there need only 4 files in the data folder: ``meta.yaml``, one CSV file for
nodes/edge/graphs respectively:

.. code::

    ./mini_multi_dataset/
    |-- meta.yaml
    |-- nodes.csv
    |-- edges.csv
    |-- graphs.csv

``meta.yaml``:

.. code:: yaml

    dataset_name: mini_multi_dataset
    edge_data:
    - file_name: edges.csv
    node_data:
    - file_name: nodes.csv
    graph_data:
    file_name: graphs.csv

.. note::
    ``graph_id`` should be specified in nodes/edges/graphs CSV files or default value ``0`` is
    used instead which probably caused unexpected/undefined behavior.

``edges.csv``:

.. code::

    graph_id,src_id,dst_id,feat
    0,0,4,"0.39534097273254654,0.9422093637539785,0.634899790318452"
    0,3,0,"0.04486384200747007,0.6453746567017163,0.8757520744192612"
    0,3,2,"0.9397636966928355,0.6526403892728874,0.8643238446466464"
    0,1,1,"0.40559906615287566,0.9848072295736628,0.493888090726854"
    0,4,1,"0.253458867276219,0.9168191778828504,0.47224962583565544"
    0,0,1,"0.3219496197945605,0.3439899477636117,0.7051530741717352"
    0,2,1,"0.692873149428549,0.4770019763881086,0.21937428942781778"
    0,4,0,"0.620118223673067,0.08691420300562658,0.86573472329756"
    0,2,1,"0.00743445923710373,0.5251800239734318,0.054016385555202384"
    0,4,1,"0.6776417760682221,0.7291568018841328,0.4523600060547709"
    1,1,3,"0.6375445528248924,0.04878384701995819,0.4081642382536248"
    1,0,4,"0.776002616178397,0.8851294998284638,0.7321742043493028"
    1,1,0,"0.0928555079874982,0.6156748364694707,0.6985674921582508"
    1,0,2,"0.31328748118329997,0.8326121496142408,0.04133991340612775"
    1,1,0,"0.36786902637778773,0.39161865931662243,0.9971749359397111"
    1,1,1,"0.4647410679872376,0.8478810655406659,0.6746269314422184"
    1,0,2,"0.8117650553546695,0.7893727601272978,0.41527155506593394"
    1,1,3,"0.40707309111756307,0.2796588354307046,0.34846782265758314"
    1,1,0,"0.18626464175355095,0.3523777809254057,0.7863421810531344"
    1,3,0,"0.28357022069634585,0.13774964202156292,0.5913335505943637"

``nodes.csv``:

.. code::

    graph_id,node_id,feat
    0,0,"0.5725330322207948,0.8451870383322376,0.44412796119211184"
    0,1,"0.6624186423087752,0.6118386331195641,0.7352138669985214"
    0,2,"0.7583372765843964,0.15218126307872892,0.6810484348765842"
    0,3,"0.14627522432017592,0.7457985352827006,0.1037097085190507"
    0,4,"0.49037522512771525,0.8778998699783784,0.0911194482288028"
    1,0,"0.11158102039672668,0.08543289788089736,0.6901745368284345"
    1,1,"0.28367647637469273,0.07502571020414439,0.01217200152200748"
    1,2,"0.2472495901894738,0.24285506608575758,0.6494437360242048"
    1,3,"0.5614197853127827,0.059172654879085296,0.4692371689047904"
    1,4,"0.17583413999295983,0.5191278830882644,0.8453123358491914"

``graphs.csv``:

.. code::

    graph_id,feat,label
    0,"0.7426272601929126,0.5197462471155317,0.8149104951283953",0
    1,"0.534822233529295,0.2863627767733977,0.1154897249106891",0


After loaded, the dataset has multiple homographs with features and labels:

.. code:: python

    import dgl
    dataset = dgl.data.DGLCSVDataset('./mini_multi_dataset')
    print(len(dataset))
    #2
    graph, label = dataset[0]
    print(graph, label)
    #Graph(num_nodes=5, num_edges=10,
    #    ndata_schemes={'feat': Scheme(shape=(3,), dtype=torch.float64)}
    #    edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.float64)}) tensor(0)
    print(dataset.data)
    #{'feat': tensor([[0.7426, 0.5197, 0.8149],
    #        [0.5348, 0.2864, 0.1155]], dtype=torch.float64), 'label': tensor([0, 0])}

YAML Specification
~~~~~~~~~~~~~~~~~~

Example
^^^^^^^

In the YAML file below, all supported keys are listed together including those that have default
values though not all the keys are required for a specific use.

.. code:: yaml

    version: 1.0.0
    dataset_name: full_yaml
    separator: ','
    edge_data:
    - file_name: edges_0.csv
      etype:
      - user
      - follow
      - user
      src_id_field: src_id
      dst_id_field: dst_id
    - file_name: edges_1.csv
      etype:
      - user
      - like
      - item
      src_id_field: src_id
      dst_id_field: dst_id
    node_data:
    - file_name: nodes_0.csv
      ntype: user
      node_id_field: node_id
    - file_name: nodes_1.csv
      ntype: item
      node_id_field: node_id
    graph_data:
      file_name: graphs.csv
      graph_id_field: graph_id

Top-level keys
^^^^^^^^^^^^^^

At the top level, only 6 keys are available.

``version``
Optional. String. It specifies which version of ``meta.yaml`` is used. more feature may be added and
version is changed accordingly.

``dataset_name``
Required. String. It specifies the dataset name.

``separator``
Optional. String. It specifies how to parse data in CSV files. Default value: ``,``.

``edge_data``
Required. List of dict. It includes several sub-keys to help parse edges from CSV files.

``node_data``
Required. List of dict. It includes several sub-keys to help parse nodes from CSV files.

``graph_data``
Required. Dict. It includes several sub-keys to help parse graph-level information from CSV files.

Keys for ``edge_data``
^^^^^^^^^^^^^^^^^^^^^^

``file_name``
Required. String. It specifies the file name which stores edge data.

``etype``
Optional. List of string. It specifies the canonical edge type.

``src_id_field``
Optional. String. It specifies which column to be read for src ids. Default value: ``src_id``.

``dst_id_field``
Optional. String. It specifies which column to be read for dst ids. Default value: ``dst_id``.

Keys for ``node_data``
^^^^^^^^^^^^^^^^^^^^^^

``file_name``
Required. String. It specifies the file name which stores node data.

``ntype``
Optional. List of string. It specifies the canonical node type.

``node_id_field``
Optional. String. It specifies which column to be read for node ids. Default value: ``node_id``.

Keys for ``graph_data``
^^^^^^^^^^^^^^^^^^^^^^

``file_name``
Required. String. It specifies the file name which stores graph data.

``graph_id_field``
Optional. String. It specifies which column to be read for graph ids. Default value: ``graph_id``.

Parse node/edge/grpah data on your own
~~~~~~~~~~~~~~~~~~~~~~~~

In default, all the data are attached to ``g.ndata`` with the same key as column name in ``nodes.csv``
except ``node_id``. So does data in ``edges.csv``. Data is auto-formatted via ``pandas`` unless it's
a string of float values(feature data is often of this format). For better experience, user is able
to self-define node/edge/graph data parser which is callable and accept ``pandas.DataFrame`` as input
data. Then pass such callable instance while instantiating ``DGLCSVDataset``. Below is an example.

``SelfDefinedDataParser``:

.. code:: python

    import numpy as np
    import ast
    import pandas as pd

    class SelfDefinedDataParser:
        """Convert labels which are in string format into numeric values.
        """
        def __call__(self, df: pd.DataFrame):
            data = {}
            for header in df:
                if 'Unnamed' in header:
                    print("Unamed column is found. Ignored...")
                    continue
                dt = df[header].to_numpy().squeeze()
                if header == 'label':
                    dt = np.array([1 if e == 'positive' else 0 for e in dt])
                data[header] = dt
            return data

Example:

``customized_parser_dataset``:

.. code::

    ./customized_parser_dataset/
    |-- meta.yaml
    |-- nodes.csv
    |-- edges.csv

``meta.yaml``:

.. code:: yaml

    dataset_name: customized_parser_dataset
    edge_data:
    - file_name: edges.csv
    node_data:
    - file_name: nodes.csv

``edges.csv``:

.. code::

    src_id,dst_id,label
    4,0,positive
    4,0,negative
    0,3,positive
    0,1,positive
    0,2,negative
    0,0,positive
    2,2,negative
    1,0,positive
    3,0,negative
    4,0,positive

``nodes.csv``:

.. code::

    node_id,label
    0,positive
    1,negative
    2,positive
    3,negative
    4,positive

After loaded, the dataset has one graph with features and labels:

.. code:: python

    import dgl
    dataset = dgl.data.DGLCSVDataset('./customized_parser_dataset',
                                     node_data_parser={'_V': SelfDefinedDataParser()},
                                     edge_data_parser={('_V','_E','_V'): SelfDefinedDataParser()})
    print(dataset[0].ndata['label'])
    #tensor([1, 0, 1, 0, 1])
    print(dataset[0].edata['label'])
    #tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])

FAQs:
~~~~~

What's the data type in CSV files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A default data parser is used for parsing node/edge/graph csv files in default which infer data type
automatically. ID related data such as ``node_id``, ``src_id``, ``dst_id``, ``graph_id`` are required
to be ``numeric`` as these fields are used for constructing graph. Any other data will be attached to
``g.ndata`` or ``g.edata`` directly, so it's user's responsibility to make sure the data type is expected
when using within graph. In particular, ``string`` data which is composed of ``float`` values is splitted
and cast into float value array by default data parser.

What if some lines in CSV have missing values in several fields?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It’s undefined behavior. Please make sure the data is complete.

What if ``graph_id`` is not specified in CSV?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a single graph, such field in ``edge_data`` and ``node_data`` is not used at all. So it’s ok
to ignore it. For multiple graphs, ``graph_id`` should be provided, or all edge/node data will be
regarded as ``graph_id = 0``. This usually is not what you expect.
