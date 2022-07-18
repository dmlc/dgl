.. _guide-distributed-preprocessing:

7.1 Preprocessing for Distributed Training
------------------------------------------

Before launching training jobs, DGL requires the input data to be partitioned
and distributed to the target machines. For relatively small graphs, DGL
provides a partitioning API :func:`~dgl.distributed.partition_graph` that
partitions an in-memory :class:`~dgl.DGLGraph` object. It supports
multiple partitioning algorithms such as random partitioning and
`Metis <http://glaros.dtc.umn.edu/gkhome/views/metis>`__.
The benefit of Metis partitioning is that it can generate
partitions with minimal edge cuts to reduce network communication for distributed training
and inference. DGL uses the latest version of Metis with the options optimized for the real-world
graphs with power-law distribution. After partitioning, the API constructs the partitioned results
in a format that is easy to load during the training.

.. code-block:: none

    data_root_dir/
      |-- mygraph.json              # partition configuration file in JSON
      |-- node_map.npy              # partition ID of each node stored in a numpy array (optional)
      |-- edge_map.npy              # partition ID of each edge stored in a numpy array (optional)
      |-- part0/                    # data for partition 0
      |  |-- node_feats.dgl         # node features stored in binary format
      |  |-- edge_feats.dgl         # edge features stored in binary format
      |  |-- graph.dgl              # graph structure of this partition stored in binary format
      |
      |-- part1/
         |-- node_feats.dgl
         |-- edge_feats.dgl
         |-- graph.dgl


For more details, check out chapter :ref:`guide-distributed-partition-format`
for an in-depth explanation.

To handle massive graph data that cannot fit in the CPU RAM of a
single machine, DGL utilizes data chunking and parallel processing to reduce
memory footprint and running time. The figure below illustrates the core steps:

.. figure:: https://data.dgl.ai/asset/image/guide_7_distdataprep.png

* **Step.1 Partition Preparation:** The input raw graph data may not be
  immediately ready for data partitioning. Moreover, some partitioning
  algorithms like METIS family require extra statistics about the graph.
  Therefore, this step transforms and augments the raw graph data with
  necessary contexts according to the specified partition algorithm.
  Optionally, it supports chunking the input graph into multiple data files to
  ease the subsequent processing steps.
* **Step.2 Graph Partitioning:** Invoke the specified partitioning algorithm to
  generate a partition assignment for each node in the graph. To speedup the
  step, some algorithms (e.g., ParMETIS) support parallel computing using
  multiple machines.
* **Step.3 Data Dispatching:** Given the partition assignment, the step then
  physically partitions the graph data and dispatches them to the machines user
  specified. It also converts the graph data into formats that are suitable for
  distributed training and evaluation.

The whole pipeline is modularized so that each step can be invoked
individually. For example, users can by-pass Step.1 and Step.2 to invoke Step.3
as long as they prepare chunked graph data and partition assignment file
correctly. This enables more advanced customization such as custom partitioning
algorithms.

In the document below, we use the MAG240M-LSC data from `Open Graph Benchmark
<https://ogb.stanford.edu/docs/lsc/mag240m/>`__  as an example to describe the
technical details. The MAG240M-LSC graph is a heterogeneous academic graph
extracted from the Microsoft Academic Graph (MAG), whose schema diagram is
illustrated below:

.. figure:: https://data.dgl.ai/asset/image/guide_7_mag240m.png

Its data files are organized as following:

.. code-block:: none

    /mydata/MAG240M-LSC/
      |-- meta.pt   # # A dictionary of the number of nodes for each type saved by torch.save,
      |             # as well as num_classes
      |-- processed/
        |-- author___affiliated_with___institution/
        |  |-- edge_index.npy            # graph, 713 MB
        |
        |-- paper/
        |  |-- node_feat.npy             # feature, 187 GB, (numpy memmap format)
        |  |-- node_label.npy            # label, 974 MB
        |  |-- node_year.npy             # year, 974 MB
        |
        |-- paper___cites___paper/
        |  |-- edge_index.npy            # graph, 21 GB
        |
        |-- author___writes___paper/
           |-- edge_index.npy            # graph, 6GB


Step.1 Partition Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TBD

Chunked Graph Data Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After step.1, the graph data will be chunked into multiple data files so that
each piece could be loaded to CPU RAM easily. As an example, we have chunked
the MAG240M-LSC graph into 2 parts, creating a data folder as follows:


.. code-block:: none

    /mydata/MAG240M-LSC_chunked/
      |-- meta.json                # metadata json file
      |-- edges/                   # stores edge ID data
      |  |-- writes-part1.csv
      |  |-- writes-part2.csv
      |  |-- affiliated_with-part1.csv
      |  |-- affiliated_with-part2.csv
      |  |-- cites-part1.csv
      |  |-- cites-part1.csv
      |
      |-- node_data/               # stores node feature data
         |-- paper-feat-part1.npy
         |-- paper-feat-part2.npy
         |-- paper-label-part1.npy
         |-- paper-label-part2.npy
         |-- paper-year-part1.npy
         |-- paper-year-part2.npy

All the data files are chunked into two parts, including the node ID data of
each type (e.g., author, institution, paper), edge ID data of each relation
(e.g., writes, affiliates, cites) and node features. All ID data are stored in
CSV (we will illustrate the contents soon) while node features are stored in
numpy arrays.

**Metadata JSON**

The ``meta.json`` stores all the metadata information such as the file names
and the chunk sizes.

.. code-block:: python

    {
       "node_type": ["author", "paper", "institution"],
       "num_nodes_per_chunk": [
           [61191556, 61191556],   # number of author nodes per chunk
           [61191553, 61191552],   # number of paper nodes per chunk
           [12861, 12860]          # number of institution nodes per chunk
       ],
       # The edge type name is a colon-joined string of source, edge, and destination type.
       "edge_type": [
           "author:writes:paper",
           "author:affiliated_with:institution",
           "paper:cites:paper"
       ],
       "num_edges_per_chunk": [
           [193011360, 193011360],  # number of author:writes:paper edges
           [22296293, 22296293],    # number of author:affiliated_with:institution edges
           [648874463, 648874463]   # number of paper:cites:paper edges
       ],
       "edges" : {
            "author:write:paper" : {  # edge type
                 "format" : {"name": "csv", "delimiter": " "},
                 # The list of paths. Can be relative or absolute.
                 "data" : ["edges/writes-part1.csv", "edges/writes-part2.csv"]
            },
            "author:affiliated_with:institution" : {
                 "format" : {"name": "csv", "delimiter": " "},
                 "data" : ["edges/affiliated_with-part1.csv", "edges/affiliated_with-part2.csv"]
            },
            "author:affiliated_with:institution" : {
                 "format" : {"name": "csv", "delimiter": " "},
                 "data" : ["edges/cites-part1.csv", "edges/cites-part2.csv"]
            }
       },
       "node_data" : {
            "paper": {       # node type
                 "feat": {   # feature key
                     "format": {"name": "numpy"},
                     "data": ["node_data/paper-feat-part1.npy", "node_data/paper-feat-part2.npy"]
                 },
                 "label": {   # feature key
                     "format": {"name": "numpy"},
                     "data": ["node_data/paper-label-part1.npy", "node_data/paper-label-part2.npy"]
                 },
                 "year": {   # feature key
                     "format": {"name": "numpy"},
                     "data": ["node_data/paper-year-part1.npy", "node_data/paper-year-part2.npy"]
                 }
            }
       },
       "edge_data" : {}  # MAG240M-LSC does not have edge features
    }

There are three parts in ``meta.json``:

* Graph schema information and chunk sizes, e.g., ``"node_type"`` , ``"num_nodes_per_chunk"``, etc.
* Edge index data under key ``"edges"``.
* Node/edge feature data under keys ``"node_data"`` and ``"edge_data"``. Currently only
  support numpy arrays. More supports will be added in the future.

Example edge index file:

.. code-block:: bash

    # writes-part1.csv
    0 0
    0 1
    0 20
    0 29
    0 1203
    ...

.. note::

    In general, a chunked graph data folder just needs a ``meta.json`` and a bunch
    of data files. The folder structure in this example is not a strict
    requirement as long as ``meta.json`` contains valid file paths.


Step.2 Graph Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step reads the chunked graph data and calculates which partition each node
should belong to. The results are saved in a set of *partition assignment files*.
For example, to randomly partition MAG240M-LSC to two parts, run the
``partition_algo/random.py`` script in the ``tools`` folder:

.. code-block:: bash

    python /my/repo/dgl/tools/partition_algo/random_partition.py
        --in-dir=/mydata/MAG240M-LSC_chunked/
        --out-dir=/mydata/MAG240M-LSC_2parts/
        --num-parts=2

, which outputs files as follows:

.. code-block:: none

    MAG240M-LSC_2parts/
      |-- paper.txt
      |-- author.txt
      |-- institution.txt

Partition assignments of different node types are stored in the file of the
same name whose contents are the partition IDs each node assigned to (row i is
the partition ID of node i).

.. code-block:: bash

    # paper.txt
    0
    0
    0
    0
    ...
    1
    1
    1
    ...

.. note::

    DGL currently requires the number of data chunks and the number of partitions to be the same.

Despite its simplicity, random partitioning may cause frequent cross machine communication.
Check out chapter :ref:`guide-distributed-parmetis` for more advanced options.

Step.3 Data Dispatching
~~~~~~~~~~~~~~~~~~~~~~~~~

DGL provides a ``dispatch_data.py`` script to physically partition the data and
dispatch partitions to each training machines. It will also convert the data
once again to data objects that can be loaded by DGL training processes
efficiently. The entire step can be further accelerated using multi-processing.

.. code-block:: bash

    python /myrepo/dgl/tools/dispatch_data.py         \
       --in-dir=/mydata/MAG240M-LSC_chunked/          \
       --partition-file=/mydata/MAG240M-LSC_2parts/   \
       --out-dir=/data/MAG_LSC_partitioned            \
       --ip-config=ip_config.txt

* ``--in-dir`` specifies the path to the folder of the input chunked graph data produced by Step.1.
* ``--partition-file`` specifies the path to the partition assignment file produced by Step.2.
* ``--out-dir`` specifies the path to stored the data partition on each machine.
* ``--ip-config`` specifies the IP configuration file of the cluster.

An example IP configuration file is as follows:

.. code-block:: bash

    172.31.19.1
    172.31.23.205

During data dispatching, DGL assumes that the combined CPU RAM of the cluster
is able to hold the entire graph data. Moreover, the number of machines (IPs) must be the
same as the number of partitions. Node ownership is determined by the result
of partitioning algorithm where as for edges the owner of the destination node
also owns the edge as well.
