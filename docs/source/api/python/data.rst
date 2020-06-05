.. _apidata:

dgl.data
=========

.. currentmodule:: dgl.data

Utils
-----

.. autosummary::
    :toctree: ../../generated/

    utils.get_download_dir
    utils.download
    utils.check_sha1
    utils.extract_archive
    utils.split_dataset
    utils.save_graphs
    utils.load_graphs
    utils.load_labels

.. autoclass:: dgl.data.utils.Subset
    :members: __getitem__, __len__

Dataset Classes
---------------

Stanford sentiment treebank dataset
```````````````````````````````````

For more information about the dataset, see `Sentiment Analysis <https://nlp.stanford.edu/sentiment/index.html>`__.

.. autoclass:: SST
    :members: __getitem__, __len__


Karate Club dataset
```````````````````````````````````

.. autoclass:: KarateClub
    :members: __getitem__, __len__


Citation Network dataset
```````````````````````````````````

.. autoclass:: CitationGraphDataset
    :members: __getitem__, __len__


CoraFull dataset
```````````````````````````````````

.. autoclass:: CoraFull
    :members: __getitem__, __len__


Amazon Co-Purchase dataset
```````````````````````````````````

.. autoclass:: AmazonCoBuy
    :members: __getitem__, __len__


Coauthor dataset
```````````````````````````````````

.. autoclass:: Coauthor
    :members: __getitem__, __len__


BitcoinOTC dataset
```````````````````````````````````

.. autoclass:: BitcoinOTC
    :members: __getitem__, __len__


ICEWS18 dataset
```````````````````````````````````

.. autoclass:: ICEWS18
    :members: __getitem__, __len__


QM7b dataset
```````````````````````````````````

.. autoclass:: QM7b
    :members: __getitem__, __len__



GDELT dataset
```````````````````````````````````

.. autoclass:: GDELT
    :members: __getitem__, __len__


Mini graph classification dataset
`````````````````````````````````

.. autoclass:: MiniGCDataset
    :members: __getitem__, __len__, num_classes


Graph kernel dataset
````````````````````

For more information about the dataset, see `Benchmark Data Sets for Graph Kernels <https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets>`__.

.. autoclass:: TUDataset
    :members: __getitem__, __len__


Graph isomorphism network dataset
```````````````````````````````````

A compact subset of graph kernel dataset

.. autoclass:: GINDataset
    :members: __getitem__, __len__


Protein-Protein Interaction dataset
```````````````````````````````````

.. autoclass:: PPIDataset
    :members: __getitem__, __len__
