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

.. autoclass:: SSTDataset
    :members: __getitem__, __len__


Karate Club dataset
```````````````````````````````````

.. autoclass:: KarateClubDataset
    :members: __getitem__, __len__


Citation Network dataset
```````````````````````````````````

.. autoclass:: CitationGraphDataset
    :members: __getitem__, __len__


CoraFull dataset
```````````````````````````````````

.. autoclass:: CoraFullDataset
    :members: __getitem__, __len__


Amazon Co-Purchase dataset
```````````````````````````````````

.. autoclass:: AmazonCoBuyComputerDataset
    :members: __getitem__, __len__

.. autoclass:: AmazonCoBuyPhotoDataset
    :members: __getitem__, __len__


Coauthor dataset
```````````````````````````````````

.. autoclass:: CoauthorCSDataset
    :members: __getitem__, __len__

.. autoclass:: CoauthorPhysicsDataset
    :members: __getitem__, __len__


BitcoinOTC dataset
```````````````````````````````````

.. autoclass:: BitcoinOTCDataset
    :members: __getitem__, __len__


ICEWS18 dataset
```````````````````````````````````

.. autoclass:: ICEWS18Dataset
    :members: __getitem__, __len__


QM7b dataset
```````````````````````````````````

.. autoclass:: QM7bDataset
    :members: __getitem__, __len__



GDELT dataset
```````````````````````````````````

.. autoclass:: GDELTDataset
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


Reddit dataset
```````````````````````````````````

.. autoclass:: RedditDataset
    :members: __getitem__, __len__


Symmetric Stochastic Block Model Mixture dataset
```````````````````````````````````

.. autoclass:: SBMMixtureDataset
    :members: __getitem__, __len__, collate_fn

