.. _apidata:

dgl.data
=========

.. currentmodule:: dgl.data

Dataset Classes
---------------

DGL dataset
```````````

.. autoclass:: DGLDataset
    :members: download, save, load, process, has_cache, __getitem__, __len__

DGL builtin dataset
```````````````````

.. autoclass:: DGLBuiltinDataset
    :members: download

Stanford sentiment treebank dataset
```````````````````````````````````

For more information about the dataset, see `Sentiment Analysis <https://nlp.stanford.edu/sentiment/index.html>`__.

.. autoclass:: SSTDataset
    :members: __getitem__, __len__


Karate club dataset
```````````````````````````````````

.. autoclass:: KarateClubDataset
    :members: __getitem__, __len__


Citation network dataset
```````````````````````````````````

.. autoclass:: CoraGraphDataset
    :members: __getitem__, __len__

.. autoclass:: CiteseerGraphDataset
    :members: __getitem__, __len__

.. autoclass:: PubmedGraphDataset
    :members: __getitem__, __len__


Knowlege graph dataset
```````````````````````````````````

.. autoclass:: FB15k237Dataset
    :members: __getitem__, __len__

.. autoclass:: FB15kDataset
    :members: __getitem__, __len__

.. autoclass:: WN18Dataset
    :members: __getitem__, __len__


RDF datasets
```````````````````````````````````

.. autoclass:: AIFBDataset
    :members: __getitem__, __len__

.. autoclass:: MUTAGDataset
    :members: __getitem__, __len__

.. autoclass:: BGSDataset
    :members: __getitem__, __len__

.. autoclass:: AMDataset
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
    :members: __getitem__, __len__

TU dataset
``````````

.. autoclass:: TUDataset
    :members: __getitem__, __len__

.. autoclass:: LegacyTUDataset
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

