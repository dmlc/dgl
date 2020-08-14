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

.. _sstdata:

Stanford sentiment treebank dataset
```````````````````````````````````

For more information about the dataset, see `Sentiment Analysis <https://nlp.stanford.edu/sentiment/index.html>`__.

.. autoclass:: SSTDataset
    :members: __getitem__, __len__

.. _karateclubdata:

Karate club dataset
```````````````````````````````````

.. autoclass:: KarateClubDataset
    :members: __getitem__, __len__

.. _citationdata:

Citation network dataset
```````````````````````````````````

.. autoclass:: CoraGraphDataset
    :members: __getitem__, __len__

.. autoclass:: CiteseerGraphDataset
    :members: __getitem__, __len__

.. autoclass:: PubmedGraphDataset
    :members: __getitem__, __len__

.. _kgdata:

Knowlege graph dataset
```````````````````````````````````

.. autoclass:: FB15k237Dataset
    :members: __getitem__, __len__

.. autoclass:: FB15kDataset
    :members: __getitem__, __len__

.. autoclass:: WN18Dataset
    :members: __getitem__, __len__

.. _rdfdata:

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

.. _corafulldata:

CoraFull dataset
```````````````````````````````````

.. autoclass:: CoraFullDataset
    :members: __getitem__, __len__

.. _amazoncobuydata:

Amazon Co-Purchase dataset
```````````````````````````````````

.. autoclass:: AmazonCoBuyComputerDataset
    :members: __getitem__, __len__

.. autoclass:: AmazonCoBuyPhotoDataset
    :members: __getitem__, __len__

.. _coauthordata:

Coauthor dataset
```````````````````````````````````

.. autoclass:: CoauthorCSDataset
    :members: __getitem__, __len__

.. autoclass:: CoauthorPhysicsDataset
    :members: __getitem__, __len__

.. _bitcoinotcdata:

BitcoinOTC dataset
```````````````````````````````````

.. autoclass:: BitcoinOTCDataset
    :members: __getitem__, __len__


ICEWS18 dataset
```````````````````````````````````

.. autoclass:: ICEWS18Dataset
    :members: __getitem__, __len__

.. _qm7bdata:

QM7b dataset
```````````````````````````````````

.. autoclass:: QM7bDataset
    :members: __getitem__, __len__



GDELT dataset
```````````````````````````````````

.. autoclass:: GDELTDataset
    :members: __getitem__, __len__

.. _minigcdataset:

Mini graph classification dataset
`````````````````````````````````

.. autoclass:: MiniGCDataset
    :members: __getitem__, __len__

.. _tudata:

TU dataset
``````````

.. autoclass:: TUDataset
    :members: __getitem__, __len__

.. autoclass:: LegacyTUDataset
    :members: __getitem__, __len__

.. _gindataset:

Graph isomorphism network dataset
```````````````````````````````````

A compact subset of graph kernel dataset

.. autoclass:: GINDataset
    :members: __getitem__, __len__

.. _ppidata:

Protein-Protein Interaction dataset
```````````````````````````````````

.. autoclass:: PPIDataset
    :members: __getitem__, __len__

.. _redditdata:

Reddit dataset
``````````````

.. autoclass:: RedditDataset
    :members: __getitem__, __len__

.. _sbmdata:

Symmetric Stochastic Block Model Mixture dataset
````````````````````````````````````````````````

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
    utils.save_info
    utils.load_info

.. autoclass:: dgl.data.utils.Subset
    :members: __getitem__, __len__

