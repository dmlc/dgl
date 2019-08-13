.. _apidata:

Dataset
=======

.. currentmodule:: dgl.data

Utils
-----

.. autosummary::
    :toctree: ../../generated/

    utils.get_download_dir
    utils.download
    utils.check_sha1
    utils.extract_archive
    utils.Subset

Dataset Classes
---------------

Stanford sentiment treebank dataset
```````````````````````````````````

For more information about the dataset, see `Sentiment Analysis <https://nlp.stanford.edu/sentiment/index.html>`__.

.. autoclass:: SST
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

Molecular Graphs
````````````````

To work on molecular graphs, make sure you have installed `RDKit 2018.09.3 <https://www.rdkit.org/docs/Install.html>`__.

.. autofunction:: dgl.data.molecule.one_hot_encoding
.. autoclass:: dgl.data.molecule.BaseAtomFeaturizer
.. autoclass:: BaseAtomFeaturizer.DefaultAtomFeaturizer
    :members: feat_size, __call__
.. autofunction:: dgl.data.molecule.mol2dgl
.. autofunction:: dgl.data.molecule.consecutive_split
.. autoclass:: BinaryClassificationDataset
.. autoclass:: Tox21
    :members: __len__, __getitem__, num_tasks, task_pos_weights
