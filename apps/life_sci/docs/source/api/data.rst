.. _apidata:

Datasets
========

.. contents:: Contents
    :local:

Molecular Property Prediction
-----------------------------

Tox21
`````

.. autoclass:: dgllife.data.Tox21
    :members: task_pos_weights, __getitem__, __len__
    :show-inheritance:

Alchemy for Quantum Chemistry
`````````````````````````````

.. autoclass:: dgllife.data.TencentAlchemyDataset
    :members: set_mean_and_std, __getitem__, __len__

Pubmed Aromaticity
``````````````````

.. autoclass:: dgllife.data.PubChemBioAssayAromaticity
    :members: __getitem__, __len__
    :show-inheritance:

Adapting to New Datasets with CSV
`````````````````````````````````

.. autoclass:: dgllife.data.MoleculeCSVDataset
    :members: __getitem__, __len__

Reaction Prediction
-------------------

USPTO
`````

.. autoclass:: dgllife.data.USPTO
    :members: __getitem__, __len__
    :show-inheritance:

Adapting to New Datasets for Weisfeiler-Lehman Networks
```````````````````````````````````````````````````````

.. autoclass:: dgllife.data.WLNReactionDataset
    :members: __getitem__, __len__

Protein-Ligand Binding Affinity Prediction
------------------------------------------

PDBBind
```````

.. autoclass:: dgllife.data.PDBBind
    :members: __getitem__, __len__
