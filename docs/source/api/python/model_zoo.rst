.. _apimodelzoo:

Model Zoo
=========

.. currentmodule:: dgl.model_zoo

Chemistry
---------

Utils
`````

.. autosummary::
    :toctree: ../../generated/

    chem.load_pretrained

Property Prediction
```````````````````

Currently supported model architectures:

* GCNClassifier
* GATClassifier
* MPNN
* SchNet
* MGCN
* AttentiveFP

.. autoclass:: dgl.model_zoo.chem.GCNClassifier
    :members: forward

.. autoclass:: dgl.model_zoo.chem.GATClassifier
    :members: forward

.. autoclass:: dgl.model_zoo.chem.MPNNModel
    :members: forward

.. autoclass:: dgl.model_zoo.chem.SchNet
    :members: forward

.. autoclass:: dgl.model_zoo.chem.MGCNModel
    :members: forward

.. autoclass:: dgl.model_zoo.chem.AttentiveFP
    :members: forward

Generative Models
`````````````````

Currently supported model architectures:

* DGMG
* JTNN

.. autoclass:: dgl.model_zoo.chem.DGMG
    :members: forward

.. autoclass:: dgl.model_zoo.chem.DGLJTNNVAE
    :members: forward

Protein Ligand Binding
``````````````````````

Currently supported model architectures:

* ACNN

.. autoclass:: dgl.model_zoo.chem.ACNN
    :members: forward
