DGL-LifeSci: Bringing Graph Neural Networks to Chemistry and Biology
===========================================================================================

DGL-LifeSci is a python package for applying graph neural networks to various tasks in chemistry
and biology, on top of PyTorch and DGL. It provides:

* Various utilities for data processing, training and evaluation.
* Efficient and flexible model implementations.
* Pre-trained models for use without training from scratch.

We cover various applications in our
`examples <https://github.com/dmlc/dgl/tree/master/apps/life_sci/examples>`_, including:

* `Molecular property prediction <https://github.com/dmlc/dgl/tree/master/apps/life_sci/examples/property_prediction>`_
* `Generative models <https://github.com/dmlc/dgl/tree/master/apps/life_sci/examples/generative_models>`_
* `Protein-ligand binding affinity prediction <https://github.com/dmlc/dgl/tree/master/apps/life_sci/examples/binding_affinity_prediction>`_
* `Reaction prediction <https://github.com/dmlc/dgl/tree/master/apps/life_sci/examples/reaction_prediction>`_

Get Started
------------

Follow the :doc:`instructions<install/index>` to install DGL.

.. toctree::
   :maxdepth: 1
   :caption: Installation
   :hidden:
   :glob:

   install/index

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    :hidden:
    :glob:

    api/utils.mols
    api/utils.splitters
    api/utils.pipeline
    api/utils.complexes
    api/data
    api/model.pretrain
    api/model.gnn
    api/model.readout
    api/model.zoo

Free software
-------------
DGL-LifeSci is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions. Join us on `GitHub <https://github.com/dmlc/dgl/tree/master/apps/life_sci>`_.

Index
-----
* :ref:`genindex`
