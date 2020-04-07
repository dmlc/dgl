.. _apiutilssplitters:

Splitting Datasets
==================

We provide multiple splitting methods for datasets.

.. contents:: Contents
    :local:

ConsecutiveSplitter
-------------------

.. autoclass:: dgllife.utils.ConsecutiveSplitter
    :members: train_val_test_split, k_fold_split

RandomSplitter
--------------

.. autoclass:: dgllife.utils.RandomSplitter
    :members: train_val_test_split, k_fold_split

MolecularWeightSplitter
-----------------------

.. autoclass:: dgllife.utils.MolecularWeightSplitter
    :members: train_val_test_split, k_fold_split

ScaffoldSplitter
----------------

.. autoclass:: dgllife.utils.ScaffoldSplitter
    :members: train_val_test_split, k_fold_split

SingleTaskStratifiedSplitter
----------------------------

.. autoclass:: dgllife.utils.SingleTaskStratifiedSplitter
    :members: train_val_test_split, k_fold_split
