.. _apiutilspipeline:

Model Development Pipeline
==========================

.. contents:: Contents
    :local:

Model Evaluation
----------------

A utility class for evaluating model performance on (multi-label) supervised learning.

.. autoclass:: dgllife.utils.Meter
    :members: update, compute_metric

Early Stopping
--------------

Early stopping is a standard practice for preventing models from overfitting and we provide a utility
class for handling it.

.. autoclass:: dgllife.utils.EarlyStopping
    :members:
