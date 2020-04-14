.. _apimodelreadout:

Readout for Computing Graph Representations
===========================================

After updating node/edge representations with graph neural networks (GNNs), a common operation is to compute
graph representations out of updated node/edge representations. For example, we need to compute molecular
representations out of atom/bond representations in molecular property prediction. We call the various modules
for computing graph-level representations **readout** as in Neural Message Passing for Quantum Chemistry and this
section lists the readout modules implemented in DGL-LifeSci.

.. contents:: Contents
    :local:

AttentiveFP Readout
-------------------
.. automodule:: dgllife.model.readout.attentivefp_readout
    :members:

MLP Readout
-----------
.. automodule:: dgllife.model.readout.mlp_readout
    :members:

Weighted Sum and Max Readout
----------------------------
.. automodule:: dgllife.model.readout.weighted_sum_and_max
    :members:

Weave Readout
-------------
.. automodule:: dgllife.model.readout.weave_readout
    :members:
