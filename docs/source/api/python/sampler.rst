.. apisampler

dgl.contrib.sampling (Deprecating)
======================

.. warning::
   This module is going to be deprecated in favor of :ref:`api-sampling`.

Module for sampling algorithms on graph. Each algorithm is implemented as a
data loader, which produces sampled subgraphs (called Nodeflow) at each
iteration.

.. autofunction:: dgl.contrib.sampling.sampler.NeighborSampler
.. autofunction:: dgl.contrib.sampling.sampler.LayerSampler
.. autofunction:: dgl.contrib.sampling.sampler.EdgeSampler

Distributed sampler
------------------------

.. currentmodule:: dgl.contrib.sampling.dis_sampler
.. autoclass:: SamplerPool

.. autosummary::
    :toctree: ../../generated/

    SamplerPool.start
    SamplerPool.worker

.. autoclass:: SamplerSender

.. autosummary::
    :toctree: ../../generated/

    SamplerSender.send
    SamplerSender.signal

.. autoclass:: SamplerReceiver

.. autosummary::
    :toctree: ../../generated/

    SamplerReceiver.__iter__
    SamplerReceiver.__next__
