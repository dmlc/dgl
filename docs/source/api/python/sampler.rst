.. apisampler

Graph samplers
==============

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
