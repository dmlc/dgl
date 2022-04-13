.. _apiunifiedtensor:

dgl.contrib.UnifiedTensor
=========

.. automodule:: dgl.contrib

UnifiedTensor enables direct CPU memory access from GPU.
This feature is especially useful when GPUs need to access sparse data structure stored in CPU memory for several reasons (e.g., when node features do not fit in GPU memory).
Without using this feature, sparsely structured data located in CPU memory must be gathered (or packed) before transferring it to the GPU memory because GPU DMA engines can only transfer data in a block granularity.

However, the gathering step wastes CPU cycles and increases the CPU to GPU data copy time.
The goal of UnifiedTensor is to skip such CPU gathering step by letting GPUs to access even non-regular data in CPU memory.
In a hardware-level, this function is enabled by NVIDIA GPUs' unified virtual address (UVM) and zero-copy access capabilities.
For those who wish to further extend the capability of UnifiedTensor may read the following paper (`link <https://arxiv.org/abs/2103.03330>`_) which explains the underlying mechanism of UnifiedTensor in detail.


Base Dataset Class
---------------------------

.. autoclass:: UnifiedTensor
    :members: __getitem__
