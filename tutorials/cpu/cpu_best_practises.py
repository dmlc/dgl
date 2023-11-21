"""
CPU Best Practices
=====================================================

This chapter focus on providing best practises for environment setup
to get the best performance during training and inference on the CPU.

Intel
`````````````````````````````

Hyper-threading
---------------------------

For specific workloads as GNN’s domain, suggested default setting for having best performance
is to turn off hyperthreading.
Turning off the hyper threading feature can be done at BIOS [#f1]_ or operating system level [#f2]_ [#f3]_ .

Alternative memory allocators
---------------------------

Alternative memory allocators, such as *tcmalloc*, might provide significant performance improvements by more efficient memory usage, reducing overhead on unnecessary memory allocations or deallocations. *tcmalloc* uses thread-local caches to reduce overhead on thread synchronization, locks contention by using spinlocks and per-thread arenas respectively and categorizes memory allocations by sizes to reduce overhead on memory fragmentation.

To take advantage of optimizations *tcmalloc* provides, install it on your system (on Ubuntu *tcmalloc* is included in libgoogle-perftools4 package) and add shared library to the LD_PRELOAD environment variable:

.. code-block:: shell

  export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4:$LD_PRELOAD

OpenMP settings
---------------------------

As `OpenMP` is the default parallel backend, we could control performance
including sampling and training via `dgl.utils.set_num_threads()`.

If number of OpenMP threads is not set and `num_workers` in dataloader is set
to 0, the OpenMP runtime typically use the number of available CPU cores by
default. This works well for most cases, and is also the default behavior in DGL.

If `num_workers` in dataloader is set to greater than 0, the number of
OpenMP threads will be set to **1** for each worker process. This is the
default behavior in PyTorch. In this case, we can set the number of OpenMP
threads to the number of CPU cores in the main process.

Performance tuning is highly dependent on the workload and hardware
configuration. We recommend users to try different settings and choose the
best one for their own cases.

**Dataloader CPU affinity**

.. note::

    This feature is available for `dgl.dataloading.DataLoader` only. Not
    available for dataloaders in `dgl.graphbolt` yet.


If number of dataloader workers is more than 0, please consider using **use_cpu_affinity()** method
of DGL Dataloader class, it will generally result in significant performance improvement for training.

*use_cpu_affinity* will set the proper OpenMP thread count (equal to the number of CPU cores allocated for main process),
affinitize dataloader workers for separate CPU cores and restrict the main process to remaining cores

In multiple NUMA nodes setups *use_cpu_affinity* will only use cores of NUMA node 0 by default
with an assumption, that the workload is scaling poorly across multiple NUMA nodes. If you believe
your workload will have better performance utilizing more than one NUMA node, you can pass
the list of cores to use for dataloading (loader_cores) and for compute (compute_cores).

loader_cores and compute_cores arguments (list of CPU cores) can be passed to *enable_cpu_affinity* for more
control over which cores should be used, e.g. in case a workload scales well across multiple NUMA nodes.

Usage:
    .. code:: python

        dataloader = dgl.dataloading.DataLoader(...)
        ...
        with dataloader.enable_cpu_affinity():
            <training loop or inferencing>

**Manual control**

For advanced and more fine-grained control over OpenMP settings please refer to Maximize Performance of Intel® Optimization for PyTorch* on CPU [#f4]_ article

.. rubric:: Footnotes

.. [#f1] https://www.intel.com/content/www/us/en/support/articles/000007645/boards-and-kits/desktop-boards.html
.. [#f2] https://aws.amazon.com/blogs/compute/disabling-intel-hyper-threading-technology-on-amazon-linux/
.. [#f3] https://aws.amazon.com/blogs/compute/disabling-intel-hyper-threading-technology-on-amazon-ec2-windows-instances/
.. [#f4] https://software.intel.com/content/www/us/en/develop/articles/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html
"""
