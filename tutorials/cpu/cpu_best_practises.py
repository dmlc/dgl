"""
CPU Best Practices
=====================================================

This chapter focus on providing best practises for environment setup
to get the best performance during training and inference on the CPU.

Intel
`````````````````````````````

Hyper-treading
---------------------------

For specific workloads as GNN’s domain, suggested default setting for having best performance
is to turn off hyperthreading.
Turning off the hyper threading feature can be done at BIOS [#f1]_ or operating system level [#f2]_ [#f3]_ .


OpenMP settings
---------------------------

During training on CPU, the training and dataloading part need to be maintained simultaneously.
Best performance of parallelization in OpenMP
can be achieved by setting up the optimal number of working threads and dataloading workers.
Nodes with high number of CPU cores may benefit from higher number of dataloading workers.
A good starting point could be setting num_threads=4 in Dataloader constructor for nodes with 32 cores or more.
If number of cores is rather small, the best performance might be achieved with just one
dataloader worker or even with dataloader num_threads=0 for dataloading and trainig performed
in the same process

**Dataloader CPU affinity**

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
