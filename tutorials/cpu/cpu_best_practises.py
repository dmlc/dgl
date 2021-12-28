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

**GNU OpenMP**
    Default BKM for setting the number of OMP threads with Pytorch backend:

    ``OMP_NUM_THREADS`` = number of physical cores – ``num_workers``

    Number of physical cores can be checked by using ``lscpu`` ("Core(s) per socket")
    or ``nproc`` command in Linux command line.
    Below simple bash script example for setting the OMP threads and ``pytorch`` backend dataloader workers:

    .. code:: bash

        cores=`nproc`
        num_workers=4
        export OMP_NUM_THREADS=$(($cores-$num_workers))
        python script.py --gpu -1 --num_workers=$num_workers

    Depending on the dataset, model and CPU optimal number of dataloader workers and OpemMP threads may vary
    but close to the general default advise presented above [#f4]_ .

.. rubric:: Footnotes

.. [#f1] https://www.intel.com/content/www/us/en/support/articles/000007645/boards-and-kits/desktop-boards.html
.. [#f2] https://aws.amazon.com/blogs/compute/disabling-intel-hyper-threading-technology-on-amazon-linux/
.. [#f3] https://aws.amazon.com/blogs/compute/disabling-intel-hyper-threading-technology-on-amazon-ec2-windows-instances/
.. [#f4] https://software.intel.com/content/www/us/en/develop/articles/how-to-get-better-performance-on-pytorchcaffe2-with-intel-acceleration.html
"""
