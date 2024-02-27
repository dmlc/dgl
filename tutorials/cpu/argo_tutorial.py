"""
Improve Scalability on Multi-Core CPUs
=====================================================

Graph Neural Network (GNN) training suffers from low scalability on multi-core CPUs. 
Specificially, the performance often caps at 16 cores, and no improvement is observed when applying more than 16 cores [#f1]_.
ARGO is a runtime system that offers scalable performance.  
With ARGO enabled, we are able to scale over 64 cores, allowing ARGO to speedup GNN training (in terms of epoch time) by up to 4.30x and 3.32x on a Xeon 8380H and a Xeon 6430L, respectively [#f2]_.
This chapter focus on how to setup ARGO to unleash the potential of multi-core CPUs to speedup GNN training.

Installation
`````````````````````````````

ARGO utilizes the scikit-optimize library for auto-tuning. Please install scikit-optimize to run ARGO:
.. code-block:: shell
    conda install -c conda-forge "scikit-optimize>=0.9.0" 
or
.. code-block:: shell
    pip install scikit-optimize>=0.9

Enabling ARGO on your own GNN program
```````````````````````````````````````````
In this section, we provide a step-by-step tutorial on how to enable ARGO on a DGL program. 
We use the *ogb_example.py* [#f3]_ as an example.
.. note::
    We also provide the complete example file *ogb_example_ARGO.py* [#f4]_ 
    which followed the steps below to enable ARGO on *ogb_example.py*.

Step 1
---------------------------
First, include all necessary packages on top of the file. Please place your file and *argo.py* [#f5]_ in the same directory.

.. code-block:: python
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import torch.multiprocessing as mp
    from argo import ARGO

Step 2
---------------------------
Setup PyTorch Distributed Data Parallel (DDP)

2.1. Add the initialization function on top of the training program, and wrap the ```model``` with the DDP wrapper
.. code-block:: python
    def train(...):
        dist.init_process_group('gloo', rank=rank, world_size=world_size) # newly added
        model = SAGE(...) # original code
        model = DistributedDataParallel(model) # newly added
        ...
     
2.2. In the main program, add the following before launching the training function
.. code-block:: python
    ...
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    mp.set_start_method('fork', force=True)
    train(args, device, data) # original code for launching the training function

Step 3
---------------------------
Enable ARGO by initializing the runtime system, and wrapping the training function
.. code-block:: python
    runtime = ARGO(n_search = 15, epoch = args.num_epochs, batch_size = args.batch_size) # initialization
    runtime.run(train, args=(args, device, data)) # wrap the training function

.. note::
    ARGO takes three input parameters: number of searches *n_search*, number of epochs, and the mini-batch size. 
    Increasing *n_search* potentially leads to a better configuration with less epoch time; 
    however, searching itself also causes extra overhead. We recommend setting *n_search* from 15 to 45 for an optimal overall performance. 

Step 4
---------------------------
Modify the input of the training function, by directly adding ARGO parameters after the original inputs.
   
This is the original function:
.. code-block:: python
   def train(args, device, data):
   
Add the following variables: *rank, world_size, comp_core, load_core, counter, b_size, ep*
.. code-block:: python
    def train(args, device, data, rank, world_size, comp_core, load_core, counter, b_size, ep):

Step 5
---------------------------
Modify the *dataloader* function in the training function
.. code-block:: python
    dataloader = dgl.dataloading.DataLoader(
            g,
            train_nid,
            sampler,
            batch_size=b_size, # modified
            shuffle=True,
            drop_last=False,
            num_workers=len(load_core), # modified
            use_ddp = True) # newly added

Step 6
---------------------------
Enable core-binding by adding *enable_cpu_affinity()* before the training for-loop, and also change the number of epochs into the variable *ep*: 
.. code-block:: python
    with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
        for epoch in range(ep): # change num_epochs to ep
   
Step 7
---------------------------
Last step! Load the model before training and save it afterward.  

Original Program:
.. code-block:: python
    with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
        for epoch in range(ep): 
        ... # training operations
   
Modified:
.. code-block:: python
    PATH = "model.pt"
    if counter[0] != 0:
        checkpoint = th.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
   
    with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
        for epoch in range(ep): 
        ... # training operations
   
    dist.barrier()
    if rank == 0:
        th.save({'epoch': counter[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)
   
Step 8
---------------------------
Done! You can now run your GNN program with ARGO enabled.
.. code-block:: shell
    python <your_code>.py

    
.. rubric:: Footnotes

.. [#f1] https://github.com/dmlc/dgl/blob/master/examples/pytorch/argo/argo_scale.png
.. [#f2] https://arxiv.org/abs/2402.03671
.. [#f3] https://github.com/dmlc/dgl/blob/master/examples/pytorch/argo/ogb_example.py
.. [#f4] https://github.com/dmlc/dgl/blob/master/examples/pytorch/argo/ogb_example_ARGO.py
.. [#f5] https://github.com/dmlc/dgl/blob/master/examples/pytorch/argo/argo.py
"""
