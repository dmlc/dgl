
Get Started
==========

Requirement
-----------

DGL-KE works on Linux and Mac. DGL-KE requires Python version 3.5 or later. Python 3.4 or earlier is not tested. DGL-KE supports PyTorch and MXNet.

Install from pip
----------------

Run the following command to install the DGL-KE package with pip.

``pip install dglke (cannot use yet)``
This is a temporary command to install DGL-KE from test pypi.
``pip3 install --index-url https://test.pypi.org/simple/ --no-deps dglke``

Usage
-----

Installation of DGL-KE puts ``dglke_train``\ , ``dglke_eval``\ , ``dglke_partition``\ , ``dglke_dist_train``\ ,  in the system execution path to support multithreading training, multi-GPU training and distributed training.

Data preparation
^^^^^^^^^^^^^^^^

DGL-KE trains embeddings on their own knowledge graphs. In this case, users need to use ``--data_path`` to specify the path to the knowledge graph dataset, ``--data_files`` to specify the triplets of a knowledge graph as well as node/relation Id mapping, ``--format`` to specify the input format of the knowledge graph.

**The input format of users' knowledge graphs**
Users need to store all the data associated with a knowledge graph in the same directory. DGL-KE supports two knowledge graph input formats:


* 
  raw\ *udd*\ [h|r|t], raw user defined dataset. In this format, users only need to provide triplets and the dataloader generates the id mappings for entities and relations in the triplets. The dataloader outputs two files: entities.tsv for entity id mapping and relations.tsv for relation id mapping while loading data. The order of head, relation and tail entities are described in [h|r|t], for example, raw_udd_trh means the triplets are stored in the order of tail, relation and head. The directory contains three files:


  * *train* stores the triplets in the training set. The format of a triplet, e.g., [src_name, rel_name, dst_name], should follow the order specified in [h|r|t]
  * *valid* stores the triplets in the validation set. The format of a triplet, e.g., [src_name, rel_name, dst_name], should follow the order specified in [h|r|t]. This is optional.
  * *test* stores the triplets in the test set. The format of a triplet, e.g., [src_name, rel_name, dst_name], should follow the order specified in [h|r|t]. This is optional.

* 
  udd_[h|r|t], user defined dataset. In this format, user should provide the id mapping for entities and relations. The order of head, relation and tail entities are described in [h|r|t], for example, raw_udd_trh means the triplets are stored in the order of tail, relation and head. The directory should contains five files:


  * *entities* stores the mapping between entity name and entity Id
  * *relations* stores the mapping between relation name relation Id
  * *train* stores the triplets in the training set. The format of a triplet, e.g., [src_id, rel_id, dst_id], should follow the order specified in [h|r|t]
  * *valid* stores the triplets in the validation set. The format of a triplet, e.g., [src_id, rel_id, dst_id], should follow the order specified in [h|r|t]
  * *test* stores the triplets in the test set. The format of a triplet, e.g., [src_id, rel_id, dst_id], should follow the order specified in [h|r|t]

**Builtin knowledge graphs**
DGL-KE provides multiple knowledge graphs, including ``FB15k``\ , ``FB15k-237``\ , ``wn18``\ , ``wn18rr``\ , ``Freebase``. To use these builtin knowledge graphs, users only need to specify the dataset name with ``--dataset`` in ``dglke_train`` and ``dglke_eval`` and ``dglke_partition``. The training script will download the dataset automatically and start to train on the dataset.

Multiprocessing & multi-GPU training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``dglke_train`` trains KG embeddings on CPUs and GPUs in a single machine and saves the trained node embeddings and relation embeddings. Here are some examples of training a knowledge graph.

Single-GPU training: By default, DGL-KE keeps all node and relation embeddings in GPU memory for single-GPU training. Therefore, it cannot train embeddings of large knowledge graphs. Users can specify ``--mix_cpu_gpu`` to keep node and relation embeddings in CPU memory and perform batch computation in GPU. This makes training slower but allows users to train embeddings on a much larger knowledge graph. 

.. code-block::

   DGLBACKEND=pytorch dglke_train --model DistMult --dataset FB15k --batch_size 1024 --neg_sample_size 256 \
    --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 --valid --test -adv \
    --gpu 0 --max_step 40000

Multi-GPU training: when a user specifies multiple GPUs with ``--gpu``\ , DGL-KE train embeddings with all specified GPUs. Multi-GPU training automatically keeps node and relation embeddings on CPUs and dispatch batches to different GPUs. In addition, multi-GPU training automatically uses one process for each GPU.  A user can enable asynchronous gradient update with ``--async_udpate`` in multi-GPU training. This optimization overlaps batch computation in GPU with gradient updates on CPU to speed up the overall training.

.. code-block::

   DGLBACKEND=pytorch dglke_train --model DistMult --dataset FB15k --batch_size 1024 --neg_sample_size 256 \
   --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 16 --valid --test -adv \
   --max_step 5000 --gpu 0 1 2 3 4 5 6 7 --async_update

Multi-CPU training: For CPU training, we suggest users to use multiprocessing training and create one process for each CPU core. A user can specify the number of processes with ``--num_proc``. When the number of processes is the same as the number of CPU cores, a user should use one thread in each process and specify the number of threads per process with ``--num_thread``. With multiprocessing, we need to adjust the number of steps in each process. Usually, we only need the total number of steps performed by all processes equal to the number of steps performed in the single-process training.

.. code-block::

   DGLBACKEND=pytorch dglke_train --model TransE_l2 --dataset FB15k --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 \
   --gamma 19.9 --lr 0.25 --max_step 500 --batch_size_eval 16 --test -adv \
   --regularization_coef 1.00E-09 --num_thread 1 --num_proc 48

 Train KG embeddings on user’s knowledge graph: If users’ knowledge graph does not have entity-to-id mapping or relation-to-id mapping, a user needs to use ``raw_udd_{urt}`` format and provides the files for the training set, validation set and test set. The files for validation set and test set are optional. Suppose we have a knowledge graph stored in the folder ``~/data/foo`` and the training set is stored in ``train.txt``. We use ``--data_path`` to specify the folder where training set is stored and use ``--data_files`` to specify the files for training set, validation set and test set. When loading the knowledge graph, DGL-KE saves the id mapping in ``~/data/foo``.

.. code-block::

   DGLBACKEND=pytorch dglke_train --model DistMult --format raw_udd_hrt \
   --data_path ~/data/foo --data_files train.txt --batch_size 1024 --neg_sample_size 256 \
   --hidden_dim 400 --gamma 143.0 --lr 0.08 --batch_size_eval 1 -adv \
   --gpu 0 --max_step 40000

**Distributed training**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Distributed training usually involves in three steps: 1) partition a knowledge graph, 2) copy partitioned data to worker machines, 3) invoke the distributed training job. We will demonstrate distributed training on **\ *FB15k*\ ** dataset with two machines (Machine_0 and machine_1). Note that, the FB15k is just a small dataset as our toy demo. An interested user can try KGE training on Freebase.

**Step 1: Prepare dataset**

Create a new directory called ``my_task``  on machine_0.

``mkdir my_task``

We use the builtin ``FB15k`` dataset and partition it into **\ *2*\ ** parts (the number of partitions needs to match the number of machines).

``dglke_partition --dataset FB15k -k 2 --data_path ~/my_task``

Note that, in this task we have only two machines, so we set ``-k`` argument to ``2``\ , which is equal to your machine number. You can change the ``—dataset`` to ``Freebase`` if your want to test a much bigger dataset.

**Step 2: Copy data to worker machines**

Create a new file called\ ``ip_config.txt``  in ``my_task`` folder and write the IPs of the machines into it. For example, the ``ip_config.txt`` of mine is as follow:

``172.31.24.245 30050 8``
``172.31.22.41 30050 8``
(DO NOT leave empty lines in this file)

In this\ ``ip_config.txt``\ , we have two lines and each line represent one machine. ``30050`` is the port and ``8`` is the number of kvstore server processes on each machine.

scp the ``my_task`` folder to another machine:
``scp -i your_ssh_key -r ~/my_task 172.31.22.41:~``

**Step 3: Run distributed training job**

Run the following command on ``machine_0`` to start a distributed task:

.. code-block::

   dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
   --num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
   --gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
   --batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --no_save_emb --num_thread 1

Note that, all the path in our arguments are **\ *absolute path. *\ **\ Also, you can remove the\ **\ * *\ **\ ``—ssh_key``\ **\ * *\ **\ argument if your machines can ``ssh`` to each other directly.

If this task executed successfully, you will see the following message:

``...``
``Pull model from kvstore: 99 / 100 ...``
``Pull model from kvstore: 100 / 100 ...``
``Total train time 63.061 seconds``
``Run test, test processes: 16``
``-------------- Test result --------------``
``Test average MRR : 0.6200464812800418``
``Test average MR : 40.713133348004945``
``Test average HITS@1 : 0.4704508134279088``
``Test average HITS@3 : 0.7423270301840158``
``Test average HITS@10 : 0.8478779773492915``

``Exit KVStore service 4, solved message count: 10084``
``Exit KVStore service 2, solved message count: 10025``
``...``

----

Output formats:
^^^^^^^^^^^^^^^

By default, ``dglke_train`` saves the embeddings in the ``ckpts`` folder. Each runs creates a new folder in ``ckpts`` to store the training results. The new folder is named after ``xxxx_yyyy_zz``\ , where ``xxxx`` is the model name, ``yyyy`` is the dataset name, ``zz`` is a sequence number that ensures a unique name for each run. ``dglke_dist_train`` saves all embeddings in ``my_task/ckpts`` on machine 0.

The saved embeddings are stored as numpy ndarrays. The node embedding is saved as ``XXX_YYY_entity.npy``.
The relation embedding is saved as ``XXX_YYY_relation.npy``. ``XXX`` is the dataset name and ``YYY`` is the model name.

 A user can disable saving embeddings with ``--no_save_emb``. This might be useful for some cases, such as hyperparameter tuning.

Evaluation
^^^^^^^^^^

``dglke_eval`` reads the pre-trained node embeddings and relation embeddings and evaluate the embeddings with a link prediction task on the test set of the knowledge graph. This is a common task used for evaluating the quality of pre-trained node/relation embeddings.

.. code-block::

   dglke_eval --model_name DistMult --dataset FB15k --hidden_dim 400 \
       --gamma 143.0 --batch_size_eval 16 --gpu 0 --model_path DistMult_FB15k_emb/
