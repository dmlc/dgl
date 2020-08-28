.. _guide-distributed-tools:

7.3 Tools for launching distributed training/inference
------------------------------------------------------

DGL provides two scripts to assist in distributed training:

* *tools/copy_files.py* for copying graph partitions to a graph,
* *tools/launch.py* for launching a distributed training job in a cluster of machines.

*copy_files.py* copies partitioned data and related files (e.g., training script)
in a machine (where the graph is partitioned) to a cluster of machines (where the distributed
training occurs). The script copies a partition to a machine where the distributed training job
will require the partition. The script contains four arguments:

* ``--part_config`` specifies the partition configuration file that contains the information
  of the partitioned data in the local machine.
* ``--ip_config`` specifies the IP configuration file of the cluster.
* ``--workspace`` specifies the directory in the training machines where all data related
  to distributed training are stored.
* ``--rel_data_path`` specifies the relative path under the workspace directory where
  the partitioned data will be stored.
* ``--script_folder`` specifies the relative path under the workspace directory where
  user's training scripts are stored.

**Note**: *copy_files.py* finds the right machine to store a partition based on the IP
configuration file. Therefore, the same IP configuration file should be used by copy_files.py
and launch.py.

DGL provides tools/launch.py to launch a distributed training job in a cluster.
This script makes the following assumptions:

* The partitioned data and the training script have been copied to the cluster or
  a global storage (e.g., NFS) accessible to all machines in the cluster.
* The master machine (where the launch script is executed) has passwordless ssh access
  to all other machines.

**Note**: The launch script has to be invoked on one of the machines in the cluster.

Below shows an example of launching a distributed training job in a cluster.

.. code:: none

    python3 tools/launch.py \
    --workspace ~graphsage/ \
    --num_trainers 2 \
    --num_samplers 4 \
    --num_servers 1 \
    --part_config data/ogb-product.json \
    --ip_config ip_config.txt \
    "python3 code/train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 5 --batch-size 1000 --lr 0.1 --num_workers 4"

The configuration file *ip_config.txt* contains the IP addresses of the machines in a cluster.
A typical example of *ip_config.txt* is as follows:

.. code:: none

    172.31.19.1
    172.31.23.205
    172.31.29.175
    172.31.16.98

Each row is an IP address of a machine. Optionally, the IP address can be followed by a port
that specifies the port used by network communication between trainers. When the port is not
provided, a default one is ``30050``.

The workspace specified in the launch script is the working directory in the machines,
which contains the training script, the IP configuration file, the partition configuration
file as well as the graph partitions. All paths of the files should be specified as relative
paths to the workspace.

The launch script creates a specified number of training jobs (``--num_trainers``) on each machine.
In addition, a user needs to specify the number of sampler processes for each trainer
(``--num_samplers``). The number of sampler processes has to match with the number of worker processes
specified in :func:`~dgl.distributed.initialize`.
