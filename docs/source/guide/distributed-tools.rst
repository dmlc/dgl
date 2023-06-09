.. _guide-distributed-tools:

7.2 Tools for launching distributed training/inference
------------------------------------------------------

DGL provides a launching script ``launch.py`` under
`dgl/tools <https://github.com/dmlc/dgl/tree/master/tools>`__ to launch a distributed
training job in a cluster. This script makes the following assumptions:

* The partitioned data and the training script have been provisioned to the cluster or
  a shared storage (e.g., NFS) accessible to all the worker machines.
* The machine that invokes ``launch.py`` has passwordless ssh access
  to all other machines. The launching machine must be one of the worker machines.

Below shows an example of launching a distributed training job in a cluster.

.. code:: bash

    python3 tools/launch.py               \
      --workspace /my/workspace/          \
      --num_trainers 2                    \
      --num_samplers 4                    \
      --num_servers 1                     \
      --part_config data/mygraph.json     \
      --ip_config ip_config.txt           \
      "python3 my_train_script.py"

The argument specifies the workspace path, where to find the partition metadata JSON
and machine IP configurations, how many trainer, sampler, and server processes to be launched
on each machine. The last argument is the command to launch which is usually the
model training/evaluation script.

Each line of ``ip_config.txt`` is the IP address of a machine in the cluster.
Optionally, the IP address can be followed by a network port (default is ``30050``).
A typical example is as follows:

.. code:: none

    172.31.19.1
    172.31.23.205
    172.31.29.175
    172.31.16.98

The workspace specified in the launch script is the working directory in the
machines, which contains the training script, the IP configuration file, the
partition configuration file as well as the graph partitions. All paths of the
files should be specified as relative paths to the workspace.

The launch script creates a specified number of training jobs
(``--num_trainers``) on each machine.  In addition, users need to specify the
number of sampler processes for each trainer (``--num_samplers``).
