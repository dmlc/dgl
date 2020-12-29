.. _guide_cn-distributed-tools:

7.3 运行分布式训练/推断所需的工具
------------------------------------------------------

:ref:`(English Version) <guide-distributed-tools>`

DGL provides two scripts to assist in distributed training:

DGL提供了两个脚本来协助进行分布式训练：

* *tools/copy_files.py* for copying graph partitions to a graph,
* *tools/launch.py* for launching a distributed training job in a cluster of machines.

* *tools/copy_files.py* 用于将图分区复制到图，
* *tools/launch.py* 用于在机器集群中启动分布式训练任务。

*copy_files.py* copies partitioned data and related files (e.g., training script)
in a machine (where the graph is partitioned) to a cluster of machines (where the distributed
training occurs). The script copies a partition to a machine where the distributed training job
will require the partition. The script contains four arguments:

*copy_files.py* 将计算机(对图进行分区的计算机)中的分区数据和相关文件(例如，训练脚本)
复制到机器集群(在其中进行分布式训练)。该脚本将分区复制到机器上，在该机器上，分布式训练将需要该分区。
该脚本包含四个参数：

* ``--part_config`` specifies the partition configuration file that contains the information
  of the partitioned data in the local machine.
* ``--ip_config`` specifies the IP configuration file of the cluster.
* ``--workspace`` specifies the directory in the training machines where all data related
  to distributed training are stored.
* ``--rel_data_path`` specifies the relative path under the workspace directory where
  the partitioned data will be stored.
* ``--script_folder`` specifies the relative path under the workspace directory where
  user's training scripts are stored.

* ``--part_config`` 指定分区配置文件，该文件包含本地计算机中分区数据的信息。
* ``--ip_config`` 指定集群的IP配置文件。
* ``--workspace`` 指定训练机器中存储与分布式训练有关的所有数据的目录。
* ``--rel_data_path`` 指定工作空间目录下将存储分区数据的相对路径。
* ``--script_folder`` 指定工作空间目录下存储用户的训练脚本的相对路径。

**Note**: *copy_files.py* finds the right machine to store a partition based on the IP
configuration file. Therefore, the same IP configuration file should be used by copy_files.py
and launch.py.

**Note**: *copy_files.py* 根据IP配置文件找到对应的计算机来存储分区。因此，copy_files.py和launch.py应该使用相同的IP配置文件。

DGL provides tools/launch.py to launch a distributed training job in a cluster.
This script makes the following assumptions:

DGL提供了用于启动集群中的分布式训练任务的tools/launch.py。 该脚本有以下假设：

* The partitioned data and the training script have been copied to the cluster or
  a global storage (e.g., NFS) accessible to all machines in the cluster.
* The master machine (where the launch script is executed) has passwordless ssh access
  to all other machines.

* 分区数据和训练脚本已复制到集群或集群中所有计算机均可访问的全局存储空间(例如NFS)。
* 主计算机(在其中执行启动脚本的计算机)具有对集群内所有其他计算机的无密码ssh访问权限。

**Note**: The launch script has to be invoked on one of the machines in the cluster.

**Note**: 必须在集群中的一台计算机上调用启动脚本。

Below shows an example of launching a distributed training job in a cluster.

下面显示了在集群中启动分布式训练任务的示例。

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

配置文件 *ip_config.txt* 包含集群中计算机的IP地址。*ip_config.txt* 的典型示例如下：

.. code:: none

    172.31.19.1
    172.31.23.205
    172.31.29.175
    172.31.16.98

Each row is an IP address of a machine. Optionally, the IP address can be followed by a port
that specifies the port used by network communication between trainers. When the port is not
provided, a default one is ``30050``.

每行是一个计算机的IP地址。IP地址后面还可以有一个端口，用来指定不同训练器之间的网络通信所使用的端口。
如果未提供具体端口，则默认值为30050。

The workspace specified in the launch script is the working directory in the machines,
which contains the training script, the IP configuration file, the partition configuration
file as well as the graph partitions. All paths of the files should be specified as relative
paths to the workspace.

启动脚本中指定的工作空间是计算机中的工作目录，其中包含训练脚本、IP配置文件、分区配置文件以及图分区。
文件的所有路径都应指定为工作空间的相对路径。

The launch script creates a specified number of training jobs (``--num_trainers``) on each machine.
In addition, a user needs to specify the number of sampler processes for each trainer
(``--num_samplers``). The number of sampler processes has to match with the number of worker processes
specified in :func:`~dgl.distributed.initialize`.

启动脚本会在每台计算机上创建指定数量的训练任务(``--num_trainers``)。另外，
用户需要为每个训练器指定采样器进程的数量(``--num_samplers``)。
采样器进程的数量必须匹配 :func:`~dgl.distributed.initialize` 中指定的工作进程的数量。
