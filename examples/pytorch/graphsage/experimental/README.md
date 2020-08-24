## Distributed training

This is an example of training GraphSage in a distributed fashion. To train GraphSage, it has five steps:

### Step 0: set IP configuration file.

User need to set their own IP configuration file before training. For example, if we have four machines in current cluster, the IP configuration
could like this:

```bash
172.31.19.1
172.31.23.205
172.31.29.175
172.31.16.98
```

Users need to make sure that the master node (node-0) has right permission to ssh to all the other nodes.

### Step 1: partition the graph.

The example provides a script to partition some builtin graphs such as Reddit and OGB product graph.
If we want to train GraphSage on 4 machines, we need to partition the graph into 4 parts.

We need to load some function from the parent directory.
```bash
export PYTHONPATH=$PYTHONPATH:..
```

In this example, we partition the OGB product graph into 4 parts with Metis on node-0. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.
```bash
python3 partition_graph.py --dataset ogb-product --num_parts 4 --balance_train --balance_edges
```

### Step 2: copy the partitioned data and files to the cluster

DGL provides a script for copying partitioned data and files to the cluster. Before that, copy the training script to a local folder:

```bash
mkdir ~/dgl_code
cp ~/dgl/examples/pytorch/graphsage/experimental/train_dist.py ~/dgl_code
cp ~/dgl/examples/pytorch/graphsage/experimental/train_dist_unsupervised.py ~/dgl_code
```

The command below copies partition data, ip config file, as well as training scripts to the machines in the cluster. The configuration of the cluster is defined by `ip_config.txt`, The data is copied to `~/graphsage/ogb-product` on each of the remote machines. The training script is copied to `~/graphsage/dgl_code` on each of the remote machines. `--part_config` specifies the location of the partitioned data in the local machine (a user only needs to specify
the location of the partition configuration file).

```bash
python3 ~/dgl/tools/copy_files.py \
--ip_config ip_config.txt \
--workspace ~/graphsage \
--rel_data_path ogb-product \
--part_config data/ogb-product.json \
--script_folder ~/dgl_code
```

After runing this command, user can find a folder called ``graphsage`` on each machine. The folder contains ``ip_config.txt``, ``dgl_code``, and ``ogb-product`` inside.

### Step 3: Launch distributed jobs

DGL provides a script to launch the training job in the cluster. `part_config` and `ip_config`
specify relative paths to the path of the workspace.

```bash
python3 ~/dgl/tools/launch.py \
--workspace ~/graphsage/ \
--num_trainers 1 \
--num_samplers 4 \
--num_servers 1 \
--part_config ogb-product/ogb-product.json \
--ip_config ip_config.txt \
"python3 dgl_code/train_dist.py --graph_name ogb-product --ip_config ip_config.txt --num_servers 1 --num_epochs 30 --batch_size 1000 --num_workers 4"
```

To run unsupervised training:

```bash
python3 ~/dgl/tools/launch.py \
--workspace ~/graphsage/ \
--num_trainers 1 \
--num_samplers 4 \
--num_servers 1 \
--part_config ogb-product/ogb-product.json \
--ip_config ip_config.txt \
"python3 dgl_code/train_dist_unsupervised.py --graph_name ogb-product --ip_config ip_config.txt --num_servers 1 --num_epochs 3 --batch_size 1000 --num_workers 4"
```

By default, this code will run on CPU. If you have GPU support, you can just add a `--num_gpus` argument in user command:

```bash
python3 ~/dgl/tools/launch.py \
--workspace ~/graphsage/ \
--num_trainers 4 \
--num_samplers 4 \
--num_servers 1 \
--part_config ogb-product/ogb-product.json \
--ip_config ip_config.txt \
"python3 dgl_code/train_dist.py --graph_name ogb-product --ip_config ip_config.txt --num_servers 1 --num_epochs 30 --batch_size 1000 --num_workers 4 --num_gpus 4"
```


## Distributed code runs in the standalone mode

The standalone mode is mainly used for development and testing. The procedure to run the code is much simpler.

### Step 1: graph construction.

When testing the standalone mode of the training script, we should construct a graph with one partition.
```bash
python3 partition_graph.py --dataset ogb-product --num_parts 1
```

### Step 2: run the training script

To run supervised training:

```bash
python3 train_dist.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 3 --batch_size 1000 --part_config data/ogb-product.json --standalone
```

To run unsupervised training:

```bash
python3 train_dist_unsupervised.py --graph_name ogb-product --ip_config ip_config.txt --num_epochs 3 --batch_size 1000 --part_config data/ogb-product.json --standalone
```

Note: please ensure that all environment variables shown above are unset if they were set for testing distributed training.
