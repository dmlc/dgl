## Distributed training

This is an example of training GraphSage in a distributed fashion. To train GraphSage, it has four steps:

### Step 1: partition the graph.

The example provides a script to partition some builtin graphs such as Reddit and OGB product graph.
If we want to train GraphSage on 4 machines, we need to partition the graph into 4 parts.

We need to load some function from the parent directory.
```bash
export PYTHONPATH=$PYTHONPATH:..
```

In this example, we partition the OGB product graph into 4 parts with Metis. The partitions are balanced with respect to
the number of nodes, the number of edges and the number of labelled nodes.
```bash
# partition graph
python3 partition_graph.py --dataset ogb-product --num_parts 4 --balance_train --balance_edges
```

### Step 2: copy the partitioned data to the cluster

DGL provides a script for copying partitioned data to the cluster. The command below copies partition data
to the machines in the cluster. The configuration of the cluster is defined by `ip_config.txt`,
The data is copied to `~/graphsage/ogb-product` on each of the remote machines. `--part_config`
specifies the location of the partitioned data in the local machine (a user only needs to specify
the location of the partition configuration file).
```bash
python3 ~/dgl/tools/copy_partitions.py --ip_config ip_config.txt \
			--workspace ~/graphsage --rel_data_path ogb-product \
			--part_config data/ogb-product.json 
```

**Note**: users need to make sure that the master node has right permission to ssh to all the other nodes.

Users need to copy the training script to the workspace directory on remote machines as well.

### Step 3: Launch distributed jobs

DGL provides a script to launch the training job in the cluster. `part_config` and `ip_config`
specify relative paths to the path of the workspace.

```bash
python3 ~/dgl/tools/launch.py \
--workspace ~/graphsage/ \
--num_client 4 \
--part_config ogb-product/ogb-product.json \
--ip_config ip_config.txt \
"python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 30 --batch-size 1000"
```

## Distributed code runs in the standalone mode

The standalone mode is mainly used for development and testing. The procedure to run the code is much simpler.

### Step 1: graph construction.

When testing the standalone mode of the training script, we should construct a graph with one partition.
```bash
python3 partition_graph.py --dataset ogb-product --num_parts 1
```

### Step 2: run the training script

```bash
python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --batch-size 1000 --conf_path data/ogb-product.json --standalone
```

Note: please ensure that all environment variables shown above are unset if they were set for testing distributed training.
