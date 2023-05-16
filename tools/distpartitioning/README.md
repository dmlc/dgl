### xxx_nodes.txt format
This file is used to provide node information to this framework. Following is the format for each line in this file:
```
<node_type> <weight1> <weight2> <weight3> <weight4> <global_type_node_id> <attributes>
```
where node_type is the type id of this node, weights can be any number of columns as determined by the user, global_type_node_id are the contiguous ids starting from `0` for a particular node_type. And attributes can be any number of columns at the end of each line. 

### xxx___edges.txt format
This file is used to provide edge information to this framework. Following is the format for each line in this file:
```
<global_src_id> <global_dst_id> <global_type_edge_id> <edge_type> <attributes>
```
where global_src_id and global_dst_id are two end points of an edge, global_type_edge_id is the unique id assigned to each edge type and are contiguous, and starting from 0, for each edge_type. Attributes can be any number of columns at the end of each line. 

### Naming convention 
`global_` prefix (for any node or edge ids) indicate that these ids are read from graph input files. These ids are allocated to nodes and edges before `data shuffling`. These ids are globally unique across all partitions.
`shuffle_global_` prefix (for any node or edge ids) indicate that these ids are assigned after the `data shuffling` is completed. These ids are globally unique across all partitions.
`part_local_` prefix (for any node or edge ids) indicate that these ids are assigned after the `data shuffling` and are unique within a given partition.
For instance, if a variable is named as `global_src_id` it means that this id is read from the graph input file and is assumed to be globally unique across all partitions. Similarly if a variable is named `part_local_node_id`  then it means that this node_id is assigned after the data shuffling is complete and is unique with a given partition.

### High level description of the algorithm
#### Single file format for graph input files
Here we assume that all the nodes' related data is present in one single file and similarly all the edges are in one single file. 
In this case following steps are executed to write dgl objects for each partition, as assigned my any partitioning algorithm, for example METIS. 
##### Step 1 (Data Loading):
Rank-0 process reads in all the graph files which are xxx_nodes.txt, xxx_edges.txt, node_feats.dgl, edge_feats.dgl and xxx_removed_edges.txt.
Rank-0 process determines the ownership of nodes by using the output of partitioning algorithm (here, we expect the output of partitioning step is a mapping between a node and its partition id for the entire graph). Edge ownership is determined by the `destination` node-id for that edge. Each edge belongs to the partition-id of the destination node-id of each edge. 
##### Step 2 (Data Shuffling):
Rank-0 process will send node-data, edge-data, node-features, edge-features to their respective processes by using the ownership rules described in Step-1. Non-Rank-0 processes will receive their own nodes, edges, node-features and edge-features and store them in local data-structures. Upon completion of sending information Rank-0 process will delete nodes, edges, node-features and edge-features which are not owned by rank-0. 
##### Step 3 (ID assignment and resolution): 
At this time all the ranks will have their own local information in their respective data structures. Then each process will perform the following steps: a) Assign shuffle_global_xxx (here xxx is node_ids and edge_ids) for nodes and edges by performing prefix sum on all ranks. b) Assign part_local_xxx (xxx means node_ids and edge_ids) to nodes and edges so that they can be used to index into the node and edge features, and c) Retrieve shuffle_global_node_ids by using global_node_ids to determine the ownership of any given node. This step is done for the node_ids (present locally on any given rank) for which shuffle_global_node_ids were assigned on a different rank'ed process.
##### Step 4 (Serialization): 
After every rank has global-ids, shuffle_global-ids, part_local-ids for all the nodes and edges present locally, then it proceeds by DGL object creation. Finally Rank-0 process will aggregate graph-level metadata and create a json file with graph-level information. 

### How to use this tool
To run this code on a single machine using multiple processes, use the following command
```
python3 data_proc_pipeline.py --world-size 2 --nodes-file mag_nodes.txt --edges-file mag_edges.txt --node-feats-file node_feat.dgl --metis-partitions mag_part.2 --input-dir /home/ubuntu/data --graph-name mag --schema mag.json --num-parts 2 --num-node-weights 4 --workspace /home/ubuntu/data --node-attr-dtype float --output /home/ubuntu/data/outputs --removed-edges mag_removed_edges.txt
```
Above command, assumes that there are `2` partitions and number of node weights are `4`. All other command line arguments are self-explanatory.
