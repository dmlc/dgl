#!/bin/bash

#python3 format_pipeline.py 2 ~/data/mag_nodes.txt ~/data/mag_edges.txt ~/data/node_feat.dgl ~/data/edge_feat.dgl ~/data/mag_part.2 ~/data/mag.json

python3 format_pipeline.py --world-size 2 --nodes-file mag_nodes.txt --edges-file mag_edges.txt --node-feats-file node_feat.dgl --metis-partitions mag_part.2 --input-dir /home/ubuntu/data --graph-name mag --schema mag.json --num-parts 2 --num-node-weights 4 --workspace /home/ubuntu/data --node-attr-dtype float --output /home/ubuntu/data/outputs --removed-edges mag_removed_edges.txt 
