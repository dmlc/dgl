### Distributed training

```bash
# partition graph
python3 partition_graph.py --dataset ogb-product --num_parts 4 --balance_train --balance_edges

# run server on machine 0
python3 train_dist.py --server --graph-name ogb-product --id 0 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
# run server on machine 1
python3 train_dist.py --server --graph-name ogb-product --id 1 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
# run server on machine 2
python3 train_dist.py --server --graph-name ogb-product --id 2 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt
# run server on machine 3
python3 train_dist.py --server --graph-name ogb-product --id 3 --num-client 4 --conf_path data/ogb-product.json --ip_config ip_config.txt

# run client on machine 0
python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4
# run client on machine 1
python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4
# run client on machine 2
python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4
# run client on machine 3
python3 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 4
```
