## DistGNN vertex-cut based graph partitioning (using Libra)

### How to run graph partitioning
```python partition_graph.py --dataset <dataset> --num-parts <num_parts> --out-dir <output_location>```

Example: The following command-line creates 4 partitions of pubmed graph   
``` python partition_graph.py --dataset pubmed --num-parts 4 --out-dir ./```
 
The ouptut partitions are created in the current directory in Libra_result_\<dataset\>/ folder.  
The *upcoming DistGNN* application can directly use these partitions for distributed training.  

### How Libra partitioning works
Libra is a vertex-cut based graph partitioning method. It applies greedy heuristics to uniquely distribute the input graph edges among the partitions. It generates the partitions as a list of edges. Script ```libra_partition.py```  after generates the Libra partitions and converts the Libra output to DGL/DistGNN input format.  


Note: Current Libra implementation is sequential. Extra overhead is paid due to the additional work of format conversion of the partitioned graph.  


### Expected partitioning timinigs  
Cora, Pubmed, Citeseer: < 10 sec (<10GB)  
Reddit: ~150 sec (~ 25GB)  
OGBN-Products: ~200 sec (~30GB)  
Proteins: 1800 sec (Format conversion from public data takes time)  (~100GB)  
OGBN-Paper100M: 2500 sec (~200GB)  


### Settings
Tested with:
Cent OS 7.6
gcc v8.3.0
PyTorch 1.7.1
Python 3.7.10
