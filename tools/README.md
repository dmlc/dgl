# DGL Utility Scripts

This folder contains the utilities that do not belong to DGL core package as standalone executable
scripts.

## Graph Chunking

`chunk_graph.py` provides an example of chunking an existing DGLGraph object into the on-disk
[chunked graph format](http://13.231.216.217/guide/distributed-preprocessing.html#chunked-graph-format).

<!-- TODO: change the link of documentation once it's merged to master -->

An example of chunking the OGB MAG240M dataset:

```python
import ogb.lsc

dataset = ogb.lsc.MAG240MDataset('.')
etypes = [
    ('paper', 'cites', 'paper'),
    ('author', 'writes', 'paper'),
    ('author', 'affiliated_with', 'institution')]
g = dgl.heterograph({k: tuple(dataset.edge_index(*k)) for k in etypes})
chunk_graph(
    g,
    'mag240m',
    {'paper': {
        'feat': 'mag240m_kddcup2021/processed/paper/node_feat.npy',
        'label': 'mag240m_kddcup2021/processed/paper/node_label.npy',
        'year': 'mag240m_kddcup2021/processed/paper/node_year.npy'}},
    {},
    4,
    'output')
```

The output chunked graph metadata will go as follows (assuming the current directory as
`/home/user`:

```json
{
    "graph_name": "mag240m",
    "node_type": [
        "author",
        "institution",
        "paper"
    ],
    "num_nodes_per_chunk": [
        [
            30595778,
            30595778,
            30595778,
            30595778
        ],
        [
            6431,
            6430,
            6430,
            6430
        ],
        [
            30437917,
            30437917,
            30437916,
            30437916
        ]
    ],
    "edge_type": [
        "author:affiliated_with:institution",
        "author:writes:paper",
        "paper:cites:paper"
    ],
    "num_edges_per_chunk": [
        [
            11148147,
            11148147,
            11148146,
            11148146
        ],
        [
            96505680,
            96505680,
            96505680,
            96505680
        ],
        [
            324437232,
            324437232,
            324437231,
            324437231
        ]
    ],
    "edges": {
        "author:affiliated_with:institution": {
            "format": {
                "name": "csv",
                "delimiter": " "
            },
            "data": [
                "/home/user/output/edge_index/author:affiliated_with:institution0.txt",
                "/home/user/output/edge_index/author:affiliated_with:institution1.txt",
                "/home/user/output/edge_index/author:affiliated_with:institution2.txt",
                "/home/user/output/edge_index/author:affiliated_with:institution3.txt"
            ]
        },
        "author:writes:paper": {
            "format": {
                "name": "csv",
                "delimiter": " "
            },
            "data": [
                "/home/user/output/edge_index/author:writes:paper0.txt",
                "/home/user/output/edge_index/author:writes:paper1.txt",
                "/home/user/output/edge_index/author:writes:paper2.txt",
                "/home/user/output/edge_index/author:writes:paper3.txt"
            ]
        },
        "paper:cites:paper": {
            "format": {
                "name": "csv",
                "delimiter": " "
            },
            "data": [
                "/home/user/output/edge_index/paper:cites:paper0.txt",
                "/home/user/output/edge_index/paper:cites:paper1.txt",
                "/home/user/output/edge_index/paper:cites:paper2.txt",
                "/home/user/output/edge_index/paper:cites:paper3.txt"
            ]
        }
    },
    "node_data": {
        "paper": {
            "feat": {
                "format": {
                    "name": "numpy"
                },
                "data": [
                    "/home/user/output/node_data/paper/feat-0.npy",
                    "/home/user/output/node_data/paper/feat-1.npy",
                    "/home/user/output/node_data/paper/feat-2.npy",
                    "/home/user/output/node_data/paper/feat-3.npy"
                ]
            },
            "label": {
                "format": {
                    "name": "numpy"
                },
                "data": [
                    "/home/user/output/node_data/paper/label-0.npy",
                    "/home/user/output/node_data/paper/label-1.npy",
                    "/home/user/output/node_data/paper/label-2.npy",
                    "/home/user/output/node_data/paper/label-3.npy"
                ]
            },
            "year": {
                "format": {
                    "name": "numpy"
                },
                "data": [
                    "/home/user/output/node_data/paper/year-0.npy",
                    "/home/user/output/node_data/paper/year-1.npy",
                    "/home/user/output/node_data/paper/year-2.npy",
                    "/home/user/output/node_data/paper/year-3.npy"
                ]
            }
        }
    },
    "edge_data": {}
}
```

## Change edge type to canonical edge type for partition configuration json

In the upcoming DGL v1.0, we will require the partition configuration file to contain only canonical edge type. This tool is designed to help migrating existing configuration files from old style to new one.

### Sample Usage

```
python tools/change_etype_to_canonical_etype.py --part_config "{configuration file path}"
```

### Requirement

Partition algorithms produce one configuration file and multiple data folders, and each data folder corresponds to a partition. **This tool needs to read from the partition configuration file (specified by the commandline argument) *and* the graph structure data (stored in `graph.dgl` under the data folder) of the first partition.** They can be local files or shared files among network, if you follow this [official tutorial](https://docs.dgl.ai/en/latest/tutorials/dist/1_node_classification.html#sphx-glr-tutorials-dist-1-node-classification-py) for distributed training, you don't need to care about this as all files are shared by every participant through NFS.

**For example, below is a typical data folder expected by this tool:**
```
data_root_dir/
|-- graph_name.json    # specified by part_config
|-- part0/
    ...
    |-- graph.dgl
...
```

For more information about partition algorithm, see https://docs.dgl.ai/en/latest/generated/dgl.distributed.partition.partition_graph.html.

### Input arguments

1. *part_config*: The path of partition json file. < **Required**>

### Result

This tool changes the key of ``etypes`` and ``edge_map`` from format ``str`` to ``str:str:str`` and it overwrites the original file instead of creating a new one.

E.g. **File content before running the script**
```json
{
    "edge_map": {
        "r1": [ [ 0, 6 ], [ 16, 20 ] ],
        "r2": [ [ 6, 11 ], [ 20, 25 ] ],
        "r3": [ [ 11, 16 ], [ 25, 30 ] ]
    },
    "etypes": {
        "r1": 0,
        "r2": 1,
        "r3": 2
    },
    ...
}
```

**After running**
```json
{
    "edge_map": {
        "n1:r1:n2": [ [ 0, 6 ], [ 16, 20 ] ],
        "n1:r2:n3": [ [ 6, 11 ], [ 20, 25 ] ],
        "n2:r3:n3": [ [ 11, 16 ], [ 25, 30 ] ] },
    "etypes": {
        "n1:r1:n2": 0,
        "n1:r2:n3": 1,
        "n2:r3:n3": 2
    }
    ...
}
```
