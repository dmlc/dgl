## How to make your dataset?

Utilize the example provided in https://ogb.stanford.edu/docs/linkprop/ to download the ogbl-citation2 dataset.

```python
import torch
from ogb.linkproppred import LinkPropPredDataset

dataset_names = "ogbl-citation2"
# Set the download directory
data_root = "./dataset" 

# Download.
dataset = LinkPropPredDataset(name=dataset_name, root=data_root)
```

After running the code above, navigate to the respective dataset folder and look for `ogbl_citation2`; all the data you need can be found there. Below is the `metadata.yaml` file we're currently using:

```yaml
dataset_name: ogbl_citation2 
graph:
  nodes:
    - num: 2927963
  edges:
    - format: csv
      path: edges/cite.csv
  feature_data:
feature_data:
  - domain: node
    type: null
    name: feat
    format: numpy
    in_memory: true
    path: data/node-feat.npy
  - domain: node
    type: null
    name: year
    format: numpy
    in_memory: true
    path: data/node-year.npy
tasks:
  - name: "link_prediction"
    num_classes: 2
    train_set:
      - type_name: null
        data:
        - format: numpy
          path: set/train_source_node.npy
          in_memory: true
        - format: numpy
          path: set/train_target_node.npy
    validation_set:
      - type_name: null
        data:
        - format: numpy
          path: set/valid_source_node.npy
        - format: numpy
          path: set/valid_target_node.npy
        - format: numpy
          path: set/valid_target_node_neg.npy
    test_set:
      - type_name: null
        data:
        - format: numpy
          path: set/test_source_node.npy
        - format: numpy
          path: set/test_target_node.npy
        - format: numpy
          path: set/test_target_node_neg.npy
```

You'll need to convert the **raw dataset** into the corresponding structure and organize it into any folder of your choice. The final file structure should look like this:

```
.
├── data
│   ├── node-feat.npy
│   └── node-year.npy
├── edges
│   └── cite.csv
├── metadata.yaml
└── set
    ├── test_source_node.npy
    ├── test_target_node.npy
    ├── test_target_node_neg.npy
    ├── train_source_node.npy
    ├── train_target_node.npy
    ├── valid_source_node.npy
    ├── valid_target_node.npy
    └── valid_target_node_neg.npy
```

## How to run the code?

```bash
python link_prediction.py
```

Results (10 epochs):
```
<Wait for adding>
```