First, download the ogbn-products dataset (graphbolt version) from s3 using the following command:

```shell
aws s3 cp s3://dgl-graphbolt/ogbn_products.tar.gz path/to/your/dataset/
```

Next, uncompress the dataset:

```shell
tar -zxvf path/to/your/dataset/ogbn_products.tar.gz
```

Then, make the following changes in node_predicition.py:

```py
dataset = gb.OnDiskDataset("path/to/your/dataset/example_ogbn_products/")
# Remove the following code.
raise NotImplementedError(
    "Please  use your absolute path to the dataset."
    "And Delete this line if you have done so."
)
```

(🌟 Important) You may need to pull the latest code from the DGL master branch and re-install DGL.