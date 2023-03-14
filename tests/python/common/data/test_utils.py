import gzip
import io
import os
import tarfile
import tempfile
import unittest

import backend as F

import dgl
import dgl.data as data
import numpy as np
import pandas as pd
import pytest
import yaml
from dgl import DGLError


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_add_nodepred_split():
    dataset = data.AmazonCoBuyComputerDataset()
    print("train_mask" in dataset[0].ndata)
    data.utils.add_nodepred_split(dataset, [0.8, 0.1, 0.1])
    assert "train_mask" in dataset[0].ndata

    dataset = data.AIFBDataset()
    print("train_mask" in dataset[0].nodes["Publikationen"].data)
    data.utils.add_nodepred_split(
        dataset, [0.8, 0.1, 0.1], ntype="Publikationen"
    )
    assert "train_mask" in dataset[0].nodes["Publikationen"].data


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_extract_archive():
    # gzip
    with tempfile.TemporaryDirectory() as src_dir:
        gz_file = "gz_archive"
        gz_path = os.path.join(src_dir, gz_file + ".gz")
        content = b"test extract archive gzip"
        with gzip.open(gz_path, "wb") as f:
            f.write(content)
        with tempfile.TemporaryDirectory() as dst_dir:
            data.utils.extract_archive(gz_path, dst_dir, overwrite=True)
            assert os.path.exists(os.path.join(dst_dir, gz_file))

    # tar
    with tempfile.TemporaryDirectory() as src_dir:
        tar_file = "tar_archive"
        tar_path = os.path.join(src_dir, tar_file + ".tar")
        # default encode to utf8
        content = "test extract archive tar\n".encode()
        info = tarfile.TarInfo(name="tar_archive")
        info.size = len(content)
        with tarfile.open(tar_path, "w") as f:
            f.addfile(info, io.BytesIO(content))
        with tempfile.TemporaryDirectory() as dst_dir:
            data.utils.extract_archive(tar_path, dst_dir, overwrite=True)
            assert os.path.exists(os.path.join(dst_dir, tar_file))


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_mask_nodes_by_property():
    num_nodes = 1000
    property_values = np.random.uniform(size=num_nodes)
    part_ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
    split_masks = data.utils.mask_nodes_by_property(
        property_values, part_ratios
    )
    assert "in_valid_mask" in split_masks


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Datasets don't need to be tested on GPU.",
)
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Skip MXNet")
def test_add_node_property_split():
    dataset = data.AmazonCoBuyComputerDataset()
    part_ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
    for property_name in ["popularity", "locality", "density"]:
        data.utils.add_node_property_split(dataset, part_ratios, property_name)
        assert "in_valid_mask" in dataset[0].ndata


if __name__ == "__main__":
    test_extract_archive()
    test_add_nodepred_split()
    test_mask_nodes_by_property()
    test_add_node_property_split()
