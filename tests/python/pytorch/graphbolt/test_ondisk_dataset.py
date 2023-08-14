import os
import re
import tempfile
import unittest

import gb_test_utils as gbt

import numpy as np
import pandas as pd

import pydantic
import pytest
import torch
import yaml
from dgl import graphbolt as gb


def test_OnDiskDataset_TVTSet_exceptions():
    """Test excpetions thrown when parsing TVTSet."""
    with tempfile.TemporaryDirectory() as test_dir:
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")

        # Case 1: ``format`` is invalid.
        yaml_content = """
        tasks:
          - name: node_classification
            train_set:
              - type: paper
                data:
                  - format: torch_invalid
                    path: set/paper-train.pt
        """
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        with pytest.raises(pydantic.ValidationError):
            _ = gb.OnDiskDataset(test_dir)

        # Case 2: ``type`` is not specified while multiple TVT sets are
        # specified.
        yaml_content = """
            tasks:
              - name: node_classification
                train_set:
                - type: null
                  data:
                    - format: numpy
                      path: set/train.npy
                - type: null
                  data:
                    - format: numpy
                      path: set/train.npy
        """
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        with pytest.raises(
            AssertionError,
            match=r"Only one TVT set is allowed if type is not specified.",
        ):
            _ = gb.OnDiskDataset(test_dir)


def test_OnDiskDataset_TVTSet_ItemSet_id_label():
    """Test TVTSet which returns ItemSet with IDs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_ids = np.arange(1000)
        train_ids_path = os.path.join(test_dir, "train_ids.npy")
        np.save(train_ids_path, train_ids)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        validation_ids = np.arange(1000, 2000)
        validation_ids_path = os.path.join(test_dir, "validation_ids.npy")
        np.save(validation_ids_path, validation_ids)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_labels_path = os.path.join(test_dir, "validation_labels.npy")
        np.save(validation_labels_path, validation_labels)

        test_ids = np.arange(2000, 3000)
        test_ids_path = os.path.join(test_dir, "test_ids.npy")
        np.save(test_ids_path, test_ids)
        test_labels = np.random.randint(0, 10, size=1000)
        test_labels_path = os.path.join(test_dir, "test_labels.npy")
        np.save(test_labels_path, test_labels)

        # Case 1:
        #   all TVT sets are specified.
        #   ``type`` is not specified or specified as ``null``.
        #   ``in_memory`` could be ``true`` and ``false``.
        yaml_content = f"""
            tasks:
              - name: node_classification
                num_classes: 10
                train_set:
                  - type: null
                    data:
                      - format: numpy
                        in_memory: true
                        path: {train_ids_path}
                      - format: numpy
                        in_memory: true
                        path: {train_labels_path}
                validation_set:
                  - data:
                      - format: numpy
                        in_memory: true
                        path: {validation_ids_path}
                      - format: numpy
                        in_memory: true
                        path: {validation_labels_path}
                test_set:
                  - type: null
                    data:
                      - format: numpy
                        in_memory: true
                        path: {test_ids_path}
                      - format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify tasks.
        assert len(dataset.tasks) == 1
        assert dataset.tasks[0].metadata["name"] == "node_classification"
        assert dataset.tasks[0].metadata["num_classes"] == 10

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (id, label) in enumerate(train_set):
            assert id == train_ids[i]
            assert label == train_labels[i]
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (id, label) in enumerate(validation_set):
            assert id == validation_ids[i]
            assert label == validation_labels[i]
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (id, label) in enumerate(test_set):
            assert id == test_ids[i]
            assert label == test_labels[i]
        test_set = None
        dataset = None

        # Case 2: Some TVT sets are None.
        yaml_content = f"""
            tasks:
              - name: node_classification
                train_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {train_ids_path}
        """
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)
        assert dataset.tasks[0].train_set is not None
        assert dataset.tasks[0].validation_set is None
        assert dataset.tasks[0].test_set is None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pair_label():
    """Test TVTSet which returns ItemSet with IDs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_src = np.arange(1000)
        train_src_path = os.path.join(test_dir, "train_src.npy")
        np.save(train_src_path, train_src)
        train_dst = np.arange(1000, 2000)
        train_dst_path = os.path.join(test_dir, "train_dst.npy")
        np.save(train_dst_path, train_dst)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        validation_src = np.arange(1000, 2000)
        validation_src_path = os.path.join(test_dir, "validation_src.npy")
        np.save(validation_src_path, validation_src)
        validation_dst = np.arange(2000, 3000)
        validation_dst_path = os.path.join(test_dir, "validation_dst.npy")
        np.save(validation_dst_path, validation_dst)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_labels_path = os.path.join(test_dir, "validation_labels.npy")
        np.save(validation_labels_path, validation_labels)

        test_src = np.arange(2000, 3000)
        test_src_path = os.path.join(test_dir, "test_src.npy")
        np.save(test_src_path, test_src)
        test_dst = np.arange(3000, 4000)
        test_dst_path = os.path.join(test_dir, "test_dst.npy")
        np.save(test_dst_path, test_dst)
        test_labels = np.random.randint(0, 10, size=1000)
        test_labels_path = os.path.join(test_dir, "test_labels.npy")
        np.save(test_labels_path, test_labels)

        yaml_content = f"""
            tasks:
              - name: link_prediction
                train_set:
                  - type: null
                    data:
                      - format: numpy
                        in_memory: true
                        path: {train_src_path}
                      - format: numpy
                        in_memory: true
                        path: {train_dst_path}
                      - format: numpy
                        in_memory: true
                        path: {train_labels_path}
                validation_set:
                  - data:
                      - format: numpy
                        in_memory: true
                        path: {validation_src_path}
                      - format: numpy
                        in_memory: true
                        path: {validation_dst_path}
                      - format: numpy
                        in_memory: true
                        path: {validation_labels_path}
                test_set:
                  - type: null
                    data:
                      - format: numpy
                        in_memory: true
                        path: {test_src_path}
                      - format: numpy
                        in_memory: true
                        path: {test_dst_path}
                      - format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (src, dst, label) in enumerate(train_set):
            assert src == train_src[i]
            assert dst == train_dst[i]
            assert label == train_labels[i]
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (src, dst, label) in enumerate(validation_set):
            assert src == validation_src[i]
            assert dst == validation_dst[i]
            assert label == validation_labels[i]
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (src, dst, label) in enumerate(test_set):
            assert src == test_src[i]
            assert dst == test_dst[i]
            assert label == test_labels[i]
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pair_negs():
    """Test TVTSet which returns ItemSet with node pairs and negative ones."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_src = np.arange(1000)
        train_src_path = os.path.join(test_dir, "train_src.npy")
        np.save(train_src_path, train_src)
        train_dst = np.arange(1000, 2000)
        train_dst_path = os.path.join(test_dir, "train_dst.npy")
        np.save(train_dst_path, train_dst)
        train_neg_dst = np.random.choice(1000 * 10, size=1000 * 10).reshape(
            1000, 10
        )
        train_neg_dst_path = os.path.join(test_dir, "train_neg_dst.npy")
        np.save(train_neg_dst_path, train_neg_dst)

        validation_src = np.arange(1000, 2000)
        validation_src_path = os.path.join(test_dir, "validation_src.npy")
        np.save(validation_src_path, validation_src)
        validation_dst = np.arange(2000, 3000)
        validation_dst_path = os.path.join(test_dir, "validation_dst.npy")
        np.save(validation_dst_path, validation_dst)
        validation_neg_dst = train_neg_dst + 1
        validation_neg_dst_path = os.path.join(
            test_dir, "validation_neg_dst.npy"
        )
        np.save(validation_neg_dst_path, validation_neg_dst)

        test_src = np.arange(2000, 3000)
        test_src_path = os.path.join(test_dir, "test_src.npy")
        np.save(test_src_path, test_src)
        test_dst = np.arange(3000, 4000)
        test_dst_path = os.path.join(test_dir, "test_dst.npy")
        np.save(test_dst_path, test_dst)
        test_neg_dst = train_neg_dst + 2
        test_neg_dst_path = os.path.join(test_dir, "test_neg_dst.npy")
        np.save(test_neg_dst_path, test_neg_dst)

        yaml_content = f"""
            tasks:
              - name: link_prediction
                train_set:
                  - type: null
                    data:
                      - format: numpy
                        in_memory: true
                        path: {train_src_path}
                      - format: numpy
                        in_memory: true
                        path: {train_dst_path}
                      - format: numpy
                        in_memory: true
                        path: {train_neg_dst_path}
                validation_set:
                  - data:
                      - format: numpy
                        in_memory: true
                        path: {validation_src_path}
                      - format: numpy
                        in_memory: true
                        path: {validation_dst_path}
                      - format: numpy
                        in_memory: true
                        path: {validation_neg_dst_path}
                test_set:
                  - type: null
                    data:
                      - format: numpy
                        in_memory: true
                        path: {test_src_path}
                      - format: numpy
                        in_memory: true
                        path: {test_dst_path}
                      - format: numpy
                        in_memory: true
                        path: {test_neg_dst_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (src, dst, negs) in enumerate(train_set):
            assert src == train_src[i]
            assert dst == train_dst[i]
            assert torch.equal(negs, torch.from_numpy(train_neg_dst[i]))
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (src, dst, negs) in enumerate(validation_set):
            assert src == validation_src[i]
            assert dst == validation_dst[i]
            assert torch.equal(negs, torch.from_numpy(validation_neg_dst[i]))
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (src, dst, negs) in enumerate(test_set):
            assert src == test_src[i]
            assert dst == test_dst[i]
            assert torch.equal(negs, torch.from_numpy(test_neg_dst[i]))
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSetDict_id_label():
    """Test TVTSet which returns ItemSetDict with IDs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_ids = np.arange(1000)
        train_labels = np.random.randint(0, 10, size=1000)
        train_data = np.vstack([train_ids, train_labels]).T
        train_path = os.path.join(test_dir, "train.npy")
        np.save(train_path, train_data)

        validation_ids = np.arange(1000, 2000)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_data = np.vstack([validation_ids, validation_labels]).T
        validation_path = os.path.join(test_dir, "validation.npy")
        np.save(validation_path, validation_data)

        test_ids = np.arange(2000, 3000)
        test_labels = np.random.randint(0, 10, size=1000)
        test_data = np.vstack([test_ids, test_labels]).T
        test_path = os.path.join(test_dir, "test.npy")
        np.save(test_path, test_data)

        yaml_content = f"""
            tasks:
              - name: node_classification
                train_set:
                  - type: paper
                    data:
                      - format: numpy
                        in_memory: true
                        path: {train_path}
                  - type: author
                    data:
                      - format: numpy
                        path: {train_path}
                validation_set:
                  - type: paper
                    data:
                      - format: numpy
                        path: {validation_path}
                  - type: author
                    data:
                      - format: numpy
                        path: {validation_path}
                test_set:
                  - type: paper
                    data:
                      - format: numpy
                        in_memory: false
                        path: {test_path}
                  - type: author
                    data:
                      - format: numpy
                        path: {test_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 2000
        assert isinstance(train_set, gb.ItemSetDict)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            id, label = item[key]
            assert id == train_ids[i % 1000]
            assert label == train_labels[i % 1000]
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 2000
        assert isinstance(validation_set, gb.ItemSetDict)
        for i, item in enumerate(validation_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            id, label = item[key]
            assert id == validation_ids[i % 1000]
            assert label == validation_labels[i % 1000]
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 2000
        assert isinstance(test_set, gb.ItemSetDict)
        for i, item in enumerate(test_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            id, label = item[key]
            assert id == test_ids[i % 1000]
            assert label == test_labels[i % 1000]
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSetDict_node_pair_label():
    """Test TVTSet which returns ItemSetDict with node pairs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_pairs = (np.arange(1000), np.arange(1000, 2000))
        train_labels = np.random.randint(0, 10, size=1000)
        train_data = np.vstack([train_pairs, train_labels]).T
        train_path = os.path.join(test_dir, "train.npy")
        np.save(train_path, train_data)

        validation_pairs = (np.arange(1000, 2000), np.arange(2000, 3000))
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_data = np.vstack([validation_pairs, validation_labels]).T
        validation_path = os.path.join(test_dir, "validation.npy")
        np.save(validation_path, validation_data)

        test_pairs = (np.arange(2000, 3000), np.arange(3000, 4000))
        test_labels = np.random.randint(0, 10, size=1000)
        test_data = np.vstack([test_pairs, test_labels]).T
        test_path = os.path.join(test_dir, "test.npy")
        np.save(test_path, test_data)

        yaml_content = f"""
            tasks:
              - name: edge_classification
                train_set:
                  - type: paper
                    data:
                      - format: numpy
                        in_memory: true
                        path: {train_path}
                  - type: author
                    data:
                      - format: numpy
                        path: {train_path}
                validation_set:
                  - type: paper
                    data:
                      - format: numpy
                        path: {validation_path}
                  - type: author
                    data:
                      - format: numpy
                        path: {validation_path}
                test_set:
                  - type: paper
                    data:
                      - format: numpy
                        in_memory: false
                        path: {test_path}
                  - type: author
                    data:
                      - format: numpy
                        path: {test_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 2000
        assert isinstance(train_set, gb.ItemSetDict)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            src, dst, label = item[key]
            assert src == train_pairs[0][i % 1000]
            assert dst == train_pairs[1][i % 1000]
            assert label == train_labels[i % 1000]
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 2000
        assert isinstance(validation_set, gb.ItemSetDict)
        for i, item in enumerate(validation_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            src, dst, label = item[key]
            assert src == validation_pairs[0][i % 1000]
            assert dst == validation_pairs[1][i % 1000]
            assert label == validation_labels[i % 1000]
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 2000
        assert isinstance(test_set, gb.ItemSetDict)
        for i, item in enumerate(test_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            src, dst, label = item[key]
            assert src == test_pairs[0][i % 1000]
            assert dst == test_pairs[1][i % 1000]
            assert label == test_labels[i % 1000]
        test_set = None
        dataset = None


def test_OnDiskDataset_Feature_heterograph():
    """Test Feature storage."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Generate node data.
        node_data_paper = np.random.rand(1000, 10)
        node_data_paper_path = os.path.join(test_dir, "node_data_paper.npy")
        np.save(node_data_paper_path, node_data_paper)
        node_data_label = np.random.randint(0, 10, size=1000)
        node_data_label_path = os.path.join(test_dir, "node_data_label.npy")
        np.save(node_data_label_path, node_data_label)

        # Generate edge data.
        edge_data_writes = np.random.rand(1000, 10)
        edge_data_writes_path = os.path.join(test_dir, "edge_writes_paper.npy")
        np.save(edge_data_writes_path, edge_data_writes)
        edge_data_label = np.random.randint(0, 10, size=1000)
        edge_data_label_path = os.path.join(test_dir, "edge_data_label.npy")
        np.save(edge_data_label_path, edge_data_label)

        # Generate YAML.
        yaml_content = f"""
            feature_data:
              - domain: node
                type: paper
                name: feat
                format: numpy
                in_memory: false
                path: {node_data_paper_path}
              - domain: node
                type: paper
                name: label
                format: numpy
                in_memory: true
                path: {node_data_label_path}
              - domain: edge
                type: "author:writes:paper"
                name: feat
                format: numpy
                in_memory: false
                path: {edge_data_writes_path}
              - domain: edge
                type: "author:writes:paper"
                name: label
                format: numpy
                in_memory: true
                path: {edge_data_label_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify feature data storage.
        feature_data = dataset.feature
        assert len(feature_data) == 4

        # Verify node feature data.
        assert torch.equal(
            feature_data.read("node", "paper", "feat"),
            torch.tensor(node_data_paper),
        )
        assert torch.equal(
            feature_data.read("node", "paper", "label"),
            torch.tensor(node_data_label),
        )

        # Verify edge feature data.
        assert torch.equal(
            feature_data.read("edge", "author:writes:paper", "feat"),
            torch.tensor(edge_data_writes),
        )
        assert torch.equal(
            feature_data.read("edge", "author:writes:paper", "label"),
            torch.tensor(edge_data_label),
        )

        feature_data = None
        dataset = None


def test_OnDiskDataset_Feature_homograph():
    """Test Feature storage."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Generate node data.
        node_data_feat = np.random.rand(1000, 10)
        node_data_feat_path = os.path.join(test_dir, "node_data_feat.npy")
        np.save(node_data_feat_path, node_data_feat)
        node_data_label = np.random.randint(0, 10, size=1000)
        node_data_label_path = os.path.join(test_dir, "node_data_label.npy")
        np.save(node_data_label_path, node_data_label)

        # Generate edge data.
        edge_data_feat = np.random.rand(1000, 10)
        edge_data_feat_path = os.path.join(test_dir, "edge_data_feat.npy")
        np.save(edge_data_feat_path, edge_data_feat)
        edge_data_label = np.random.randint(0, 10, size=1000)
        edge_data_label_path = os.path.join(test_dir, "edge_data_label.npy")
        np.save(edge_data_label_path, edge_data_label)

        # Generate YAML.
        # ``type`` is not specified in the YAML.
        yaml_content = f"""
            feature_data:
              - domain: node
                name: feat
                format: numpy
                in_memory: false
                path: {node_data_feat_path}
              - domain: node
                name: label
                format: numpy
                in_memory: true
                path: {node_data_label_path}
              - domain: edge
                name: feat
                format: numpy
                in_memory: false
                path: {edge_data_feat_path}
              - domain: edge
                name: label
                format: numpy
                in_memory: true
                path: {edge_data_label_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)

        # Verify feature data storage.
        feature_data = dataset.feature
        assert len(feature_data) == 4

        # Verify node feature data.
        assert torch.equal(
            feature_data.read("node", None, "feat"),
            torch.tensor(node_data_feat),
        )
        assert torch.equal(
            feature_data.read("node", None, "label"),
            torch.tensor(node_data_label),
        )

        # Verify edge feature data.
        assert torch.equal(
            feature_data.read("edge", None, "feat"),
            torch.tensor(edge_data_feat),
        )
        assert torch.equal(
            feature_data.read("edge", None, "label"),
            torch.tensor(edge_data_label),
        )

        feature_data = None
        dataset = None


def test_OnDiskDataset_Graph_Exceptions():
    """Test exceptions in parsing graph topology."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Invalid graph type.
        yaml_content = """
            graph_topology:
              type: CSRSamplingGraph
              path: /path/to/graph
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        with pytest.raises(
            pydantic.ValidationError,
            match="1 validation error for OnDiskMetaData",
        ):
            _ = gb.OnDiskDataset(test_dir)


def test_OnDiskDataset_Graph_homogeneous():
    """Test homogeneous graph topology."""
    csc_indptr, indices = gbt.random_homo_graph(1000, 10 * 1000)
    graph = gb.from_csc(csc_indptr, indices)

    with tempfile.TemporaryDirectory() as test_dir:
        graph_path = os.path.join(test_dir, "csc_sampling_graph.tar")
        gb.save_csc_sampling_graph(graph, graph_path)

        yaml_content = f"""
            graph_topology:
              type: CSCSamplingGraph
              path: {graph_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)
        graph2 = dataset.graph

        assert graph.num_nodes == graph2.num_nodes
        assert graph.num_edges == graph2.num_edges

        assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
        assert torch.equal(graph.indices, graph2.indices)

        assert graph.metadata is None and graph2.metadata is None
        assert (
            graph.node_type_offset is None and graph2.node_type_offset is None
        )
        assert graph.type_per_edge is None and graph2.type_per_edge is None


def test_OnDiskDataset_Graph_heterogeneous():
    """Test heterogeneous graph topology."""
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    ) = gbt.random_hetero_graph(1000, 10 * 1000, 3, 4)
    graph = gb.from_csc(
        csc_indptr, indices, node_type_offset, type_per_edge, None, metadata
    )

    with tempfile.TemporaryDirectory() as test_dir:
        graph_path = os.path.join(test_dir, "csc_sampling_graph.tar")
        gb.save_csc_sampling_graph(graph, graph_path)

        yaml_content = f"""
            graph_topology:
              type: CSCSamplingGraph
              path: {graph_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)
        graph2 = dataset.graph

        assert graph.num_nodes == graph2.num_nodes
        assert graph.num_edges == graph2.num_edges

        assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
        assert torch.equal(graph.indices, graph2.indices)
        assert torch.equal(graph.node_type_offset, graph2.node_type_offset)
        assert torch.equal(graph.type_per_edge, graph2.type_per_edge)
        assert graph.metadata.node_type_to_id == graph2.metadata.node_type_to_id
        assert graph.metadata.edge_type_to_id == graph2.metadata.edge_type_to_id


def test_OnDiskDataset_Metadata():
    """Test metadata of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)
        assert dataset.dataset_name == dataset_name

        # Only dataset_name is specified.
        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir)
        assert dataset.dataset_name == dataset_name


def test_OnDiskDataset_preprocess_homogeneous():
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random edges.
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors], axis=1)
        # Wrtie into edges/edge.csv
        os.makedirs(os.path.join(test_dir, "edges/"), exist_ok=True)
        edges = pd.DataFrame(edges, columns=["src", "dst"])
        edges.to_csv(
            os.path.join(test_dir, "edges/edge.csv"),
            index=False,
            header=False,
        )

        # Generate random graph edge-feats.
        edge_feats = np.random.rand(num_edges, 5)
        os.makedirs(os.path.join(test_dir, "data/"), exist_ok=True)
        np.save(os.path.join(test_dir, "data/edge-feat.npy"), edge_feats)

        # Generate random node-feats.
        node_feats = np.random.rand(num_nodes, 10)
        np.save(os.path.join(test_dir, "data/node-feat.npy"), node_feats)

        # Generate train/test/valid set.
        os.makedirs(os.path.join(test_dir, "set/"), exist_ok=True)
        train_pairs = (np.arange(1000), np.arange(1000, 2000))
        train_labels = np.random.randint(0, 10, size=1000)
        train_data = np.vstack([train_pairs, train_labels]).T
        train_path = os.path.join(test_dir, "set/train.npy")
        np.save(train_path, train_data)

        validation_pairs = (np.arange(1000, 2000), np.arange(2000, 3000))
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_data = np.vstack([validation_pairs, validation_labels]).T
        validation_path = os.path.join(test_dir, "set/validation.npy")
        np.save(validation_path, validation_data)

        test_pairs = (np.arange(2000, 3000), np.arange(3000, 4000))
        test_labels = np.random.randint(0, 10, size=1000)
        test_data = np.vstack([test_pairs, test_labels]).T
        test_path = os.path.join(test_dir, "set/test.npy")
        np.save(test_path, test_data)

        yaml_content = f"""
            dataset_name: {dataset_name}
            graph: # graph structure and required attributes.
                nodes:
                    - num: {num_nodes}
                edges:
                    - format: csv
                      path: edges/edge.csv
                feature_data:
                    - domain: edge
                      type: null
                      name: feat
                      format: numpy
                      in_memory: true
                      path: data/edge-feat.npy
            feature_data:
                - domain: node
                  type: null
                  name: feat
                  format: numpy
                  in_memory: false
                  path: data/node-feat.npy
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set/train.npy
                validation_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set/validation.npy
                test_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set/test.npy
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        output_file = gb.ondisk_dataset.preprocess_ondisk_dataset(test_dir)

        with open(output_file, "rb") as f:
            processed_dataset = yaml.load(f, Loader=yaml.Loader)

        assert processed_dataset["dataset_name"] == dataset_name
        assert processed_dataset["tasks"][0]["num_classes"] == num_classes
        assert "graph" not in processed_dataset
        assert "graph_topology" in processed_dataset

        csc_sampling_graph = gb.csc_sampling_graph.load_csc_sampling_graph(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert csc_sampling_graph.num_nodes == num_nodes
        assert csc_sampling_graph.num_edges == num_edges

        num_samples = 100
        fanout = 1
        subgraph = csc_sampling_graph.sample_neighbors(
            torch.arange(num_samples),
            torch.tensor([fanout]),
        )
        assert len(list(subgraph.node_pairs.values())[0][0]) <= num_samples


def test_OnDiskDataset_preprocess_path():
    """Test if the preprocess function can catch the path error."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"

        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Case1. Test the passed in is the yaml file path.
        with pytest.raises(
            RuntimeError,
            match="The dataset must be a directory. "
            rf"But got {re.escape(yaml_file)}",
        ):
            _ = gb.OnDiskDataset(yaml_file)

        # Case2. Test the passed in is a fake directory.
        fake_dir = os.path.join(test_dir, "fake_dir")
        with pytest.raises(
            RuntimeError,
            match=rf"Invalid dataset path: {re.escape(fake_dir)}",
        ):
            _ = gb.OnDiskDataset(fake_dir)

        # Case3. Test the passed in is the dataset directory.
        # But the metadata.yaml is not in the directory.
        os.makedirs(os.path.join(test_dir, "fake_dir"), exist_ok=True)
        with pytest.raises(
            RuntimeError,
            match=r"metadata.yaml does not exist.",
        ):
            _ = gb.OnDiskDataset(fake_dir)


@unittest.skipIf(os.name == "nt", "Skip on Windows")
def test_OnDiskDataset_preprocess_yaml_content_unix():
    """Test if the preprocessed metadata.yaml is correct."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random edges.
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors], axis=1)
        # Wrtie into edges/edge.csv
        os.makedirs(os.path.join(test_dir, "edges/"), exist_ok=True)
        edges = pd.DataFrame(edges, columns=["src", "dst"])
        edges.to_csv(
            os.path.join(test_dir, "edges/edge.csv"),
            index=False,
            header=False,
        )

        # Generate random graph edge-feats.
        edge_feats = np.random.rand(num_edges, 5)
        os.makedirs(os.path.join(test_dir, "data/"), exist_ok=True)
        np.save(os.path.join(test_dir, "data/edge-feat.npy"), edge_feats)

        # Generate random node-feats.
        node_feats = np.random.rand(num_nodes, 10)
        np.save(os.path.join(test_dir, "data/node-feat.npy"), node_feats)

        # Generate train/test/valid set.
        os.makedirs(os.path.join(test_dir, "set/"), exist_ok=True)
        train_pairs = (np.arange(1000), np.arange(1000, 2000))
        train_labels = np.random.randint(0, 10, size=1000)
        train_data = np.vstack([train_pairs, train_labels]).T
        train_path = os.path.join(test_dir, "set/train.npy")
        np.save(train_path, train_data)

        validation_pairs = (np.arange(1000, 2000), np.arange(2000, 3000))
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_data = np.vstack([validation_pairs, validation_labels]).T
        validation_path = os.path.join(test_dir, "set/validation.npy")
        np.save(validation_path, validation_data)

        test_pairs = (np.arange(2000, 3000), np.arange(3000, 4000))
        test_labels = np.random.randint(0, 10, size=1000)
        test_data = np.vstack([test_pairs, test_labels]).T
        test_path = os.path.join(test_dir, "set/test.npy")
        np.save(test_path, test_data)

        yaml_content = f"""
            dataset_name: {dataset_name}
            graph: # graph structure and required attributes.
                nodes:
                    - num: {num_nodes}
                edges:
                    - format: csv
                      path: edges/edge.csv
                feature_data:
                    - domain: edge
                      type: null
                      name: feat
                      format: numpy
                      in_memory: true
                      path: data/edge-feat.npy
            feature_data:
                - domain: node
                  type: null
                  name: feat
                  format: numpy
                  in_memory: false
                  path: data/node-feat.npy
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set/train.npy
                validation_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set/validation.npy
                test_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set/test.npy
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        preprocessed_metadata_path = gb.preprocess_ondisk_dataset(test_dir)
        with open(preprocessed_metadata_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        target_yaml_content = f"""
            dataset_name: {dataset_name}
            graph_topology:
              type: CSCSamplingGraph
              path: preprocessed/csc_sampling_graph.tar
            feature_data:
              - domain: node
                type: null
                name: feat
                format: numpy
                in_memory: false
                path: preprocessed/data/node-feat.npy
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: preprocessed/set/train.npy
                validation_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: preprocessed/set/validation.npy
                test_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: preprocessed/set/test.npy
        """
        target_yaml_data = yaml.safe_load(target_yaml_content)
        # Check yaml content.
        assert (
            yaml_data == target_yaml_data
        ), "The preprocessed metadata.yaml is not correct."

        # Check file existence.
        assert os.path.exists(
            os.path.join(test_dir, yaml_data["graph_topology"]["path"])
        )
        assert os.path.exists(
            os.path.join(test_dir, yaml_data["feature_data"][0]["path"])
        )
        for set_name in ["train_set", "validation_set", "test_set"]:
            assert os.path.exists(
                os.path.join(
                    test_dir,
                    yaml_data["tasks"][0][set_name][0]["data"][0]["path"],
                )
            )


@unittest.skipIf(os.name != "nt", "Skip on Unix")
def test_OnDiskDataset_preprocess_yaml_content_windows():
    """Test if the preprocessed metadata.yaml is correct."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random edges.
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors], axis=1)
        # Wrtie into edges/edge.csv
        os.makedirs(os.path.join(test_dir, "edges\\"), exist_ok=True)
        edges = pd.DataFrame(edges, columns=["src", "dst"])
        edges.to_csv(
            os.path.join(test_dir, "edges\\edge.csv"),
            index=False,
            header=False,
        )

        # Generate random graph edge-feats.
        edge_feats = np.random.rand(num_edges, 5)
        os.makedirs(os.path.join(test_dir, "data\\"), exist_ok=True)
        np.save(os.path.join(test_dir, "data\\edge-feat.npy"), edge_feats)

        # Generate random node-feats.
        node_feats = np.random.rand(num_nodes, 10)
        np.save(os.path.join(test_dir, "data\\node-feat.npy"), node_feats)

        # Generate train/test/valid set.
        os.makedirs(os.path.join(test_dir, "set\\"), exist_ok=True)
        train_pairs = (np.arange(1000), np.arange(1000, 2000))
        train_labels = np.random.randint(0, 10, size=1000)
        train_data = np.vstack([train_pairs, train_labels]).T
        train_path = os.path.join(test_dir, "set\\train.npy")
        np.save(train_path, train_data)

        validation_pairs = (np.arange(1000, 2000), np.arange(2000, 3000))
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_data = np.vstack([validation_pairs, validation_labels]).T
        validation_path = os.path.join(test_dir, "set\\validation.npy")
        np.save(validation_path, validation_data)

        test_pairs = (np.arange(2000, 3000), np.arange(3000, 4000))
        test_labels = np.random.randint(0, 10, size=1000)
        test_data = np.vstack([test_pairs, test_labels]).T
        test_path = os.path.join(test_dir, "set\\test.npy")
        np.save(test_path, test_data)

        yaml_content = f"""
            dataset_name: {dataset_name}
            graph: # graph structure and required attributes.
                nodes:
                    - num: {num_nodes}
                edges:
                    - format: csv
                      path: edges\\edge.csv
                feature_data:
                    - domain: edge
                      type: null
                      name: feat
                      format: numpy
                      in_memory: true
                      path: data\\edge-feat.npy
            feature_data:
                - domain: node
                  type: null
                  name: feat
                  format: numpy
                  in_memory: false
                  path: data\\node-feat.npy
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set\\train.npy
                validation_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set\\validation.npy
                test_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: set\\test.npy
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        preprocessed_metadata_path = gb.preprocess_ondisk_dataset(test_dir)
        with open(preprocessed_metadata_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        target_yaml_content = f"""
            dataset_name: {dataset_name}
            graph_topology:
              type: CSCSamplingGraph
              path: preprocessed\\csc_sampling_graph.tar
            feature_data:
              - domain: node
                type: null
                name: feat
                format: numpy
                in_memory: false
                path: preprocessed\\data\\node-feat.npy
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: preprocessed\\set\\train.npy
                validation_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: preprocessed\\set\\validation.npy
                test_set:
                  - type_name: null
                    data:
                      - format: numpy
                        path: preprocessed\\set\\test.npy
        """
        target_yaml_data = yaml.safe_load(target_yaml_content)
        # Check yaml content.
        assert (
            yaml_data == target_yaml_data
        ), "The preprocessed metadata.yaml is not correct."

        # Check file existence.
        assert os.path.exists(
            os.path.join(test_dir, yaml_data["graph_topology"]["path"])
        )
        assert os.path.exists(
            os.path.join(test_dir, yaml_data["feature_data"][0]["path"])
        )
        for set_name in ["train_set", "validation_set", "test_set"]:
            assert os.path.exists(
                os.path.join(
                    test_dir,
                    yaml_data["tasks"][0][set_name][0]["data"][0]["path"],
                )
            )
