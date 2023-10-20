import os
import pickle
import random
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
            _ = gb.OnDiskDataset(test_dir).load()

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
            _ = gb.OnDiskDataset(test_dir).load()


def test_OnDiskDataset_TVTSet_ItemSet_names():
    """Test TVTSet which returns ItemSet with IDs, labels and corresponding names."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_ids = np.arange(1000)
        train_ids_path = os.path.join(test_dir, "train_ids.npy")
        np.save(train_ids_path, train_ids)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        yaml_content = f"""
            tasks:
              - name: node_classification
                num_classes: 10
                train_set:
                  - type: null
                    data:
                      - name: seed_nodes
                        format: numpy
                        in_memory: true
                        path: {train_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                      - format: numpy
                        in_memory: true
                        path: {train_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (id, label, _) in enumerate(train_set):
            assert id == train_ids[i]
            assert label == train_labels[i]
        assert train_set.names == ("seed_nodes", "labels", None)
        train_set = None


def test_OnDiskDataset_TVTSet_ItemSetDict_names():
    """Test TVTSet which returns ItemSet with IDs, labels and corresponding names."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_ids = np.arange(1000)
        train_ids_path = os.path.join(test_dir, "train_ids.npy")
        np.save(train_ids_path, train_ids)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        yaml_content = f"""
            tasks:
              - name: node_classification
                num_classes: 10
                train_set:
                  - type: "author:writes:paper"
                    data:
                      - name: seed_nodes
                        format: numpy
                        in_memory: true
                        path: {train_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                      - format: numpy
                        in_memory: true
                        path: {train_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSetDict)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert "author:writes:paper" in item
            id, label, _ = item["author:writes:paper"]
            assert id == train_ids[i]
            assert label == train_labels[i]
        assert train_set.names == ("seed_nodes", "labels", None)
        train_set = None


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
                      - name: seed_nodes
                        format: numpy
                        in_memory: true
                        path: {train_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                validation_set:
                  - data:
                      - name: seed_nodes
                        format: numpy
                        in_memory: true
                        path: {validation_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {validation_labels_path}
                test_set:
                  - type: null
                    data:
                      - name: seed_nodes
                        format: numpy
                        in_memory: true
                        path: {test_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

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
        assert train_set.names == ("seed_nodes", "labels")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (id, label) in enumerate(validation_set):
            assert id == validation_ids[i]
            assert label == validation_labels[i]
        assert validation_set.names == ("seed_nodes", "labels")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (id, label) in enumerate(test_set):
            assert id == test_ids[i]
            assert label == test_labels[i]
        assert test_set.names == ("seed_nodes", "labels")
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

        dataset = gb.OnDiskDataset(test_dir).load()
        assert dataset.tasks[0].train_set is not None
        assert dataset.tasks[0].validation_set is None
        assert dataset.tasks[0].test_set is None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pairs_labels():
    """Test TVTSet which returns ItemSet with node pairs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_node_pairs = np.arange(2000).reshape(1000, 2)
        train_node_pairs_path = os.path.join(test_dir, "train_node_pairs.npy")
        np.save(train_node_pairs_path, train_node_pairs)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        validation_node_pairs = np.arange(2000, 4000).reshape(1000, 2)
        validation_node_pairs_path = os.path.join(
            test_dir, "validation_node_pairs.npy"
        )
        np.save(validation_node_pairs_path, validation_node_pairs)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_labels_path = os.path.join(test_dir, "validation_labels.npy")
        np.save(validation_labels_path, validation_labels)

        test_node_pairs = np.arange(4000, 6000).reshape(1000, 2)
        test_node_pairs_path = os.path.join(test_dir, "test_node_pairs.npy")
        np.save(test_node_pairs_path, test_node_pairs)
        test_labels = np.random.randint(0, 10, size=1000)
        test_labels_path = os.path.join(test_dir, "test_labels.npy")
        np.save(test_labels_path, test_labels)

        yaml_content = f"""
            tasks:
              - name: link_prediction
                train_set:
                  - type: null
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {train_node_pairs_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                validation_set:
                  - data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {validation_node_pairs_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {validation_labels_path}
                test_set:
                  - type: null
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {test_node_pairs_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (node_pair, label) in enumerate(train_set):
            assert node_pair[0] == train_node_pairs[i][0]
            assert node_pair[1] == train_node_pairs[i][1]
            assert label == train_labels[i]
        assert train_set.names == ("node_pairs", "labels")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (node_pair, label) in enumerate(validation_set):
            assert node_pair[0] == validation_node_pairs[i][0]
            assert node_pair[1] == validation_node_pairs[i][1]
            assert label == validation_labels[i]
        assert validation_set.names == ("node_pairs", "labels")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (node_pair, label) in enumerate(test_set):
            assert node_pair[0] == test_node_pairs[i][0]
            assert node_pair[1] == test_node_pairs[i][1]
            assert label == test_labels[i]
        assert test_set.names == ("node_pairs", "labels")
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pairs_negs():
    """Test TVTSet which returns ItemSet with node pairs and negative ones."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_node_pairs = np.arange(2000).reshape(1000, 2)
        train_node_pairs_path = os.path.join(test_dir, "train_node_pairs.npy")
        np.save(train_node_pairs_path, train_node_pairs)
        train_neg_dst = np.random.choice(1000 * 10, size=1000 * 10).reshape(
            1000, 10
        )
        train_neg_dst_path = os.path.join(test_dir, "train_neg_dst.npy")
        np.save(train_neg_dst_path, train_neg_dst)

        validation_node_pairs = np.arange(2000, 4000).reshape(1000, 2)
        validation_node_pairs_path = os.path.join(
            test_dir, "validation_node_pairs.npy"
        )
        np.save(validation_node_pairs_path, validation_node_pairs)
        validation_neg_dst = train_neg_dst + 1
        validation_neg_dst_path = os.path.join(
            test_dir, "validation_neg_dst.npy"
        )
        np.save(validation_neg_dst_path, validation_neg_dst)

        test_node_pairs = np.arange(4000, 6000).reshape(1000, 2)
        test_node_pairs_path = os.path.join(test_dir, "test_node_pairs.npy")
        np.save(test_node_pairs_path, test_node_pairs)
        test_neg_dst = train_neg_dst + 2
        test_neg_dst_path = os.path.join(test_dir, "test_neg_dst.npy")
        np.save(test_neg_dst_path, test_neg_dst)

        yaml_content = f"""
            tasks:
              - name: link_prediction
                train_set:
                  - type: null
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {train_node_pairs_path}
                      - name: negative_dsts
                        format: numpy
                        in_memory: true
                        path: {train_neg_dst_path}
                validation_set:
                  - data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {validation_node_pairs_path}
                      - name: negative_dsts
                        format: numpy
                        in_memory: true
                        path: {validation_neg_dst_path}
                test_set:
                  - type: null
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {test_node_pairs_path}
                      - name: negative_dsts
                        format: numpy
                        in_memory: true
                        path: {test_neg_dst_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (node_pair, negs) in enumerate(train_set):
            assert node_pair[0] == train_node_pairs[i][0]
            assert node_pair[1] == train_node_pairs[i][1]
            assert torch.equal(negs, torch.from_numpy(train_neg_dst[i]))
        assert train_set.names == ("node_pairs", "negative_dsts")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (node_pair, negs) in enumerate(validation_set):
            assert node_pair[0] == validation_node_pairs[i][0]
            assert node_pair[1] == validation_node_pairs[i][1]
            assert torch.equal(negs, torch.from_numpy(validation_neg_dst[i]))
        assert validation_set.names == ("node_pairs", "negative_dsts")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (node_pair, negs) in enumerate(test_set):
            assert node_pair[0] == test_node_pairs[i][0]
            assert node_pair[1] == test_node_pairs[i][1]
            assert torch.equal(negs, torch.from_numpy(test_neg_dst[i]))
        assert test_set.names == ("node_pairs", "negative_dsts")
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
                      - name: seed_nodes
                        format: numpy
                        in_memory: true
                        path: {train_path}
                  - type: author
                    data:
                      - name: seed_nodes
                        format: numpy
                        path: {train_path}
                validation_set:
                  - type: paper
                    data:
                      - name: seed_nodes
                        format: numpy
                        path: {validation_path}
                  - type: author
                    data:
                      - name: seed_nodes
                        format: numpy
                        path: {validation_path}
                test_set:
                  - type: paper
                    data:
                      - name: seed_nodes
                        format: numpy
                        in_memory: false
                        path: {test_path}
                  - type: author
                    data:
                      - name: seed_nodes
                        format: numpy
                        path: {test_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

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
        assert train_set.names == ("seed_nodes",)
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
        assert validation_set.names == ("seed_nodes",)
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
        assert test_set.names == ("seed_nodes",)
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSetDict_node_pairs_labels():
    """Test TVTSet which returns ItemSetDict with node pairs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_node_pairs = np.arange(2000).reshape(1000, 2)
        train_node_pairs_path = os.path.join(test_dir, "train_node_pairs.npy")
        np.save(train_node_pairs_path, train_node_pairs)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        validation_node_pairs = np.arange(2000, 4000).reshape(1000, 2)
        validation_node_pairs_path = os.path.join(
            test_dir, "validation_node_pairs.npy"
        )
        np.save(validation_node_pairs_path, validation_node_pairs)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_labels_path = os.path.join(test_dir, "validation_labels.npy")
        np.save(validation_labels_path, validation_labels)

        test_node_pairs = np.arange(4000, 6000).reshape(1000, 2)
        test_node_pairs_path = os.path.join(test_dir, "test_node_pairs.npy")
        np.save(test_node_pairs_path, test_node_pairs)
        test_labels = np.random.randint(0, 10, size=1000)
        test_labels_path = os.path.join(test_dir, "test_labels.npy")
        np.save(test_labels_path, test_labels)

        yaml_content = f"""
            tasks:
              - name: edge_classification
                train_set:
                  - type: paper:cites:paper
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {train_node_pairs_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                  - type: author:writes:paper
                    data:
                      - name: node_pairs
                        format: numpy
                        path: {train_node_pairs_path}
                      - name: labels
                        format: numpy
                        path: {train_labels_path}
                validation_set:
                  - type: paper:cites:paper
                    data:
                      - name: node_pairs
                        format: numpy
                        path: {validation_node_pairs_path}
                      - name: labels
                        format: numpy
                        path: {validation_labels_path}
                  - type: author:writes:paper
                    data:
                      - name: node_pairs
                        format: numpy
                        path: {validation_node_pairs_path}
                      - name: labels
                        format: numpy
                        path: {validation_labels_path}
                test_set:
                  - type: paper:cites:paper
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {test_node_pairs_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
                  - type: author:writes:paper
                    data:
                      - name: node_pairs
                        format: numpy
                        in_memory: true
                        path: {test_node_pairs_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 2000
        assert isinstance(train_set, gb.ItemSetDict)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper:cites:paper", "author:writes:paper"]
            node_pair, label = item[key]
            assert node_pair[0] == train_node_pairs[i % 1000][0]
            assert node_pair[1] == train_node_pairs[i % 1000][1]
            assert label == train_labels[i % 1000]
        assert train_set.names == ("node_pairs", "labels")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 2000
        assert isinstance(validation_set, gb.ItemSetDict)
        for i, item in enumerate(validation_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper:cites:paper", "author:writes:paper"]
            node_pair, label = item[key]
            assert node_pair[0] == validation_node_pairs[i % 1000][0]
            assert node_pair[1] == validation_node_pairs[i % 1000][1]
            assert label == validation_labels[i % 1000]
        assert validation_set.names == ("node_pairs", "labels")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 2000
        assert isinstance(test_set, gb.ItemSetDict)
        for i, item in enumerate(test_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper:cites:paper", "author:writes:paper"]
            node_pair, label = item[key]
            assert node_pair[0] == test_node_pairs[i % 1000][0]
            assert node_pair[1] == test_node_pairs[i % 1000][1]
            assert label == test_labels[i % 1000]
        assert test_set.names == ("node_pairs", "labels")
        test_set = None
        dataset = None


def test_OnDiskDataset_Feature_heterograph():
    """Test Feature storage."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Generate node data.
        node_data_paper = np.random.rand(1000, 10)
        node_data_paper_path = os.path.join(test_dir, "node_data_paper.npy")
        np.save(node_data_paper_path, node_data_paper)
        node_data_label = torch.tensor(
            [[random.randint(0, 10)] for _ in range(1000)]
        )
        node_data_label_path = os.path.join(test_dir, "node_data_label.npy")
        np.save(node_data_label_path, node_data_label)

        # Generate edge data.
        edge_data_writes = np.random.rand(1000, 10)
        edge_data_writes_path = os.path.join(test_dir, "edge_writes_paper.npy")
        np.save(edge_data_writes_path, edge_data_writes)
        edge_data_label = torch.tensor(
            [[random.randint(0, 10)] for _ in range(1000)]
        )
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
                name: labels
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
                name: labels
                format: numpy
                in_memory: true
                path: {edge_data_label_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify feature data storage.
        feature_data = dataset.feature
        assert len(feature_data) == 4

        # Verify node feature data.
        assert torch.equal(
            feature_data.read("node", "paper", "feat"),
            torch.tensor(node_data_paper),
        )
        assert torch.equal(
            feature_data.read("node", "paper", "labels"),
            torch.tensor(node_data_label),
        )

        # Verify edge feature data.
        assert torch.equal(
            feature_data.read("edge", "author:writes:paper", "feat"),
            torch.tensor(edge_data_writes),
        )
        assert torch.equal(
            feature_data.read("edge", "author:writes:paper", "labels"),
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
        node_data_label = torch.tensor(
            [[random.randint(0, 10)] for _ in range(1000)]
        )
        node_data_label_path = os.path.join(test_dir, "node_data_label.npy")
        np.save(node_data_label_path, node_data_label)

        # Generate edge data.
        edge_data_feat = np.random.rand(1000, 10)
        edge_data_feat_path = os.path.join(test_dir, "edge_data_feat.npy")
        np.save(edge_data_feat_path, edge_data_feat)
        edge_data_label = torch.tensor(
            [[random.randint(0, 10)] for _ in range(1000)]
        )
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
                name: labels
                format: numpy
                in_memory: true
                path: {node_data_label_path}
              - domain: edge
                name: feat
                format: numpy
                in_memory: false
                path: {edge_data_feat_path}
              - domain: edge
                name: labels
                format: numpy
                in_memory: true
                path: {edge_data_label_path}
        """
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()

        # Verify feature data storage.
        feature_data = dataset.feature
        assert len(feature_data) == 4

        # Verify node feature data.
        assert torch.equal(
            feature_data.read("node", None, "feat"),
            torch.tensor(node_data_feat),
        )
        assert torch.equal(
            feature_data.read("node", None, "labels"),
            torch.tensor(node_data_label),
        )

        # Verify edge feature data.
        assert torch.equal(
            feature_data.read("edge", None, "feat"),
            torch.tensor(edge_data_feat),
        )
        assert torch.equal(
            feature_data.read("edge", None, "labels"),
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
            _ = gb.OnDiskDataset(test_dir).load()


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

        dataset = gb.OnDiskDataset(test_dir).load()
        graph2 = dataset.graph

        assert graph.total_num_nodes == graph2.total_num_nodes
        assert graph.total_num_edges == graph2.total_num_edges

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

        dataset = gb.OnDiskDataset(test_dir).load()
        graph2 = dataset.graph

        assert graph.total_num_nodes == graph2.total_num_nodes
        assert graph.total_num_edges == graph2.total_num_edges

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

        dataset = gb.OnDiskDataset(test_dir).load()
        assert dataset.dataset_name == dataset_name

        # Only dataset_name is specified.
        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        yaml_file = os.path.join(test_dir, "preprocessed/metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()
        assert dataset.dataset_name == dataset_name


def test_OnDiskDataset_preprocess_homogeneous():
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        output_file = gb.ondisk_dataset.preprocess_ondisk_dataset(
            test_dir, include_original_edge_id=False
        )

        with open(output_file, "rb") as f:
            processed_dataset = yaml.load(f, Loader=yaml.Loader)

        assert processed_dataset["dataset_name"] == dataset_name
        assert processed_dataset["tasks"][0]["num_classes"] == num_classes
        assert "graph" not in processed_dataset
        assert "graph_topology" in processed_dataset

        csc_sampling_graph = gb.csc_sampling_graph.load_csc_sampling_graph(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert csc_sampling_graph.total_num_nodes == num_nodes
        assert csc_sampling_graph.total_num_edges == num_edges
        assert (
            csc_sampling_graph.edge_attributes is None
            or gb.ORIGINAL_EDGE_ID not in csc_sampling_graph.edge_attributes
        )

        num_samples = 100
        fanout = 1
        subgraph = csc_sampling_graph.sample_neighbors(
            torch.arange(num_samples),
            torch.tensor([fanout]),
        )
        assert len(subgraph.node_pairs[0]) <= num_samples

    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        # Test do not generate original_edge_id.
        output_file = gb.ondisk_dataset.preprocess_ondisk_dataset(
            test_dir, include_original_edge_id=False
        )
        with open(output_file, "rb") as f:
            processed_dataset = yaml.load(f, Loader=yaml.Loader)
        csc_sampling_graph = gb.csc_sampling_graph.load_csc_sampling_graph(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert (
            csc_sampling_graph.edge_attributes is not None
            and gb.ORIGINAL_EDGE_ID not in csc_sampling_graph.edge_attributes
        )
        csc_sampling_graph = None


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


def test_OnDiskDataset_load_name():
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Check modify `dataset_name` field.
        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["dataset_name"] = "fake_name"
        dataset.load()
        assert dataset.dataset_name == "fake_name"
        dataset = None


def test_OnDiskDataset_load_feature():
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Case1. Test modify the `in_memory` field.
        dataset = gb.OnDiskDataset(test_dir).load()
        original_feature_data = dataset.feature
        dataset.yaml_data["feature_data"][0]["in_memory"] = True
        dataset.load()
        modify_feature_data = dataset.feature
        # After modify the `in_memory` field, the feature data should be
        # equal.
        assert torch.equal(
            original_feature_data.read("node", None, "feat"),
            modify_feature_data.read("node", None, "feat"),
        )

        # Case2. Test modify the `format` field.
        dataset = gb.OnDiskDataset(test_dir)
        # If `format` is torch and `in_memory` is False, it will
        # raise an AssertionError.
        dataset.yaml_data["feature_data"][0]["format"] = "torch"
        with pytest.raises(
            AssertionError,
            match="^Pytorch tensor can only be loaded in memory,",
        ):
            dataset.load()

        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["feature_data"][0]["in_memory"] = True
        dataset.yaml_data["feature_data"][0]["format"] = "torch"
        # If `format` is torch and `in_memory` is True, it will
        # raise an UnpicklingError.
        with pytest.raises(pickle.UnpicklingError):
            dataset.load()

        # Case3. Test modify the `path` field.
        dataset = gb.OnDiskDataset(test_dir)
        # Use invalid path will raise an FileNotFoundError.
        dataset.yaml_data["feature_data"][0]["path"] = "fake_path"
        with pytest.raises(
            FileNotFoundError,
            match=r"\[Errno 2\] No such file or directory:",
        ):
            dataset.load()
        # Modifying the `path` field to an absolute path should work.
        # In os.path.join, if a segment is an absolute path (which
        # on Windows requires both a drive and a root), then all
        # previous segments are ignored and joining continues from
        # the absolute path segment.
        dataset = gb.OnDiskDataset(test_dir).load()
        original_feature_data = dataset.feature
        dataset.yaml_data["feature_data"][0]["path"] = os.path.join(
            test_dir, dataset.yaml_data["feature_data"][0]["path"]
        )
        dataset.load()
        modify_feature_data = dataset.feature
        assert torch.equal(
            original_feature_data.read("node", None, "feat"),
            modify_feature_data.read("node", None, "feat"),
        )
        original_feature_data = None
        modify_feature_data = None
        dataset = None


def test_OnDiskDataset_load_graph():
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Check the different original_edge_id option to load edge_attributes.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=True
        ).load()
        assert (
            dataset.graph.edge_attributes is not None
            and gb.ORIGINAL_EDGE_ID in dataset.graph.edge_attributes
        )

        # Case1. Test modify the `type` field.
        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["graph_topology"]["type"] = "fake_type"
        with pytest.raises(
            pydantic.ValidationError,
            # As error message diffs in pydantic 1.x and 2.x, we just match
            # keyword only.
            match="'CSCSamplingGraph'",
        ):
            dataset.load()

        # Case2. Test modify the `path` field.
        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["graph_topology"]["path"] = "fake_path"
        with pytest.raises(
            FileNotFoundError,
            match=r"\[Errno 2\] No such file or directory:",
        ):
            dataset.load()
        # Modifying the `path` field to an absolute path should work.
        # In os.path.join, if a segment is an absolute path (which
        # on Windows requires both a drive and a root), then all
        # previous segments are ignored and joining continues from
        # the absolute path segment.
        dataset = gb.OnDiskDataset(test_dir).load()
        original_graph = dataset.graph
        dataset.yaml_data["graph_topology"]["path"] = os.path.join(
            test_dir, dataset.yaml_data["graph_topology"]["path"]
        )
        dataset.load()
        modify_graph = dataset.graph
        assert torch.equal(
            original_graph.csc_indptr,
            modify_graph.csc_indptr,
        )
        original_graph = None
        modify_graph = None
        dataset = None

    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Test do not generate original_edge_id.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False
        ).load()
        assert (
            dataset.graph.edge_attributes is None
            or gb.ORIGINAL_EDGE_ID not in dataset.graph.edge_attributes
        )
        dataset = None


def test_OnDiskDataset_load_tasks():
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 10

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Case1. Test modify the `name` field.
        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["tasks"][0]["name"] = "fake_name"
        dataset.load()
        assert dataset.tasks[0].metadata["name"] == "fake_name"

        # Case2. Test modify the `num_classes` field.
        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["tasks"][0]["num_classes"] = 100
        dataset.load()
        assert dataset.tasks[0].metadata["num_classes"] == 100

        # Case3. Test modify the `format` field.
        dataset = gb.OnDiskDataset(test_dir)
        # Change the `format` field to torch.
        dataset.yaml_data["tasks"][0]["train_set"][0]["data"][0][
            "format"
        ] = "torch"
        with pytest.raises(pickle.UnpicklingError):
            dataset.load()

        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["tasks"][0]["train_set"][0]["data"][0][
            "format"
        ] = "torch"
        # Change the `in_memory` field to False will also raise an
        # UnpicklingError. Unlike the case of testing `feature_data`.
        dataset.yaml_data["tasks"][0]["train_set"][0]["data"][0][
            "in_memory"
        ] = False
        with pytest.raises(pickle.UnpicklingError):
            dataset.load()

        # Case4. Test modify the `path` field.
        dataset = gb.OnDiskDataset(test_dir)
        # Use invalid path will raise an FileNotFoundError.
        dataset.yaml_data["tasks"][0]["train_set"][0]["data"][0][
            "path"
        ] = "fake_path"
        with pytest.raises(
            FileNotFoundError,
            match=r"\[Errno 2\] No such file or directory:",
        ):
            dataset.load()

        # Modifying the `path` field to an absolute path should work.
        # In os.path.join, if a segment is an absolute path (which
        # on Windows requires both a drive and a root), then all
        # previous segments are ignored and joining continues from
        # the absolute path segment.
        dataset = gb.OnDiskDataset(test_dir).load()
        original_train_set = dataset.tasks[0].train_set._items
        dataset.yaml_data["tasks"][0]["train_set"][0]["data"][0][
            "path"
        ] = os.path.join(
            test_dir,
            dataset.yaml_data["tasks"][0]["train_set"][0]["data"][0]["path"],
        )
        dataset.load()
        modify_train_set = dataset.tasks[0].train_set._items
        assert torch.equal(
            original_train_set[0],
            modify_train_set[0],
        )
        original_train_set = None
        modify_train_set = None
        dataset = None


def test_OnDiskDataset_all_nodes_set_homo():
    """Test homograph's all nodes set of OnDiskDataset."""
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

        dataset = gb.OnDiskDataset(test_dir).load()
        all_nodes_set = dataset.all_nodes_set
        assert isinstance(all_nodes_set, gb.ItemSet)
        assert all_nodes_set.names == ("seed_nodes",)
        for i, item in enumerate(all_nodes_set):
            assert i == item

        dataset = None


def test_OnDiskDataset_all_nodes_set_hetero():
    """Test heterograph's all nodes set of OnDiskDataset."""
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

        dataset = gb.OnDiskDataset(test_dir).load()
        all_nodes_set = dataset.all_nodes_set
        assert isinstance(all_nodes_set, gb.ItemSetDict)
        assert all_nodes_set.names == ("seed_nodes",)
        for i, item in enumerate(all_nodes_set):
            assert len(item) == 1
            assert isinstance(item, dict)

        dataset = None


def test_OnDiskDataset_load_1D_feature():
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000
        num_classes = 1

        # Generate random graph.
        yaml_content = gbt.random_homo_graphbolt_graph(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        with open(yaml_file, "r") as f:
            input_config = yaml.safe_load(f)
        node_feat = np.load(
            os.path.join(test_dir, input_config["feature_data"][0]["path"])
        )
        dataset = gb.OnDiskDataset(test_dir).load()
        feature = dataset.feature.read("node", None, "feat")
        assert torch.equal(torch.from_numpy(node_feat.reshape(-1, 1)), feature)

        dataset = None
        node_feat = None
        feature = None


def test_BuiltinDataset():
    """Test BuiltinDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Case 1: download from DGL S3 storage.
        dataset_name = "test-only"
        # Add test-only dataset to the builtin dataset list for testing only.
        gb.BuiltinDataset._all_datasets.append(dataset_name)
        dataset = gb.BuiltinDataset(name=dataset_name, root=test_dir).load()
        assert dataset.graph is not None
        assert dataset.feature is not None
        assert dataset.tasks is not None
        assert dataset.dataset_name == dataset_name

        # Case 2: dataset is already downloaded.
        dataset = gb.BuiltinDataset(name=dataset_name, root=test_dir).load()
        assert dataset.graph is not None
        assert dataset.feature is not None
        assert dataset.tasks is not None
        assert dataset.dataset_name == dataset_name

        dataset = None

        # Case 3: dataset is not available.
        dataset_name = "fake_name"
        with pytest.raises(
            RuntimeError,
            match=rf"Dataset {dataset_name} is not available.*",
        ):
            _ = gb.BuiltinDataset(name=dataset_name, root=test_dir).load()
