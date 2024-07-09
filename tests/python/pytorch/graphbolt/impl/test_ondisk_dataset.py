import os
import pickle
import random
import re
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import pydantic
import pytest
import torch
import yaml
from dgl import graphbolt as gb
from dgl.graphbolt import GBWarning

from .. import gb_test_utils as gbt


def write_yaml_file(yaml_content, dir):
    os.makedirs(os.path.join(dir, "preprocessed"), exist_ok=True)
    yaml_file = os.path.join(dir, "preprocessed/metadata.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)


def load_dataset(dataset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return dataset.load()


def write_yaml_and_load_dataset(yaml_content, dir, force_preprocess=False):
    write_yaml_file(yaml_content, dir)
    return load_dataset(
        gb.OnDiskDataset(dir, force_preprocess=force_preprocess)
    )


def test_OnDiskDataset_TVTSet_exceptions():
    """Test excpetions thrown when parsing TVTSet."""
    with tempfile.TemporaryDirectory() as test_dir:
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
        write_yaml_file(yaml_content, test_dir)
        with pytest.raises(pydantic.ValidationError):
            _ = gb.OnDiskDataset(test_dir, force_preprocess=False).load()

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
        write_yaml_file(yaml_content, test_dir)
        with pytest.raises(
            AssertionError,
            match=r"Only one TVT set is allowed if type is not specified.",
        ):
            _ = gb.OnDiskDataset(test_dir, force_preprocess=False).load()


def test_OnDiskDataset_multiple_tasks():
    """Teset multiple tasks are supported."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_ids = np.arange(1000)
        train_ids_path = os.path.join(test_dir, "train_ids.npy")
        np.save(train_ids_path, train_ids)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        yaml_content = f"""
            tasks:
              - name: node_classification_1
                num_classes: 10
                train_set:
                  - type: null
                    data:
                      - name: seeds
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
              - name: node_classification_2
                num_classes: 10
                train_set:
                  - type: null
                    data:
                      - name: seeds
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
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        assert len(dataset.tasks) == 2

        for task_id in range(2):
            assert (
                dataset.tasks[task_id].metadata["name"]
                == f"node_classification_{task_id + 1}"
            )
            assert dataset.tasks[task_id].metadata["num_classes"] == 10
            # Verify train set.
            train_set = dataset.tasks[task_id].train_set
            assert len(train_set) == 1000
            assert isinstance(train_set, gb.ItemSet)
            for i, (id, label, _) in enumerate(train_set):
                assert id == train_ids[i]
                assert label == train_labels[i]
            assert train_set.names == ("seeds", "labels", None)
            train_set = None
        dataset = None


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
                      - name: seeds
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
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (id, label, _) in enumerate(train_set):
            assert id == train_ids[i]
            assert label == train_labels[i]
        assert train_set.names == ("seeds", "labels", None)
        train_set = None


def test_OnDiskDataset_TVTSet_HeteroItemSet_names():
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
                      - name: seeds
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
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.HeteroItemSet)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert "author:writes:paper" in item
            id, label, _ = item["author:writes:paper"]
            assert id == train_ids[i]
            assert label == train_labels[i]
        assert train_set.names == ("seeds", "labels", None)
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
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {train_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                validation_set:
                  - data:
                      - name: seeds
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
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {test_ids_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

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
        assert train_set.names == ("seeds", "labels")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (id, label) in enumerate(validation_set):
            assert id == validation_ids[i]
            assert label == validation_labels[i]
        assert validation_set.names == ("seeds", "labels")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (id, label) in enumerate(test_set):
            assert id == test_ids[i]
            assert label == test_labels[i]
        assert test_set.names == ("seeds", "labels")
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
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        assert dataset.tasks[0].train_set is not None
        assert dataset.tasks[0].validation_set is None
        assert dataset.tasks[0].test_set is None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pairs_labels():
    """Test TVTSet which returns ItemSet with node pairs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_seeds = np.arange(2000).reshape(1000, 2)
        train_seeds_path = os.path.join(test_dir, "train_seeds.npy")
        np.save(train_seeds_path, train_seeds)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        validation_seeds = np.arange(2000, 4000).reshape(1000, 2)
        validation_seeds_path = os.path.join(test_dir, "validation_seeds.npy")
        np.save(validation_seeds_path, validation_seeds)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_labels_path = os.path.join(test_dir, "validation_labels.npy")
        np.save(validation_labels_path, validation_labels)

        test_seeds = np.arange(4000, 6000).reshape(1000, 2)
        test_seeds_path = os.path.join(test_dir, "test_seeds.npy")
        np.save(test_seeds_path, test_seeds)
        test_labels = np.random.randint(0, 10, size=1000)
        test_labels_path = os.path.join(test_dir, "test_labels.npy")
        np.save(test_labels_path, test_labels)

        yaml_content = f"""
            tasks:
              - name: link_prediction
                train_set:
                  - type: null
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {train_seeds_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                validation_set:
                  - data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {validation_seeds_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {validation_labels_path}
                test_set:
                  - type: null
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {test_seeds_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000
        assert isinstance(train_set, gb.ItemSet)
        for i, (node_pair, label) in enumerate(train_set):
            assert node_pair[0] == train_seeds[i][0]
            assert node_pair[1] == train_seeds[i][1]
            assert label == train_labels[i]
        assert train_set.names == ("seeds", "labels")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000
        assert isinstance(validation_set, gb.ItemSet)
        for i, (node_pair, label) in enumerate(validation_set):
            assert node_pair[0] == validation_seeds[i][0]
            assert node_pair[1] == validation_seeds[i][1]
            assert label == validation_labels[i]
        assert validation_set.names == ("seeds", "labels")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000
        assert isinstance(test_set, gb.ItemSet)
        for i, (node_pair, label) in enumerate(test_set):
            assert node_pair[0] == test_seeds[i][0]
            assert node_pair[1] == test_seeds[i][1]
            assert label == test_labels[i]
        assert test_set.names == ("seeds", "labels")
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pairs_labels_indexes():
    """Test TVTSet which returns ItemSet with node pairs and negative ones."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_seeds = np.arange(2000).reshape(1000, 2)
        train_neg_dst = np.random.choice(1000 * 10, size=1000 * 10)
        train_neg_src = train_seeds[:, 0].repeat(10)
        train_neg_seeds = (
            np.concatenate((train_neg_dst, train_neg_src)).reshape(2, -1).T
        )
        train_seeds = np.concatenate((train_seeds, train_neg_seeds))
        train_seeds_path = os.path.join(test_dir, "train_seeds.npy")
        np.save(train_seeds_path, train_seeds)

        train_labels = torch.empty(1000 * 11)
        train_labels[:1000] = 1
        train_labels[1000:] = 0
        train_labels_path = os.path.join(test_dir, "train_labels.pt")
        torch.save(train_labels, train_labels_path)

        train_indexes = torch.arange(0, 1000)
        train_indexes = np.concatenate(
            (train_indexes, train_indexes.repeat_interleave(10))
        )
        train_indexes_path = os.path.join(test_dir, "train_indexes.pt")
        torch.save(train_indexes, train_indexes_path)

        validation_seeds = np.arange(2000, 4000).reshape(1000, 2)
        validation_neg_seeds = train_neg_seeds + 1
        validation_seeds = np.concatenate(
            (validation_seeds, validation_neg_seeds)
        )
        validation_seeds_path = os.path.join(test_dir, "validation_seeds.npy")
        np.save(validation_seeds_path, validation_seeds)
        validation_labels = train_labels
        validation_labels_path = os.path.join(test_dir, "validation_labels.pt")
        torch.save(validation_labels, validation_labels_path)

        validation_indexes = train_indexes
        validation_indexes_path = os.path.join(
            test_dir, "validation_indexes.pt"
        )
        torch.save(validation_indexes, validation_indexes_path)

        test_seeds = np.arange(4000, 6000).reshape(1000, 2)
        test_neg_seeds = train_neg_seeds + 2
        test_seeds = np.concatenate((test_seeds, test_neg_seeds))
        test_seeds_path = os.path.join(test_dir, "test_seeds.npy")
        np.save(test_seeds_path, test_seeds)
        test_labels = train_labels
        test_labels_path = os.path.join(test_dir, "test_labels.pt")
        torch.save(test_labels, test_labels_path)

        test_indexes = train_indexes
        test_indexes_path = os.path.join(test_dir, "test_indexes.pt")
        torch.save(test_indexes, test_indexes_path)

        yaml_content = f"""
            tasks:
              - name: link_prediction
                train_set:
                  - type: null
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {train_seeds_path}
                      - name: labels
                        format: torch
                        in_memory: true
                        path: {train_labels_path}
                      - name: indexes
                        format: torch
                        in_memory: true
                        path: {train_indexes_path}
                validation_set:
                  - data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {validation_seeds_path}
                      - name: labels
                        format: torch
                        in_memory: true
                        path: {validation_labels_path}
                      - name: indexes
                        format: torch
                        in_memory: true
                        path: {validation_indexes_path}
                test_set:
                  - type: null
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {test_seeds_path}
                      - name: labels
                        format: torch
                        in_memory: true
                        path: {test_labels_path}
                      - name: indexes
                        format: torch
                        in_memory: true
                        path: {test_indexes_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 1000 * 11
        assert isinstance(train_set, gb.ItemSet)
        for i, (node_pair, label, index) in enumerate(train_set):
            assert node_pair[0] == train_seeds[i][0]
            assert node_pair[1] == train_seeds[i][1]
            assert label == train_labels[i]
            assert index == train_indexes[i]
        assert train_set.names == ("seeds", "labels", "indexes")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 1000 * 11
        assert isinstance(validation_set, gb.ItemSet)
        for i, (node_pair, label, index) in enumerate(validation_set):
            assert node_pair[0] == validation_seeds[i][0]
            assert node_pair[1] == validation_seeds[i][1]
            assert label == validation_labels[i]
            assert index == validation_indexes[i]
        assert validation_set.names == ("seeds", "labels", "indexes")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 1000 * 11
        assert isinstance(test_set, gb.ItemSet)
        for i, (node_pair, label, index) in enumerate(test_set):
            assert node_pair[0] == test_seeds[i][0]
            assert label == test_labels[i]
            assert index == test_indexes[i]
        assert test_set.names == ("seeds", "labels", "indexes")
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_HeteroItemSet_id_label():
    """Test TVTSet which returns HeteroItemSet with IDs and labels."""
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
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {train_path}
                  - type: author
                    data:
                      - name: seeds
                        format: numpy
                        path: {train_path}
                validation_set:
                  - type: paper
                    data:
                      - name: seeds
                        format: numpy
                        path: {validation_path}
                  - type: author
                    data:
                      - name: seeds
                        format: numpy
                        path: {validation_path}
                test_set:
                  - type: paper
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: false
                        path: {test_path}
                  - type: author
                    data:
                      - name: seeds
                        format: numpy
                        path: {test_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 2000
        assert isinstance(train_set, gb.HeteroItemSet)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            id, label = item[key]
            assert id == train_ids[i % 1000]
            assert label == train_labels[i % 1000]
        assert train_set.names == ("seeds",)
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 2000
        assert isinstance(validation_set, gb.HeteroItemSet)
        for i, item in enumerate(validation_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            id, label = item[key]
            assert id == validation_ids[i % 1000]
            assert label == validation_labels[i % 1000]
        assert validation_set.names == ("seeds",)
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 2000
        assert isinstance(test_set, gb.HeteroItemSet)
        for i, item in enumerate(test_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper", "author"]
            id, label = item[key]
            assert id == test_ids[i % 1000]
            assert label == test_labels[i % 1000]
        assert test_set.names == ("seeds",)
        test_set = None
        dataset = None


def test_OnDiskDataset_TVTSet_HeteroItemSet_node_pairs_labels():
    """Test TVTSet which returns HeteroItemSet with node pairs and labels."""
    with tempfile.TemporaryDirectory() as test_dir:
        train_seeds = np.arange(2000).reshape(1000, 2)
        train_seeds_path = os.path.join(test_dir, "train_seeds.npy")
        np.save(train_seeds_path, train_seeds)
        train_labels = np.random.randint(0, 10, size=1000)
        train_labels_path = os.path.join(test_dir, "train_labels.npy")
        np.save(train_labels_path, train_labels)

        validation_seeds = np.arange(2000, 4000).reshape(1000, 2)
        validation_seeds_path = os.path.join(test_dir, "validation_seeds.npy")
        np.save(validation_seeds_path, validation_seeds)
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_labels_path = os.path.join(test_dir, "validation_labels.npy")
        np.save(validation_labels_path, validation_labels)

        test_seeds = np.arange(4000, 6000).reshape(1000, 2)
        test_seeds_path = os.path.join(test_dir, "test_seeds.npy")
        np.save(test_seeds_path, test_seeds)
        test_labels = np.random.randint(0, 10, size=1000)
        test_labels_path = os.path.join(test_dir, "test_labels.npy")
        np.save(test_labels_path, test_labels)

        yaml_content = f"""
            tasks:
              - name: edge_classification
                train_set:
                  - type: paper:cites:paper
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {train_seeds_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {train_labels_path}
                  - type: author:writes:paper
                    data:
                      - name: seeds
                        format: numpy
                        path: {train_seeds_path}
                      - name: labels
                        format: numpy
                        path: {train_labels_path}
                validation_set:
                  - type: paper:cites:paper
                    data:
                      - name: seeds
                        format: numpy
                        path: {validation_seeds_path}
                      - name: labels
                        format: numpy
                        path: {validation_labels_path}
                  - type: author:writes:paper
                    data:
                      - name: seeds
                        format: numpy
                        path: {validation_seeds_path}
                      - name: labels
                        format: numpy
                        path: {validation_labels_path}
                test_set:
                  - type: paper:cites:paper
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {test_seeds_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
                  - type: author:writes:paper
                    data:
                      - name: seeds
                        format: numpy
                        in_memory: true
                        path: {test_seeds_path}
                      - name: labels
                        format: numpy
                        in_memory: true
                        path: {test_labels_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify train set.
        train_set = dataset.tasks[0].train_set
        assert len(train_set) == 2000
        assert isinstance(train_set, gb.HeteroItemSet)
        for i, item in enumerate(train_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper:cites:paper", "author:writes:paper"]
            node_pair, label = item[key]
            assert node_pair[0] == train_seeds[i % 1000][0]
            assert node_pair[1] == train_seeds[i % 1000][1]
            assert label == train_labels[i % 1000]
        assert train_set.names == ("seeds", "labels")
        train_set = None

        # Verify validation set.
        validation_set = dataset.tasks[0].validation_set
        assert len(validation_set) == 2000
        assert isinstance(validation_set, gb.HeteroItemSet)
        for i, item in enumerate(validation_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper:cites:paper", "author:writes:paper"]
            node_pair, label = item[key]
            assert node_pair[0] == validation_seeds[i % 1000][0]
            assert node_pair[1] == validation_seeds[i % 1000][1]
            assert label == validation_labels[i % 1000]
        assert validation_set.names == ("seeds", "labels")
        validation_set = None

        # Verify test set.
        test_set = dataset.tasks[0].test_set
        assert len(test_set) == 2000
        assert isinstance(test_set, gb.HeteroItemSet)
        for i, item in enumerate(test_set):
            assert isinstance(item, dict)
            assert len(item) == 1
            key = list(item.keys())[0]
            assert key in ["paper:cites:paper", "author:writes:paper"]
            node_pair, label = item[key]
            assert node_pair[0] == test_seeds[i % 1000][0]
            assert node_pair[1] == test_seeds[i % 1000][1]
            assert label == test_labels[i % 1000]
        assert test_set.names == ("seeds", "labels")
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
                num_categories: 10
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
                num_categories: 10
              - domain: edge
                type: "author:writes:paper"
                name: labels
                format: numpy
                in_memory: true
                path: {edge_data_label_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify feature data storage.
        feature_data = dataset.feature
        assert len(feature_data) == 4

        # Verify node feature data.
        assert torch.equal(
            feature_data.read("node", "paper", "feat"),
            torch.tensor(node_data_paper),
        )
        assert (
            feature_data.metadata("node", "paper", "feat")["num_categories"]
            == 10
        )
        assert torch.equal(
            feature_data.read("node", "paper", "labels"),
            node_data_label.clone().detach(),
        )
        assert len(feature_data.metadata("node", "paper", "labels")) == 0

        # Verify edge feature data.
        assert torch.equal(
            feature_data.read("edge", "author:writes:paper", "feat"),
            torch.tensor(edge_data_writes),
        )
        assert (
            feature_data.metadata("edge", "author:writes:paper", "feat")[
                "num_categories"
            ]
            == 10
        )
        assert torch.equal(
            feature_data.read("edge", "author:writes:paper", "labels"),
            edge_data_label.clone().detach(),
        )
        assert (
            len(feature_data.metadata("edge", "author:writes:paper", "labels"))
            == 0
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
                num_categories: 10
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
                num_categories: 10
              - domain: edge
                name: labels
                format: numpy
                in_memory: true
                path: {edge_data_label_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)

        # Verify feature data storage.
        feature_data = dataset.feature
        assert len(feature_data) == 4

        # Verify node feature data.
        assert torch.equal(
            feature_data.read("node", None, "feat"),
            torch.tensor(node_data_feat),
        )
        assert (
            feature_data.metadata("node", None, "feat")["num_categories"] == 10
        )
        assert torch.equal(
            feature_data.read("node", None, "labels"),
            node_data_label.clone().detach(),
        )
        assert len(feature_data.metadata("node", None, "labels")) == 0

        # Verify edge feature data.
        assert torch.equal(
            feature_data.read("edge", None, "feat"),
            torch.tensor(edge_data_feat),
        )
        assert (
            feature_data.metadata("edge", None, "feat")["num_categories"] == 10
        )
        assert torch.equal(
            feature_data.read("edge", None, "labels"),
            edge_data_label.clone().detach(),
        )
        assert len(feature_data.metadata("edge", None, "labels")) == 0

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
        write_yaml_file(yaml_content, test_dir)

        with pytest.raises(
            pydantic.ValidationError,
            match="1 validation error for OnDiskMetaData",
        ):
            _ = gb.OnDiskDataset(test_dir, force_preprocess=False).load()


def test_OnDiskDataset_Graph_homogeneous():
    """Test homogeneous graph topology."""
    csc_indptr, indices = gbt.random_homo_graph(1000, 10 * 1000)
    graph = gb.fused_csc_sampling_graph(csc_indptr, indices)

    with tempfile.TemporaryDirectory() as test_dir:
        graph_path = os.path.join(test_dir, "fused_csc_sampling_graph.pt")
        torch.save(graph, graph_path)

        yaml_content = f"""
            graph_topology:
              type: FusedCSCSamplingGraph
              path: {graph_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        graph2 = dataset.graph

        assert graph.total_num_nodes == graph2.total_num_nodes
        assert graph.total_num_edges == graph2.total_num_edges

        assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
        assert torch.equal(graph.indices, graph2.indices)

        assert (
            graph.node_type_offset is None and graph2.node_type_offset is None
        )
        assert graph.type_per_edge is None and graph2.type_per_edge is None
        assert graph.node_type_to_id is None and graph2.node_type_to_id is None
        assert graph.edge_type_to_id is None and graph2.edge_type_to_id is None


def test_OnDiskDataset_Graph_heterogeneous():
    """Test heterogeneous graph topology."""
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(1000, 10 * 1000, 3, 4)
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
    )

    with tempfile.TemporaryDirectory() as test_dir:
        graph_path = os.path.join(test_dir, "fused_csc_sampling_graph.pt")
        torch.save(graph, graph_path)

        yaml_content = f"""
            graph_topology:
              type: FusedCSCSamplingGraph
              path: {graph_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        graph2 = dataset.graph

        assert graph.total_num_nodes == graph2.total_num_nodes
        assert graph.total_num_edges == graph2.total_num_edges

        assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
        assert torch.equal(graph.indices, graph2.indices)
        assert torch.equal(graph.node_type_offset, graph2.node_type_offset)
        assert torch.equal(graph.type_per_edge, graph2.type_per_edge)
        assert graph.node_type_to_id == graph2.node_type_to_id
        assert graph.edge_type_to_id == graph2.edge_type_to_id


def test_OnDiskDataset_Metadata():
    """Test metadata of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        assert dataset.dataset_name == dataset_name

        # Only dataset_name is specified.
        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        assert dataset.dataset_name == dataset_name


@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_preprocess_homogeneous(edge_fmt):
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
            edge_fmt=edge_fmt,
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

        fused_csc_sampling_graph = torch.load(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert fused_csc_sampling_graph.total_num_nodes == num_nodes
        assert fused_csc_sampling_graph.total_num_edges == num_edges
        assert (
            fused_csc_sampling_graph.node_attributes is not None
            and "feat" in fused_csc_sampling_graph.node_attributes
        )
        assert (
            fused_csc_sampling_graph.edge_attributes is not None
            and gb.ORIGINAL_EDGE_ID
            not in fused_csc_sampling_graph.edge_attributes
            and "feat" in fused_csc_sampling_graph.edge_attributes
        )

        num_samples = 100
        fanout = 1
        subgraph = fused_csc_sampling_graph.sample_neighbors(
            torch.arange(
                0,
                num_samples,
                dtype=fused_csc_sampling_graph.indices.dtype,
            ),
            torch.tensor([fanout]),
        )
        assert len(subgraph.sampled_csc.indices) <= num_samples

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
            edge_fmt=edge_fmt,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        # Test generating original_edge_id.
        output_file = gb.ondisk_dataset.preprocess_ondisk_dataset(
            test_dir, include_original_edge_id=True
        )
        with open(output_file, "rb") as f:
            processed_dataset = yaml.load(f, Loader=yaml.Loader)
        fused_csc_sampling_graph = torch.load(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert (
            fused_csc_sampling_graph.edge_attributes is not None
            and gb.ORIGINAL_EDGE_ID in fused_csc_sampling_graph.edge_attributes
        )
        fused_csc_sampling_graph = None


@pytest.mark.parametrize("auto_cast", [False, True])
def test_OnDiskDataset_preprocess_homogeneous_hardcode(
    auto_cast, edge_fmt="numpy"
):
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        """Original graph in COO:
        0   1   1   0   0
        0   0   1   1   0
        0   0   0   1   1
        1   0   0   0   1
        1   1   0   0   0

        node_feats: [0.0, 1.9, 2.8, 3.7, 4.6]
        edge_feats: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        """
        dataset_name = "graphbolt_test"
        num_nodes = 5
        num_edges = 10
        num_classes = 1

        # Generate edges.
        edges = np.array(
            [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 2, 2, 3, 3, 4, 4, 0, 0, 1]],
            dtype=np.int64,
        ).T
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        edges = edges.T
        edge_path = os.path.join("edges", "edge.npy")
        np.save(os.path.join(test_dir, edge_path), edges)

        # Generate graph edge-feats.
        edge_feats = np.array(
            [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
            dtype=np.float64,
        )
        os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
        edge_feat_path = os.path.join("data", "edge-feat.npy")
        np.save(os.path.join(test_dir, edge_feat_path), edge_feats)

        # Generate node-feats.
        node_feats = np.array(
            [0.0, 1.9, 2.8, 3.7, 4.6],
            dtype=np.float64,
        )
        node_feat_path = os.path.join("data", "node-feat.npy")
        np.save(os.path.join(test_dir, node_feat_path), node_feats)

        # Generate train/test/valid set.
        os.makedirs(os.path.join(test_dir, "set"), exist_ok=True)
        train_data = np.array([0, 1, 2, 3, 4])
        train_path = os.path.join("set", "train.npy")
        np.save(os.path.join(test_dir, train_path), train_data)
        valid_data = np.array([0, 1, 2, 3, 4])
        valid_path = os.path.join("set", "valid.npy")
        np.save(os.path.join(test_dir, valid_path), valid_data)
        test_data = np.array([0, 1, 2, 3, 4])
        test_path = os.path.join("set", "test.npy")
        np.save(os.path.join(test_dir, test_path), test_data)

        yaml_content = (
            f"dataset_name: {dataset_name}\n"
            f"graph:\n"
            f"  nodes:\n"
            f"    - num: {num_nodes}\n"
            f"  edges:\n"
            f"    - format: {edge_fmt}\n"
            f"      path: {edge_path}\n"
            f"  feature_data:\n"
            f"    - domain: node\n"
            f"      type: null\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {node_feat_path}\n"
            f"    - domain: edge\n"
            f"      type: null\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {edge_feat_path}\n"
            f"feature_data:\n"
            f"  - domain: node\n"
            f"    type: null\n"
            f"    name: feat\n"
            f"    format: numpy\n"
            f"    in_memory: true\n"
            f"    path: {node_feat_path}\n"
            f"  - domain: edge\n"
            f"    type: null\n"
            f"    name: feat\n"
            f"    format: numpy\n"
            f"    path: {edge_feat_path}\n"
            f"tasks:\n"
            f"  - name: node_classification\n"
            f"    num_classes: {num_classes}\n"
            f"    train_set:\n"
            f"      - type: null\n"
            f"        data:\n"
            f"          - name: seeds\n"
            f"            format: numpy\n"
            f"            in_memory: true\n"
            f"            path: {train_path}\n"
            f"    validation_set:\n"
            f"      - type: null\n"
            f"        data:\n"
            f"          - name: seeds\n"
            f"            format: numpy\n"
            f"            in_memory: true\n"
            f"            path: {valid_path}\n"
            f"    test_set:\n"
            f"      - type: null\n"
            f"        data:\n"
            f"          - name: seeds\n"
            f"            format: numpy\n"
            f"            in_memory: true\n"
            f"            path: {test_path}\n"
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        output_file = gb.ondisk_dataset.preprocess_ondisk_dataset(
            test_dir,
            include_original_edge_id=True,
            auto_cast_to_optimal_dtype=auto_cast,
        )

        with open(output_file, "rb") as f:
            processed_dataset = yaml.load(f, Loader=yaml.Loader)

        assert processed_dataset["dataset_name"] == dataset_name
        assert processed_dataset["tasks"][0]["num_classes"] == num_classes
        assert "graph" not in processed_dataset
        assert "graph_topology" in processed_dataset

        fused_csc_sampling_graph = torch.load(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert fused_csc_sampling_graph.total_num_nodes == num_nodes
        assert fused_csc_sampling_graph.total_num_edges == num_edges
        assert torch.equal(
            fused_csc_sampling_graph.csc_indptr,
            torch.tensor([0, 2, 4, 6, 8, 10]),
        )
        assert torch.equal(
            fused_csc_sampling_graph.indices,
            torch.tensor([3, 4, 0, 4, 0, 1, 1, 2, 2, 3]),
        )
        assert torch.equal(
            fused_csc_sampling_graph.node_attributes["feat"],
            torch.tensor([0.0, 1.9, 2.8, 3.7, 4.6], dtype=torch.float64),
        )
        assert torch.equal(
            fused_csc_sampling_graph.edge_attributes["feat"],
            torch.tensor(
                [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
                dtype=torch.float64,
            ),
        )
        assert torch.equal(
            fused_csc_sampling_graph.edge_attributes[gb.ORIGINAL_EDGE_ID],
            torch.tensor([7, 8, 0, 9, 1, 2, 3, 4, 5, 6]),
        )

        expected_dtype = torch.int32 if auto_cast else torch.int64
        assert fused_csc_sampling_graph.csc_indptr.dtype == expected_dtype
        assert fused_csc_sampling_graph.indices.dtype == expected_dtype
        assert (
            fused_csc_sampling_graph.edge_attributes[gb.ORIGINAL_EDGE_ID].dtype
            == expected_dtype
        )

        num_samples = 5
        fanout = 1
        subgraph = fused_csc_sampling_graph.sample_neighbors(
            torch.arange(
                0,
                num_samples,
                dtype=fused_csc_sampling_graph.indices.dtype,
            ),
            torch.tensor([fanout]),
        )
        assert len(subgraph.sampled_csc.indices) <= num_samples


@pytest.mark.parametrize("auto_cast", [False, True])
def test_OnDiskDataset_preprocess_heterogeneous_hardcode(
    auto_cast, edge_fmt="numpy"
):
    """Test preprocess of OnDiskDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        """Original graph in COO:
        0   1   1   0   0
        0   0   1   1   0
        0   0   0   1   1
        1   0   0   0   1
        1   1   0   0   0

        node_type_0: [0, 1]
        node_type_1: [2, 3, 4]
        edge_type_0: node_type_0 -> node_type_0
        edge_type_1: node_type_0 -> node_type_1
        edge_type_2: node_type_1 -> node_type_1
        edge_type_3: node_type_1 -> node_type_0

        node_feats: [0.0, 1.9, 2.8, 3.7, 4.6]
        edge_feats: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
        """
        dataset_name = "graphbolt_test"
        num_nodes = {
            "A": 2,
            "B": 3,
        }
        num_edges = {
            ("A", "a_a", "A"): 1,
            ("A", "a_b", "B"): 3,
            ("B", "b_b", "A"): 3,
            ("B", "b_a", "B"): 3,
        }
        num_classes = 1

        # Generate edges.
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        np.save(
            os.path.join(test_dir, "edges", "a_a.npy"),
            np.array([[0], [1]], dtype=np.int64),
        )
        np.save(
            os.path.join(test_dir, "edges", "a_b.npy"),
            np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64),
        )
        np.save(
            os.path.join(test_dir, "edges", "b_b.npy"),
            np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64),
        )
        np.save(
            os.path.join(test_dir, "edges", "b_a.npy"),
            np.array([[1, 2, 2], [0, 0, 1]], dtype=np.int64),
        )

        # Generate node features.
        os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
        np.save(
            os.path.join(test_dir, "data", "A-feat.npy"),
            np.array([0.0, 1.9], dtype=np.float64),
        )
        np.save(
            os.path.join(test_dir, "data", "B-feat.npy"),
            np.array([2.8, 3.7, 4.6], dtype=np.float64),
        )

        # Generate edge features.
        os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
        np.save(
            os.path.join(test_dir, "data", "a_a-feat.npy"),
            np.array([0.0], dtype=np.float64),
        )
        np.save(
            os.path.join(test_dir, "data", "a_b-feat.npy"),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
        )
        np.save(
            os.path.join(test_dir, "data", "b_b-feat.npy"),
            np.array([4.4, 5.5, 6.6], dtype=np.float64),
        )
        np.save(
            os.path.join(test_dir, "data", "b_a-feat.npy"),
            np.array([7.7, 8.8, 9.9], dtype=np.float64),
        )

        yaml_content = (
            f"dataset_name: {dataset_name}\n"
            f"graph:\n"
            f"  nodes:\n"
            f"    - type: A\n"
            f"      num: 2\n"
            f"    - type: B\n"
            f"      num: 3\n"
            f"  edges:\n"
            f"    - type: A:a_a:A\n"
            f"      format: {edge_fmt}\n"
            f"      path: {os.path.join('edges', 'a_a.npy')}\n"
            f"    - type: A:a_b:B\n"
            f"      format: {edge_fmt}\n"
            f"      path: {os.path.join('edges', 'a_b.npy')}\n"
            f"    - type: B:b_b:B\n"
            f"      format: {edge_fmt}\n"
            f"      path: {os.path.join('edges', 'b_b.npy')}\n"
            f"    - type: B:b_a:A\n"
            f"      format: {edge_fmt}\n"
            f"      path: {os.path.join('edges', 'b_a.npy')}\n"
            f"  feature_data:\n"
            f"    - domain: node\n"
            f"      type: A\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {os.path.join(test_dir, 'data', 'A-feat.npy')}\n"
            f"    - domain: node\n"
            f"      type: B\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {os.path.join(test_dir, 'data', 'B-feat.npy')}\n"
            f"    - domain: edge\n"
            f"      type: A:a_a:A\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {os.path.join(test_dir, 'data', 'a_a-feat.npy')}\n"
            f"    - domain: edge\n"
            f"      type: A:a_b:B\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {os.path.join(test_dir, 'data', 'a_b-feat.npy')}\n"
            f"    - domain: edge\n"
            f"      type: B:b_b:B\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {os.path.join(test_dir, 'data', 'b_b-feat.npy')}\n"
            f"    - domain: edge\n"
            f"      type: B:b_a:A\n"
            f"      name: feat\n"
            f"      format: numpy\n"
            f"      in_memory: true\n"
            f"      path: {os.path.join(test_dir, 'data', 'b_a-feat.npy')}\n"
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        output_file = gb.ondisk_dataset.preprocess_ondisk_dataset(
            test_dir,
            include_original_edge_id=True,
            auto_cast_to_optimal_dtype=auto_cast,
        )

        with open(output_file, "rb") as f:
            processed_dataset = yaml.load(f, Loader=yaml.Loader)

        assert processed_dataset["dataset_name"] == dataset_name
        assert "graph" not in processed_dataset
        assert "graph_topology" in processed_dataset

        fused_csc_sampling_graph = torch.load(
            os.path.join(test_dir, processed_dataset["graph_topology"]["path"])
        )
        assert fused_csc_sampling_graph.total_num_nodes == 5
        assert fused_csc_sampling_graph.total_num_edges == 10
        assert torch.equal(
            fused_csc_sampling_graph.csc_indptr,
            torch.tensor([0, 2, 4, 6, 8, 10]),
        )
        assert torch.equal(
            fused_csc_sampling_graph.indices,
            torch.tensor([3, 4, 0, 4, 0, 1, 1, 2, 2, 3]),
        )
        assert torch.equal(
            fused_csc_sampling_graph.node_attributes["feat"],
            torch.tensor([0.0, 1.9, 2.8, 3.7, 4.6], dtype=torch.float64),
        )
        assert torch.equal(
            fused_csc_sampling_graph.edge_attributes["feat"],
            torch.tensor(
                [0.0, 1.1, 2.2, 3.3, 7.7, 8.8, 9.9, 4.4, 5.5, 6.6],
                dtype=torch.float64,
            ),
        )
        assert torch.equal(
            fused_csc_sampling_graph.type_per_edge,
            torch.tensor([2, 2, 0, 2, 1, 1, 1, 3, 3, 3]),
        )
        assert torch.equal(
            fused_csc_sampling_graph.edge_attributes[gb.ORIGINAL_EDGE_ID],
            torch.tensor([0, 1, 0, 2, 0, 1, 2, 0, 1, 2]),
        )
        expected_dtype = torch.int32 if auto_cast else torch.int64
        assert fused_csc_sampling_graph.csc_indptr.dtype == expected_dtype
        assert fused_csc_sampling_graph.indices.dtype == expected_dtype
        assert (
            fused_csc_sampling_graph.edge_attributes[gb.ORIGINAL_EDGE_ID].dtype
            == expected_dtype
        )
        assert fused_csc_sampling_graph.node_type_offset.dtype == expected_dtype
        expected_etype_dtype = torch.uint8 if auto_cast else torch.int64
        assert (
            fused_csc_sampling_graph.type_per_edge.dtype == expected_etype_dtype
        )


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


def test_OnDiskDataset_preprocess_yaml_content():
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
        # Write into edges/edge.csv
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        edges = pd.DataFrame(edges, columns=["src", "dst"])
        edge_path = os.path.join("edges", "edge.csv")
        edges.to_csv(
            os.path.join(test_dir, edge_path),
            index=False,
            header=False,
        )

        # Generate random graph edge-feats.
        edge_feats = np.random.rand(num_edges, 5)
        os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
        feature_edge = os.path.join("data", "edge-feat.npy")
        np.save(os.path.join(test_dir, feature_edge), edge_feats)

        # Generate random node-feats.
        node_feats = np.random.rand(num_nodes, 10)
        feature_node = os.path.join("data", "node-feat.npy")
        np.save(os.path.join(test_dir, feature_node), node_feats)

        # Generate train/test/valid set.
        os.makedirs(os.path.join(test_dir, "set"), exist_ok=True)
        train_pairs = (np.arange(1000), np.arange(1000, 2000))
        train_labels = np.random.randint(0, 10, size=1000)
        train_data = np.vstack([train_pairs, train_labels]).T
        train_path = os.path.join("set", "train.npy")
        np.save(os.path.join(test_dir, train_path), train_data)

        validation_pairs = (np.arange(1000, 2000), np.arange(2000, 3000))
        validation_labels = np.random.randint(0, 10, size=1000)
        validation_data = np.vstack([validation_pairs, validation_labels]).T
        validation_path = os.path.join("set", "validation.npy")
        np.save(os.path.join(test_dir, validation_path), validation_data)

        test_pairs = (np.arange(2000, 3000), np.arange(3000, 4000))
        test_labels = np.random.randint(0, 10, size=1000)
        test_data = np.vstack([test_pairs, test_labels]).T
        test_path = os.path.join("set", "test.npy")
        np.save(os.path.join(test_dir, test_path), test_data)

        yaml_content = f"""
            dataset_name: {dataset_name}
            graph: # graph structure and required attributes.
                nodes:
                    - num: {num_nodes}
                edges:
                    - format: csv
                      path: {edge_path}
                feature_data:
                    - domain: edge
                      type: null
                      name: feat
                      format: numpy
                      in_memory: true
                      path: {feature_edge}
            feature_data:
                - domain: node
                  type: null
                  name: feat
                  format: numpy
                  in_memory: false
                  path: {feature_node}
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {train_path}
                validation_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {validation_path}
                test_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {test_path}
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        preprocessed_metadata_path = gb.preprocess_ondisk_dataset(test_dir)
        with open(preprocessed_metadata_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        topo_path = os.path.join("preprocessed", "fused_csc_sampling_graph.pt")
        target_yaml_content = f"""
            dataset_name: {dataset_name}
            graph_topology:
              type: FusedCSCSamplingGraph
              path: {topo_path}
            feature_data:
              - domain: node
                type: null
                name: feat
                format: numpy
                in_memory: false
                path: {os.path.join("preprocessed", feature_node)}
            tasks:
              - name: node_classification
                num_classes: {num_classes}
                train_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {os.path.join("preprocessed", train_path)}
                validation_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {os.path.join("preprocessed", validation_path)}
                test_set:
                  - type: null
                    data:
                      - format: numpy
                        path: {os.path.join("preprocessed", test_path)}
            include_original_edge_id: False
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


def test_OnDiskDataset_preprocess_force_preprocess(capsys):
    """Test force preprocess of OnDiskDataset."""
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

        # First preprocess on-disk dataset.
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False, force_preprocess=False
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["tasks"][0]["name"] == "link_prediction"

        # Change yaml_data, but do not force preprocess on-disk dataset.
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["tasks"][0]["name"] = "fake_name"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_data, f)
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False, force_preprocess=False
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == ["The dataset is already preprocessed.", ""]
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["tasks"][0]["name"] == "link_prediction"

        # Force preprocess on-disk dataset.
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False, force_preprocess=True
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["tasks"][0]["name"] == "fake_name"


def test_OnDiskDataset_preprocess_auto_force_preprocess(capsys):
    """Test force preprocess of OnDiskDataset."""
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

        # First preprocess on-disk dataset.
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["tasks"][0]["name"] == "link_prediction"

        # 1. Change yaml_data.
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["tasks"][0]["name"] = "fake_name"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_data, f)
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["tasks"][0]["name"] == "fake_name"

        # 2. Change edge feature.
        edge_feats = np.random.rand(num_edges, num_classes)
        edge_feat_path = os.path.join("data", "edge-feat.npy")
        np.save(os.path.join(test_dir, edge_feat_path), edge_feats)
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        preprocessed_edge_feat = np.load(
            os.path.join(test_dir, "preprocessed", edge_feat_path)
        )
        assert preprocessed_edge_feat.all() == edge_feats.all()
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["include_original_edge_id"] == False

        # 3. Change include_original_edge_id.
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=True
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        with open(preprocessed_metadata_path, "r") as f:
            target_yaml_data = yaml.safe_load(f)
        assert target_yaml_data["include_original_edge_id"] == True

        # 4. Change nothing.
        preprocessed_metadata_path = (
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=True
            )
        )
        captured = capsys.readouterr().out.split("\n")
        assert captured == ["The dataset is already preprocessed.", ""]


def test_OnDiskDataset_preprocess_not_include_eids():
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

        with pytest.warns(
            GBWarning,
            match="Edge feature is stored, but edge IDs are not saved.",
        ):
            gb.ondisk_dataset.preprocess_ondisk_dataset(
                test_dir, include_original_edge_id=False
            )


@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_load_name(edge_fmt):
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
            edge_fmt=edge_fmt,
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


@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_load_feature(edge_fmt):
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
            edge_fmt=edge_fmt,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Case1. Test modify the `in_memory` field.
        dataset = gb.OnDiskDataset(test_dir).load()
        original_feature_data = dataset.feature
        dataset.yaml_data["feature_data"][0]["in_memory"] = True
        load_dataset(dataset)
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
        dataset.yaml_data["feature_data"][0]["in_memory"] = False
        dataset.yaml_data["feature_data"][0]["format"] = "torch"
        with pytest.raises(
            AssertionError,
            match="^Pytorch tensor can only be loaded in memory,",
        ):
            load_dataset(dataset)

        dataset = gb.OnDiskDataset(test_dir)
        dataset.yaml_data["feature_data"][0]["in_memory"] = True
        dataset.yaml_data["feature_data"][0]["format"] = "torch"
        # If `format` is torch and `in_memory` is True, it will
        # raise an UnpicklingError.
        with pytest.raises(pickle.UnpicklingError):
            load_dataset(dataset)

        # Case3. Test modify the `path` field.
        dataset = gb.OnDiskDataset(test_dir)
        # Use invalid path will raise an FileNotFoundError.
        dataset.yaml_data["feature_data"][0]["path"] = "fake_path"
        with pytest.raises(
            FileNotFoundError,
            match=r"\[Errno 2\] No such file or directory:",
        ):
            load_dataset(dataset)
        # Modifying the `path` field to an absolute path should work.
        # In os.path.join, if a segment is an absolute path (which
        # on Windows requires both a drive and a root), then all
        # previous segments are ignored and joining continues from
        # the absolute path segment.
        dataset = load_dataset(gb.OnDiskDataset(test_dir))
        original_feature_data = dataset.feature
        dataset.yaml_data["feature_data"][0]["path"] = os.path.join(
            test_dir, dataset.yaml_data["feature_data"][0]["path"]
        )
        load_dataset(dataset)
        modify_feature_data = dataset.feature
        assert torch.equal(
            original_feature_data.read("node", None, "feat"),
            modify_feature_data.read("node", None, "feat"),
        )
        original_feature_data = None
        modify_feature_data = None
        dataset = None


@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_load_graph(edge_fmt):
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
            edge_fmt=edge_fmt,
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
            match="'FusedCSCSamplingGraph'",
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
            edge_fmt=edge_fmt,
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


@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_load_tasks(edge_fmt):
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
            edge_fmt=edge_fmt,
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
    graph = gb.fused_csc_sampling_graph(csc_indptr, indices)

    with tempfile.TemporaryDirectory() as test_dir:
        graph_path = os.path.join(test_dir, "fused_csc_sampling_graph.pt")
        torch.save(graph, graph_path)

        yaml_content = f"""
            graph_topology:
              type: FusedCSCSamplingGraph
              path: {graph_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        all_nodes_set = dataset.all_nodes_set
        assert isinstance(all_nodes_set, gb.ItemSet)
        assert all_nodes_set.names == ("seeds",)
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
        node_type_to_id,
        edge_type_to_id,
    ) = gbt.random_hetero_graph(1000, 10 * 1000, 3, 4)
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        edge_attributes=None,
    )

    with tempfile.TemporaryDirectory() as test_dir:
        graph_path = os.path.join(test_dir, "fused_csc_sampling_graph.pt")
        torch.save(graph, graph_path)

        yaml_content = f"""
            graph_topology:
              type: FusedCSCSamplingGraph
              path: {graph_path}
        """
        dataset = write_yaml_and_load_dataset(yaml_content, test_dir)
        all_nodes_set = dataset.all_nodes_set
        assert isinstance(all_nodes_set, gb.HeteroItemSet)
        assert all_nodes_set.names == ("seeds",)
        for i, item in enumerate(all_nodes_set):
            assert len(item) == 1
            assert isinstance(item, dict)

        dataset = None


@pytest.mark.parametrize("fmt", ["numpy", "torch"])
def test_OnDiskDataset_load_1D_feature(fmt):
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4
        num_edges = 20
        num_classes = 1

        type_name = "npy" if fmt == "numpy" else "pt"
        # Generate random edges.
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors], axis=1)
        # Write into edges/edge.csv
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        edges = pd.DataFrame(edges, columns=["src", "dst"])
        edge_path = os.path.join("edges", "edge.csv")
        edges.to_csv(
            os.path.join(test_dir, edge_path),
            index=False,
            header=False,
        )

        # Generate random graph edge-feats.
        edge_feats = np.random.rand(num_edges, 5)
        os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
        edge_feat_path = os.path.join("data", f"edge-feat.{type_name}")

        # Generate random 1-D node-feats.
        node_feats = np.random.rand(num_nodes)
        node_feat_path = os.path.join("data", f"node-feat.{type_name}")
        assert node_feats.ndim == 1

        # Generate 1-D train set.
        os.makedirs(os.path.join(test_dir, "set"), exist_ok=True)
        train_path = os.path.join("set", f"train.{type_name}")

        if fmt == "numpy":
            np.save(os.path.join(test_dir, edge_feat_path), edge_feats)
            np.save(os.path.join(test_dir, node_feat_path), node_feats)
            np.save(os.path.join(test_dir, train_path), np.array([0, 1, 0]))
        else:
            torch.save(
                torch.from_numpy(edge_feats),
                os.path.join(test_dir, edge_feat_path),
            )
            torch.save(
                torch.from_numpy(node_feats),
                os.path.join(test_dir, node_feat_path),
            )
            torch.save(
                torch.tensor([0, 1, 0]), os.path.join(test_dir, train_path)
            )

        yaml_content = f"""
            dataset_name: {dataset_name}
            graph: # graph structure and required attributes.
              nodes:
                - num: {num_nodes}
              edges:
                - format: csv
                  path: {edge_path}
              feature_data:
                  - domain: edge
                    type: null
                    name: feat
                    format: {fmt}
                    in_memory: true
                    path: {edge_feat_path}
            feature_data:
              - domain: node
                type: null
                name: feat
                format: {fmt}
                in_memory: false
                path: {node_feat_path}
            tasks:
                - name: node_classification
                  num_classes: {num_classes}
                  train_set:
                    - type: null
                      data:
                        - format: {fmt}
                          path: {train_path}
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()
        feature = dataset.feature.read("node", None, "feat")
        # Test whether feature has changed.
        assert torch.equal(torch.from_numpy(node_feats.reshape(-1, 1)), feature)
        # Test whether itemsets keep same.
        assert torch.equal(
            dataset.tasks[0].train_set._items[0], torch.tensor([0, 1, 0])
        )
        dataset = None
        node_feats = None
        feature = None


def test_BuiltinDataset():
    """Test BuiltinDataset."""
    with tempfile.TemporaryDirectory() as test_dir:
        # Case 1: download from DGL S3 storage.
        dataset_name = "test-dataset-231207"
        # Add dataset to the builtin dataset list for testing only. Due to we
        # add `seeds` suffix to datasets when downloading, so we append
        # dataset name with `-seeds` suffix here.
        gb.BuiltinDataset._all_datasets.append(dataset_name + "-seeds")
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
        dataset_name = "fake_name-seeds"
        with pytest.raises(
            RuntimeError,
            match=rf"Dataset {dataset_name} is not available.*",
        ):
            _ = gb.BuiltinDataset(name=dataset_name, root=test_dir).load()


@pytest.mark.parametrize("auto_cast", [True, False])
@pytest.mark.parametrize("include_original_edge_id", [True, False])
@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_homogeneous(
    auto_cast, include_original_edge_id, edge_fmt
):
    """Preprocess and instantiate OnDiskDataset for homogeneous graph."""
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
            edge_fmt=edge_fmt,
        )
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(
            test_dir,
            include_original_edge_id=include_original_edge_id,
            auto_cast_to_optimal_dtype=auto_cast,
        ).load()

        assert dataset.dataset_name == dataset_name

        graph = dataset.graph
        assert isinstance(graph, gb.FusedCSCSamplingGraph)
        assert graph.total_num_nodes == num_nodes
        assert graph.total_num_edges == num_edges
        assert (
            graph.node_attributes is not None
            and "feat" in graph.node_attributes
        )
        assert (
            graph.edge_attributes is not None
            and "feat" in graph.edge_attributes
        )
        assert (
            not include_original_edge_id
        ) or gb.ORIGINAL_EDGE_ID in graph.edge_attributes

        tasks = dataset.tasks
        assert len(tasks) == 1
        assert isinstance(tasks[0].train_set, gb.ItemSet)
        assert isinstance(tasks[0].validation_set, gb.ItemSet)
        assert isinstance(tasks[0].test_set, gb.ItemSet)
        assert tasks[0].train_set._items[0].dtype == graph.indices.dtype
        assert tasks[0].validation_set._items[0].dtype == graph.indices.dtype
        assert tasks[0].test_set._items[0].dtype == graph.indices.dtype
        assert dataset.all_nodes_set._items.dtype == graph.indices.dtype
        assert tasks[0].metadata["num_classes"] == num_classes
        assert tasks[0].metadata["name"] == "link_prediction"

        assert dataset.feature.size("node", None, "feat")[0] == num_classes
        assert dataset.feature.size("edge", None, "feat")[0] == num_classes

        for itemset in [
            tasks[0].train_set,
            tasks[0].validation_set,
            tasks[0].test_set,
            dataset.all_nodes_set,
        ]:
            datapipe = gb.ItemSampler(itemset, batch_size=10)
            datapipe = datapipe.sample_neighbor(graph, [-1])
            datapipe = datapipe.fetch_feature(
                dataset.feature, node_feature_keys=["feat"]
            )
            dataloader = gb.DataLoader(datapipe)
            for _ in dataloader:
                pass

        graph = None
        tasks = None
        dataset = None


@pytest.mark.parametrize("auto_cast", [True, False])
@pytest.mark.parametrize("include_original_edge_id", [True, False])
@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_OnDiskDataset_heterogeneous(
    auto_cast, include_original_edge_id, edge_fmt
):
    """Preprocess and instantiate OnDiskDataset for heterogeneous graph."""
    with tempfile.TemporaryDirectory() as test_dir:
        dataset_name = "OnDiskDataset_hetero"
        num_nodes = {
            "user": 1000,
            "item": 2000,
        }
        num_edges = {
            ("user", "follow", "user"): 10000,
            ("user", "click", "item"): 20000,
        }
        num_classes = 10
        gbt.generate_raw_data_for_hetero_dataset(
            test_dir,
            dataset_name,
            num_nodes,
            num_edges,
            num_classes,
            edge_fmt=edge_fmt,
        )

        dataset = gb.OnDiskDataset(
            test_dir,
            include_original_edge_id=include_original_edge_id,
            auto_cast_to_optimal_dtype=auto_cast,
        ).load()

        assert dataset.dataset_name == dataset_name

        graph = dataset.graph
        assert isinstance(graph, gb.FusedCSCSamplingGraph)
        assert graph.total_num_nodes == sum(
            num_nodes for num_nodes in num_nodes.values()
        )
        assert graph.total_num_edges == sum(
            num_edge for num_edge in num_edges.values()
        )
        expected_dtype = torch.int32 if auto_cast else torch.int64
        assert graph.indices.dtype == expected_dtype
        assert (
            graph.node_attributes is not None
            and "feat" in graph.node_attributes
        )
        assert (
            graph.edge_attributes is not None
            and "feat" in graph.edge_attributes
        )
        assert (
            not include_original_edge_id
        ) or gb.ORIGINAL_EDGE_ID in graph.edge_attributes

        tasks = dataset.tasks
        assert len(tasks) == 1
        assert isinstance(tasks[0].train_set, gb.HeteroItemSet)
        assert isinstance(tasks[0].validation_set, gb.HeteroItemSet)
        assert isinstance(tasks[0].test_set, gb.HeteroItemSet)
        assert tasks[0].metadata["num_classes"] == num_classes
        assert tasks[0].metadata["name"] == "node_classification"

        assert dataset.feature.size("node", "user", "feat")[0] == num_classes
        assert dataset.feature.size("node", "item", "feat")[0] == num_classes

        for itemset in [
            tasks[0].train_set,
            tasks[0].validation_set,
            tasks[0].test_set,
            dataset.all_nodes_set,
        ]:
            datapipe = gb.ItemSampler(itemset, batch_size=10)
            datapipe = datapipe.sample_neighbor(graph, [-1])
            datapipe = datapipe.fetch_feature(
                dataset.feature, node_feature_keys={"user": ["feat"]}
            )
            dataloader = gb.DataLoader(datapipe)
            for _ in dataloader:
                pass

        graph = None
        tasks = None
        dataset = None


def test_OnDiskDataset_force_preprocess(capsys):
    """Test force preprocess of OnDiskDataset."""
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

        # First preprocess on-disk dataset.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False, force_preprocess=False
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        tasks = dataset.tasks
        assert tasks[0].metadata["name"] == "link_prediction"

        # Change yaml_data, but do not force preprocess on-disk dataset.
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["tasks"][0]["name"] = "fake_name"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_data, f)
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False, force_preprocess=False
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == ["The dataset is already preprocessed.", ""]
        tasks = dataset.tasks
        assert tasks[0].metadata["name"] == "link_prediction"

        # Force preprocess on-disk dataset.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False, force_preprocess=True
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        tasks = dataset.tasks
        assert tasks[0].metadata["name"] == "fake_name"

        tasks = None
        dataset = None


def test_OnDiskDataset_auto_force_preprocess(capsys):
    """Test force preprocess of OnDiskDataset."""
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

        # First preprocess on-disk dataset.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        tasks = dataset.tasks
        assert tasks[0].metadata["name"] == "link_prediction"

        # 1. Change yaml_data.
        with open(yaml_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        yaml_data["tasks"][0]["name"] = "fake_name"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_data, f)
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        tasks = dataset.tasks
        assert tasks[0].metadata["name"] == "fake_name"

        # 2. Change edge feature.
        edge_feats = np.random.rand(num_edges, num_classes)
        edge_feat_path = os.path.join("data", "edge-feat.npy")
        np.save(os.path.join(test_dir, edge_feat_path), edge_feats)
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=False
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        assert torch.equal(
            dataset.feature.read("edge", None, "feat"),
            torch.from_numpy(edge_feats),
        )
        graph = dataset.graph
        assert gb.ORIGINAL_EDGE_ID not in graph.edge_attributes

        # 3. Change include_original_edge_id.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=True
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == [
            "The on-disk dataset is re-preprocessing, so the existing "
            + "preprocessed dataset has been removed.",
            "Start to preprocess the on-disk dataset.",
            "Finish preprocessing the on-disk dataset.",
            "",
        ]
        graph = dataset.graph
        assert gb.ORIGINAL_EDGE_ID in graph.edge_attributes

        # 4. Change Nothing.
        dataset = gb.OnDiskDataset(
            test_dir, include_original_edge_id=True
        ).load()
        captured = capsys.readouterr().out.split("\n")
        assert captured == ["The dataset is already preprocessed.", ""]

        graph = None
        tasks = None
        dataset = None


def test_OnDiskTask_repr_homogeneous():
    item_set = gb.ItemSet(
        (torch.arange(0, 5), torch.arange(5, 10)),
        names=("seeds", "labels"),
    )
    metadata = {"name": "node_classification"}
    task = gb.OnDiskTask(metadata, item_set, item_set, item_set)
    expected_str = (
        "OnDiskTask(validation_set=ItemSet(\n"
        "               items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "               names=('seeds', 'labels'),\n"
        "           ),\n"
        "           train_set=ItemSet(\n"
        "               items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "               names=('seeds', 'labels'),\n"
        "           ),\n"
        "           test_set=ItemSet(\n"
        "               items=(tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9])),\n"
        "               names=('seeds', 'labels'),\n"
        "           ),\n"
        "           metadata={'name': 'node_classification'},)"
    )
    assert repr(task) == expected_str, task


def test_OnDiskDataset_not_include_eids():
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

        with pytest.warns(
            GBWarning,
            match="Edge feature is stored, but edge IDs are not saved.",
        ):
            gb.OnDiskDataset(test_dir, include_original_edge_id=False)


def test_OnDiskTask_repr_heterogeneous():
    item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(torch.arange(0, 5), names="seeds"),
            "item": gb.ItemSet(torch.arange(5, 10), names="seeds"),
        }
    )
    metadata = {"name": "node_classification"}
    task = gb.OnDiskTask(metadata, item_set, item_set, item_set)
    expected_str = (
        "OnDiskTask(validation_set=HeteroItemSet(\n"
        "               itemsets={'user': ItemSet(\n"
        "                            items=(tensor([0, 1, 2, 3, 4]),),\n"
        "                            names=('seeds',),\n"
        "                        ), 'item': ItemSet(\n"
        "                            items=(tensor([5, 6, 7, 8, 9]),),\n"
        "                            names=('seeds',),\n"
        "                        )},\n"
        "               names=('seeds',),\n"
        "           ),\n"
        "           train_set=HeteroItemSet(\n"
        "               itemsets={'user': ItemSet(\n"
        "                            items=(tensor([0, 1, 2, 3, 4]),),\n"
        "                            names=('seeds',),\n"
        "                        ), 'item': ItemSet(\n"
        "                            items=(tensor([5, 6, 7, 8, 9]),),\n"
        "                            names=('seeds',),\n"
        "                        )},\n"
        "               names=('seeds',),\n"
        "           ),\n"
        "           test_set=HeteroItemSet(\n"
        "               itemsets={'user': ItemSet(\n"
        "                            items=(tensor([0, 1, 2, 3, 4]),),\n"
        "                            names=('seeds',),\n"
        "                        ), 'item': ItemSet(\n"
        "                            items=(tensor([5, 6, 7, 8, 9]),),\n"
        "                            names=('seeds',),\n"
        "                        )},\n"
        "               names=('seeds',),\n"
        "           ),\n"
        "           metadata={'name': 'node_classification'},)"
    )
    assert repr(task) == expected_str, task


def test_OnDiskDataset_load_tasks_selectively():
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
        train_path = os.path.join("set", "train.npy")

        yaml_content += f"""      - name: node_classification
            num_classes: {num_classes}
            train_set:
              - type: null
                data:
                  - format: numpy
                    path: {train_path}
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        # Case1. Test load all tasks.
        dataset = gb.OnDiskDataset(test_dir).load()
        assert len(dataset.tasks) == 2

        # Case2. Test load tasks selectively.
        dataset = gb.OnDiskDataset(test_dir).load(tasks="link_prediction")
        assert len(dataset.tasks) == 1
        assert dataset.tasks[0].metadata["name"] == "link_prediction"
        dataset = gb.OnDiskDataset(test_dir).load(tasks=["link_prediction"])
        assert len(dataset.tasks) == 1
        assert dataset.tasks[0].metadata["name"] == "link_prediction"

        # Case3. Test load tasks with non-existent task name.
        with pytest.warns(
            GBWarning,
            match="Below tasks are not found in YAML: {'fake-name'}. Skipped.",
        ):
            dataset = gb.OnDiskDataset(test_dir).load(tasks=["fake-name"])
            assert len(dataset.tasks) == 0

        # Case4. Test load tasks selectively with incorrect task type.
        with pytest.raises(TypeError):
            dataset = gb.OnDiskDataset(test_dir).load(tasks=2)

        dataset = None


def test_OnDiskDataset_preprocess_graph_with_single_type():
    """Test for graph with single node/edge type."""
    with tempfile.TemporaryDirectory() as test_dir:
        # All metadata fields are specified.
        dataset_name = "graphbolt_test"
        num_nodes = 4000
        num_edges = 20000

        # Generate random edges.
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors], axis=1)
        # Write into edges/edge.csv
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

        yaml_content = f"""
            dataset_name: {dataset_name}
            graph: # graph structure and required attributes.
                nodes:
                    - num: {num_nodes}
                      type: author
                edges:
                    - type: author:collab:author
                      format: csv
                      path: edges/edge.csv
                feature_data:
                    - domain: edge
                      type: author:collab:author
                      name: feat
                      format: numpy
                      path: data/edge-feat.npy
                    - domain: node
                      type: author
                      name: feat
                      format: numpy
                      path: data/node-feat.npy
        """
        yaml_file = os.path.join(test_dir, "metadata.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(test_dir).load()
        assert dataset.dataset_name == dataset_name

        graph = dataset.graph
        assert isinstance(graph, gb.FusedCSCSamplingGraph)
        assert graph.total_num_nodes == num_nodes
        assert graph.total_num_edges == num_edges
        assert (
            graph.node_attributes is not None
            and "feat" in graph.node_attributes
        )
        assert (
            graph.edge_attributes is not None
            and "feat" in graph.edge_attributes
        )
        assert torch.equal(graph.node_type_offset, torch.tensor([0, num_nodes]))
        assert torch.equal(
            graph.type_per_edge,
            torch.zeros(num_edges),
        )
        assert graph.edge_type_to_id == {"author:collab:author": 0}
        assert graph.node_type_to_id == {"author": 0}
