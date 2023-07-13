import os
import tempfile

import gb_test_utils as gbt

import numpy as np

import pydantic
import pytest
import torch
from dgl import graphbolt as gb


def test_OnDiskDataset_TVTSet_exceptions():
    """Test excpetions thrown when parsing TVTSet."""
    with tempfile.TemporaryDirectory() as test_dir:
        yaml_file = os.path.join(test_dir, "test.yaml")

        # Case 1: ``format`` is invalid.
        yaml_content = """
        train_sets:
          - - type: paper
              format: torch_invalid
              path: set/paper-train.pt
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        with pytest.raises(pydantic.ValidationError):
            _ = gb.OnDiskDataset(yaml_file)

        # Case 2: ``type`` is not specified while multiple TVT sets are specified.
        yaml_content = """
            train_sets:
              - - type: null
                  format: numpy
                  path: set/train.npy
                - type: null
                  format: numpy
                  path: set/train.npy
        """
        with open(yaml_file, "w") as f:
            f.write(yaml_content)
        with pytest.raises(
            AssertionError,
            match=r"Only one TVT set is allowed if type is not specified.",
        ):
            _ = gb.OnDiskDataset(yaml_file)


def test_OnDiskDataset_TVTSet_ItemSet_id_label():
    """Test TVTSet which returns ItemSet with IDs and labels."""
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

        # Case 1:
        #   all TVT sets are specified.
        #   ``type`` is not specified or specified as ``null``.
        #   ``in_memory`` could be ``true`` and ``false``.
        yaml_content = f"""
            train_sets:
              - - type: null
                  format: numpy
                  in_memory: true
                  path: {train_path}
              - - type: null
                  format: numpy
                  path: {train_path}
            validation_sets:
              - - format: numpy
                  path: {validation_path}
              - - type: null
                  format: numpy
                  path: {validation_path}
            test_sets:
              - - type: null
                  format: numpy
                  in_memory: false
                  path: {test_path}
              - - type: null
                  format: numpy
                  path: {test_path}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)

        # Verify train set.
        train_sets = dataset.train_sets()
        assert len(train_sets) == 2
        for train_set in train_sets:
            assert len(train_set) == 1000
            assert isinstance(train_set, gb.ItemSet)
            for i, (id, label) in enumerate(train_set):
                assert id == train_ids[i]
                assert label == train_labels[i]
        train_sets = None

        # Verify validation set.
        validation_sets = dataset.validation_sets()
        assert len(validation_sets) == 2
        for validation_set in validation_sets:
            assert len(validation_set) == 1000
            assert isinstance(validation_set, gb.ItemSet)
            for i, (id, label) in enumerate(validation_set):
                assert id == validation_ids[i]
                assert label == validation_labels[i]
        validation_sets = None

        # Verify test set.
        test_sets = dataset.test_sets()
        assert len(test_sets) == 2
        for test_set in test_sets:
            assert len(test_set) == 1000
            assert isinstance(test_set, gb.ItemSet)
            for i, (id, label) in enumerate(test_set):
                assert id == test_ids[i]
                assert label == test_labels[i]
        test_sets = None
        dataset = None

        # Case 2: Some TVT sets are None.
        yaml_content = f"""
            train_sets:
              - - type: null
                  format: numpy
                  path: {train_path}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)
        assert dataset.train_sets() is not None
        assert dataset.validation_sets() is None
        assert dataset.test_sets() is None
        dataset = None


def test_OnDiskDataset_TVTSet_ItemSet_node_pair_label():
    """Test TVTSet which returns ItemSet with IDs and labels."""
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
            train_sets:
              - - type: null
                  format: numpy
                  in_memory: true
                  path: {train_path}
              - - type: null
                  format: numpy
                  path: {train_path}
            validation_sets:
              - - format: numpy
                  path: {validation_path}
              - - type: null
                  format: numpy
                  path: {validation_path}
            test_sets:
              - - type: null
                  format: numpy
                  in_memory: false
                  path: {test_path}
              - - type: null
                  format: numpy
                  path: {test_path}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)

        # Verify train set.
        train_sets = dataset.train_sets()
        assert len(train_sets) == 2
        for train_set in train_sets:
            assert len(train_set) == 1000
            assert isinstance(train_set, gb.ItemSet)
            for i, (src, dst, label) in enumerate(train_set):
                assert src == train_pairs[0][i]
                assert dst == train_pairs[1][i]
                assert label == train_labels[i]
        train_sets = None

        # Verify validation set.
        validation_sets = dataset.validation_sets()
        assert len(validation_sets) == 2
        for validation_set in validation_sets:
            assert len(validation_set) == 1000
            assert isinstance(validation_set, gb.ItemSet)
            for i, (src, dst, label) in enumerate(validation_set):
                assert src == validation_pairs[0][i]
                assert dst == validation_pairs[1][i]
                assert label == validation_labels[i]
        validation_sets = None

        # Verify test set.
        test_sets = dataset.test_sets()
        assert len(test_sets) == 2
        for test_set in test_sets:
            assert len(test_set) == 1000
            assert isinstance(test_set, gb.ItemSet)
            for i, (src, dst, label) in enumerate(test_set):
                assert src == test_pairs[0][i]
                assert dst == test_pairs[1][i]
                assert label == test_labels[i]
        test_sets = None
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
            train_sets:
              - - type: paper
                  format: numpy
                  in_memory: true
                  path: {train_path}
              - - type: author
                  format: numpy
                  path: {train_path}
            validation_sets:
              - - type: paper
                  format: numpy
                  path: {validation_path}
              - - type: author
                  format: numpy
                  path: {validation_path}
            test_sets:
              - - type: paper
                  format: numpy
                  in_memory: false
                  path: {test_path}
              - - type: author
                  format: numpy
                  path: {test_path}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)

        # Verify train set.
        train_sets = dataset.train_sets()
        assert len(train_sets) == 2
        for train_set in train_sets:
            assert len(train_set) == 1000
            assert isinstance(train_set, gb.ItemSetDict)
            for i, item in enumerate(train_set):
                assert isinstance(item, dict)
                assert len(item) == 1
                key = list(item.keys())[0]
                assert key in ["paper", "author"]
                id, label = item[key]
                assert id == train_ids[i]
                assert label == train_labels[i]
        train_sets = None

        # Verify validation set.
        validation_sets = dataset.validation_sets()
        assert len(validation_sets) == 2
        for validation_set in validation_sets:
            assert len(validation_set) == 1000
            assert isinstance(train_set, gb.ItemSetDict)
            for i, item in enumerate(validation_set):
                assert isinstance(item, dict)
                assert len(item) == 1
                key = list(item.keys())[0]
                assert key in ["paper", "author"]
                id, label = item[key]
                assert id == validation_ids[i]
                assert label == validation_labels[i]
        validation_sets = None

        # Verify test set.
        test_sets = dataset.test_sets()
        assert len(test_sets) == 2
        for test_set in test_sets:
            assert len(test_set) == 1000
            assert isinstance(train_set, gb.ItemSetDict)
            for i, item in enumerate(test_set):
                assert isinstance(item, dict)
                assert len(item) == 1
                key = list(item.keys())[0]
                assert key in ["paper", "author"]
                id, label = item[key]
                assert id == test_ids[i]
                assert label == test_labels[i]
        test_sets = None
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
            train_sets:
              - - type: paper
                  format: numpy
                  in_memory: true
                  path: {train_path}
              - - type: author
                  format: numpy
                  path: {train_path}
            validation_sets:
              - - type: paper
                  format: numpy
                  path: {validation_path}
              - - type: author
                  format: numpy
                  path: {validation_path}
            test_sets:
              - - type: paper
                  format: numpy
                  in_memory: false
                  path: {test_path}
              - - type: author
                  format: numpy
                  path: {test_path}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)

        # Verify train set.
        train_sets = dataset.train_sets()
        assert len(train_sets) == 2
        for train_set in train_sets:
            assert len(train_set) == 1000
            assert isinstance(train_set, gb.ItemSetDict)
            for i, item in enumerate(train_set):
                assert isinstance(item, dict)
                assert len(item) == 1
                key = list(item.keys())[0]
                assert key in ["paper", "author"]
                src, dst, label = item[key]
                assert src == train_pairs[0][i]
                assert dst == train_pairs[1][i]
                assert label == train_labels[i]
        train_sets = None

        # Verify validation set.
        validation_sets = dataset.validation_sets()
        assert len(validation_sets) == 2
        for validation_set in validation_sets:
            assert len(validation_set) == 1000
            assert isinstance(train_set, gb.ItemSetDict)
            for i, item in enumerate(validation_set):
                assert isinstance(item, dict)
                assert len(item) == 1
                key = list(item.keys())[0]
                assert key in ["paper", "author"]
                src, dst, label = item[key]
                assert src == validation_pairs[0][i]
                assert dst == validation_pairs[1][i]
                assert label == validation_labels[i]
        validation_sets = None

        # Verify test set.
        test_sets = dataset.test_sets()
        assert len(test_sets) == 2
        for test_set in test_sets:
            assert len(test_set) == 1000
            assert isinstance(train_set, gb.ItemSetDict)
            for i, item in enumerate(test_set):
                assert isinstance(item, dict)
                assert len(item) == 1
                key = list(item.keys())[0]
                assert key in ["paper", "author"]
                src, dst, label = item[key]
                assert src == test_pairs[0][i]
                assert dst == test_pairs[1][i]
                assert label == test_labels[i]
        test_sets = None
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
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)

        # Verify feature data storage.
        feature_data = dataset.feature()
        assert len(feature_data) == 4

        # Verify node feature data.
        node_paper_feat = feature_data[("node", "paper", "feat")]
        assert isinstance(node_paper_feat, gb.TorchBasedFeatureStore)
        assert torch.equal(
            node_paper_feat.read(), torch.tensor(node_data_paper)
        )
        node_paper_label = feature_data[("node", "paper", "label")]
        assert isinstance(node_paper_label, gb.TorchBasedFeatureStore)
        assert torch.equal(
            node_paper_label.read(), torch.tensor(node_data_label)
        )

        # Verify edge feature data.
        edge_writes_feat = feature_data[("edge", "author:writes:paper", "feat")]
        assert isinstance(edge_writes_feat, gb.TorchBasedFeatureStore)
        assert torch.equal(
            edge_writes_feat.read(), torch.tensor(edge_data_writes)
        )
        edge_writes_label = feature_data[
            ("edge", "author:writes:paper", "label")
        ]
        assert isinstance(edge_writes_label, gb.TorchBasedFeatureStore)
        assert torch.equal(
            edge_writes_label.read(), torch.tensor(edge_data_label)
        )

        node_paper_feat = None
        node_paper_label = None
        edge_writes_feat = None
        edge_writes_label = None
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
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)

        # Verify feature data storage.
        feature_data = dataset.feature()
        assert len(feature_data) == 4

        # Verify node feature data.
        node_feat = feature_data[("node", None, "feat")]
        assert isinstance(node_feat, gb.TorchBasedFeatureStore)
        assert torch.equal(node_feat.read(), torch.tensor(node_data_feat))
        node_label = feature_data[("node", None, "label")]
        assert isinstance(node_label, gb.TorchBasedFeatureStore)
        assert torch.equal(node_label.read(), torch.tensor(node_data_label))

        # Verify edge feature data.
        edge_feat = feature_data[("edge", None, "feat")]
        assert isinstance(edge_feat, gb.TorchBasedFeatureStore)
        assert torch.equal(edge_feat.read(), torch.tensor(edge_data_feat))
        edge_label = feature_data[("edge", None, "label")]
        assert isinstance(edge_label, gb.TorchBasedFeatureStore)
        assert torch.equal(edge_label.read(), torch.tensor(edge_data_label))

        node_feat = None
        node_label = None
        edge_feat = None
        edge_label = None
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
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        with pytest.raises(
            pydantic.ValidationError,
            match="value is not a valid enumeration member",
        ):
            _ = gb.OnDiskDataset(yaml_file)


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
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)
        graph2 = dataset.graph()

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
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)
        graph2 = dataset.graph()

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
        num_classes = 10
        num_labels = 9
        yaml_content = f"""
            dataset_name: {dataset_name}
            num_classes: {num_classes}
            num_labels: {num_labels}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)
        assert dataset.dataset_name == dataset_name
        assert dataset.num_classes == num_classes
        assert dataset.num_labels == num_labels

        # Only dataset_name is specified.
        yaml_content = f"""
            dataset_name: {dataset_name}
        """
        yaml_file = os.path.join(test_dir, "test.yaml")
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        dataset = gb.OnDiskDataset(yaml_file)
        assert dataset.dataset_name == dataset_name
        assert dataset.num_classes is None
        assert dataset.num_labels is None
