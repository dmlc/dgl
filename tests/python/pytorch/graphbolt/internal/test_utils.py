import json
import os
import re
import tempfile
from functools import partial

import dgl.graphbolt as gb

import dgl.graphbolt.internal as internal
import numpy as np
import pandas as pd
import pytest
import torch


def test_read_torch_data():
    with tempfile.TemporaryDirectory() as test_dir:
        save_tensor = torch.tensor([[1, 2, 4], [2, 5, 3]])
        file_name = os.path.join(test_dir, "save_tensor.pt")
        torch.save(save_tensor, file_name)
        read_tensor = internal.utils._read_torch_data(file_name)
        assert torch.equal(save_tensor, read_tensor)
        save_tensor = read_tensor = None


@pytest.mark.parametrize("in_memory", [True, False])
def test_read_numpy_data(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        save_numpy = np.array([[1, 2, 4], [2, 5, 3]])
        file_name = os.path.join(test_dir, "save_numpy.npy")
        np.save(file_name, save_numpy)
        read_tensor = internal.utils._read_numpy_data(file_name, in_memory)
        assert torch.equal(torch.from_numpy(save_numpy), read_tensor)
        save_numpy = read_tensor = None


@pytest.mark.parametrize("fmt", ["torch", "numpy"])
def test_read_data(fmt):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.array([[1, 2, 4], [2, 5, 3]])
        type_name = "pt" if fmt == "torch" else "npy"
        file_name = os.path.join(test_dir, f"save_data.{type_name}")
        if fmt == "numpy":
            np.save(file_name, data)
        elif fmt == "torch":
            torch.save(torch.from_numpy(data), file_name)
        read_tensor = internal.read_data(file_name, fmt)
        assert torch.equal(torch.from_numpy(data), read_tensor)


@pytest.mark.parametrize(
    "data_fmt, save_fmt, contiguous",
    [
        ("torch", "torch", True),
        ("torch", "torch", False),
        ("torch", "numpy", True),
        ("torch", "numpy", False),
        ("numpy", "torch", True),
        ("numpy", "torch", False),
        ("numpy", "numpy", True),
        ("numpy", "numpy", False),
    ],
)
def test_save_data(data_fmt, save_fmt, contiguous):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.array([[1, 2, 4], [2, 5, 3]])
        if not contiguous:
            data = np.asfortranarray(data)
        tensor_data = torch.from_numpy(data)
        type_name = "pt" if save_fmt == "torch" else "npy"
        save_file_name = os.path.join(test_dir, f"save_data.{type_name}")
        # Step1. Save the data.
        if data_fmt == "torch":
            internal.save_data(tensor_data, save_file_name, save_fmt)
        elif data_fmt == "numpy":
            internal.save_data(data, save_file_name, save_fmt)

        # Step2. Load the data.
        if save_fmt == "torch":
            loaded_data = torch.load(save_file_name)
            assert loaded_data.is_contiguous()
            assert torch.equal(tensor_data, loaded_data)
        elif save_fmt == "numpy":
            loaded_data = np.load(save_file_name)
            # Checks if the loaded data is C-contiguous.
            assert loaded_data.flags["C_CONTIGUOUS"]
            assert np.array_equal(tensor_data.numpy(), loaded_data)

        data = tensor_data = loaded_data = None


@pytest.mark.parametrize("fmt", ["torch", "numpy"])
def test_get_npy_dim(fmt):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.array([[1, 2, 4], [2, 5, 3]])
        type_name = "pt" if fmt == "torch" else "npy"
        file_name = os.path.join(test_dir, f"save_data.{type_name}")
        if fmt == "numpy":
            np.save(file_name, data)
            assert internal.get_npy_dim(file_name) == 2
        elif fmt == "torch":
            torch.save(torch.from_numpy(data), file_name)
            with pytest.raises(ValueError):
                internal.get_npy_dim(file_name)
        data = None


@pytest.mark.parametrize("data_fmt", ["numpy", "torch"])
@pytest.mark.parametrize("save_fmt", ["numpy", "torch"])
@pytest.mark.parametrize("is_feature", [True, False])
def test_copy_or_convert_data(data_fmt, save_fmt, is_feature):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.arange(10)
        tensor_data = torch.from_numpy(data)
        in_type_name = "npy" if data_fmt == "numpy" else "pt"
        input_path = os.path.join(test_dir, f"data.{in_type_name}")
        out_type_name = "npy" if save_fmt == "numpy" else "pt"
        output_path = os.path.join(test_dir, f"out_data.{out_type_name}")
        if data_fmt == "numpy":
            np.save(input_path, data)
        else:
            torch.save(tensor_data, input_path)
        if save_fmt == "torch":
            with pytest.raises(AssertionError):
                internal.copy_or_convert_data(
                    input_path,
                    output_path,
                    data_fmt,
                    save_fmt,
                    is_feature=is_feature,
                )
        else:
            internal.copy_or_convert_data(
                input_path,
                output_path,
                data_fmt,
                save_fmt,
                is_feature=is_feature,
            )
        if is_feature:
            data = data.reshape(-1, 1)
            tensor_data = tensor_data.reshape(-1, 1)
        if save_fmt == "numpy":
            out_data = np.load(output_path)
            assert (data == out_data).all()

        data = None
        tensor_data = None
        out_data = None


@pytest.mark.parametrize("edge_fmt", ["csv", "numpy"])
def test_read_edges(edge_fmt):
    with tempfile.TemporaryDirectory() as test_dir:
        num_nodes = 40
        num_edges = 200
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors], axis=1)
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        if edge_fmt == "csv":
            # Wrtie into edges/edge.csv
            edges = pd.DataFrame(edges, columns=["src", "dst"])
            edge_path = os.path.join("edges", "edge.csv")
            edges.to_csv(
                os.path.join(test_dir, edge_path),
                index=False,
                header=False,
            )
        else:
            # Wrtie into edges/edge.npy
            edges = edges.T
            edge_path = os.path.join("edges", "edge.npy")
            np.save(os.path.join(test_dir, edge_path), edges)
        src, dst = internal.read_edges(test_dir, edge_fmt, edge_path)
        assert src.all() == nodes.all()
        assert dst.all() == neighbors.all()


def test_read_edges_error():
    # 1. Unsupported file format.
    with pytest.raises(
        AssertionError,
        match="`numpy` or `csv` is expected when reading edges but got `fake-type`.",
    ):
        internal.read_edges("test_dir", "fake-type", "edge_path")

    # 2. Unexpected shape of numpy array
    with tempfile.TemporaryDirectory() as test_dir:
        num_nodes = 40
        num_edges = 200
        nodes = np.repeat(np.arange(num_nodes), 5)
        neighbors = np.random.randint(0, num_nodes, size=(num_edges))
        edges = np.stack([nodes, neighbors, nodes], axis=1)
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        # Wrtie into edges/edge.npy
        edges = edges.T
        edge_path = os.path.join("edges", "edge.npy")
        np.save(os.path.join(test_dir, edge_path), edges)
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "The shape of edges should be (2, N), but got torch.Size([3, 200])."
            ),
        ):
            internal.read_edges(test_dir, "numpy", edge_path)


def test_calculate_file_hash():
    with tempfile.TemporaryDirectory() as test_dir:
        test_file_path = os.path.join(test_dir, "test.txt")
        with open(test_file_path, "w") as file:
            file.write("test content")
        hash_value = internal.calculate_file_hash(
            test_file_path, hash_algo="md5"
        )
        expected_hash_value = "9473fdd0d880a43c21b7778d34872157"
        assert expected_hash_value == hash_value
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Hash algorithm must be one of: ['md5', 'sha1', 'sha224', "
                + "'sha256', 'sha384', 'sha512'], but got `fake`."
            ),
        ):
            hash_value = internal.calculate_file_hash(
                test_file_path, hash_algo="fake"
            )


def test_calculate_dir_hash():
    with tempfile.TemporaryDirectory() as test_dir:
        test_file_path_1 = os.path.join(test_dir, "test_1.txt")
        test_file_path_2 = os.path.join(test_dir, "test_2.txt")
        with open(test_file_path_1, "w") as file:
            file.write("test content")
        with open(test_file_path_2, "w") as file:
            file.write("test contents of directory")
        hash_value = internal.calculate_dir_hash(test_dir, hash_algo="md5")
        expected_hash_value = [
            "56e708a2bdf92887d4a7f25cbc13c555",
            "9473fdd0d880a43c21b7778d34872157",
        ]
        assert len(hash_value) == 2
        for val in hash_value.values():
            assert val in expected_hash_value


def test_check_dataset_change():
    with tempfile.TemporaryDirectory() as test_dir:
        # Generate directory and record its hash value.
        test_file_path_1 = os.path.join(test_dir, "test_1.txt")
        test_file_path_2 = os.path.join(test_dir, "test_2.txt")
        with open(test_file_path_1, "w") as file:
            file.write("test content")
        with open(test_file_path_2, "w") as file:
            file.write("test contents of directory")
        hash_value = internal.calculate_dir_hash(test_dir, hash_algo="md5")
        hash_value_file = "dataset_hash_value.txt"
        hash_value_file_paht = os.path.join(
            test_dir, "preprocessed", hash_value_file
        )
        os.makedirs(os.path.join(test_dir, "preprocessed"), exist_ok=True)
        with open(hash_value_file_paht, "w") as file:
            file.write(json.dumps(hash_value, indent=4))

        # Modify the content of a file.
        with open(test_file_path_2, "w") as file:
            file.write("test contents of directory changed")

        assert internal.check_dataset_change(test_dir, "preprocessed")


def test_numpy_save_aligned():
    assert_equal = partial(torch.testing.assert_close, rtol=0, atol=0)
    a = torch.randn(1024, dtype=torch.float32)  # 4096 bytes
    with tempfile.TemporaryDirectory() as test_dir:
        aligned_path = os.path.join(test_dir, "aligned.npy")
        gb.numpy_save_aligned(aligned_path, a.numpy())

        nonaligned_path = os.path.join(test_dir, "nonaligned.npy")
        np.save(nonaligned_path, a.numpy())

        assert_equal(np.load(aligned_path), np.load(nonaligned_path))
        # The size of the file should be 4K (aligned header) + 4K (tensor).
        assert os.path.getsize(aligned_path) == 4096 * 2
