import argparse, hashlib, os, shutil, tarfile, yaml
import subprocess
import urllib.request as ur

import dgl.graphbolt as gb
import numpy as np
from tqdm import tqdm

GBFACTOR = 1 << 30


def build_yaml_helper(path, dataset_size, in_memory=True):
    """The stirng to build the yaml file. (Still need modification)"""

    data = {
        "dataset_name": os.path.basename(path),
        "feature_data": [
            {
                "domain": "node",
                "format": "numpy",
                "in_memory": in_memory,
                "name": "feat",
                "path": "data/paper_feat.npy",
            }
        ],
        "graph": {
            "edges": [
                {
                    "format": "numpy",
                    "path": "edges/paper__cites__paper.npy",
                    # "type": "paper:cites:paper"
                },
                # {
                #     "format": "numpy",
                #     "path": "edges/paper__cites__paper_rev.npy",
                #     "type": "paper:rev_cites:paper"
                # },
            ],
            "nodes": [{"num": num_nodes[dataset_size]["paper"]}],
        },
        "tasks": [
            {
                "name": "node_classification",
                "num_classes": 19,
                "test_set": [
                    {
                        "data": [
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "seeds",
                                "path": "set/test_indices_19.npy",
                            },
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "labels",
                                "path": "set/test_labels_19.npy",
                            },
                        ]
                    }
                ],
                "train_set": [
                    {
                        "data": [
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "seeds",
                                "path": "set/train_indices_19.npy",
                            },
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "labels",
                                "path": "set/train_labels_19.npy",
                            },
                        ]
                    }
                ],
                "validation_set": [
                    {
                        "data": [
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "seeds",
                                "path": "set/validation_indices_19.npy",
                            },
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "labels",
                                "path": "set/validation_labels_19.npy",
                            },
                        ]
                    }
                ],
            },
            {
                "name": "node_classification_2K",
                "num_classes": 2983,
                "test_set": [
                    {
                        "data": [
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "seeds",
                                "path": "set/test_indices_2983.npy",
                            },
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "labels",
                                "path": "set/test_labels_2983.npy",
                            },
                        ]
                    }
                ],
                "train_set": [
                    {
                        "data": [
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "seeds",
                                "path": "set/train_indices_2983.npy",
                            },
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "labels",
                                "path": "set/train_labels_2983.npy",
                            },
                        ]
                    }
                ],
                "validation_set": [
                    {
                        "data": [
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "seeds",
                                "path": "set/validation_indices_2983.npy",
                            },
                            {
                                "format": "numpy",
                                "in_memory": in_memory,
                                "name": "labels",
                                "path": "set/validation_labels_2983.npy",
                            },
                        ]
                    }
                ],
            },
        ],
    }

    return data


def build_yaml(original_path, current_path, dataset_size):
    """This build the yaml file differently based on the dataset size.
    The two large datasets are put in disk while the other three smaller versions are in memory.
    """
    if "large" == dataset_size or "full" == dataset_size:
        data = build_yaml_helper(
            path=original_path, dataset_size=dataset_size, in_memory=False
        )
    else:
        data = build_yaml_helper(path=original_path, dataset_size=dataset_size)
    with open(f"{current_path}/metadata.yaml", "w") as file:
        yaml.dump(data=data, stream=file, default_flow_style=False)


dataset_urls = {
    "homogeneous": {
        "tiny": "https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_tiny.tar.gz",
        "small": "https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_small.tar.gz",
        "medium": "https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_medium.tar.gz",
    },
    "heterogeneous": {
        "tiny": "https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_tiny.tar.gz",
        "small": "https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_small.tar.gz",
        "medium": "https://igb-public-awsopen.s3.amazonaws.com/igb-heterogeneous/igb_heterogeneous_medium.tar.gz",
    },
}


md5checksums = {
    "homogeneous": {
        "tiny": "34856534da55419b316d620e2d5b21be",
        "small": "6781c699723529902ace0a95cafe6fe4",
        "medium": "4640df4ceee46851fd18c0a44ddcc622",
    },
    "heterogeneous": {
        "tiny": "83fbc1091497ff92cf20afe82fae0ade",
        "small": "2f42077be60a074aec24f7c60089e1bd",
        "medium": "7f0df4296eca36553ff3a6a63abbd347",
    },
}


def decide_download(url):
    """An interactive command line to confirm download."""
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"]) / GBFACTOR
    ### confirm if larger than 1GB
    if size > 1:
        return (
            input(
                "This will download %.2fGB. Will you proceed? (y/N) " % (size)
            ).lower()
            == "y"
        )
    else:
        return True


def check_md5sum(dataset_type, dataset_size, filename):
    """This is for checking the data correctness of the downloaded datasets."""
    original_md5 = md5checksums[dataset_type][dataset_size]

    with open(filename, "rb") as file_to_check:
        data = file_to_check.read()
        md5_returned = hashlib.md5(data).hexdigest()

    if original_md5 == md5_returned:
        print(" md5sum verified.")
        return
    else:
        os.remove(filename)
        raise Exception(" md5sum verification failed!.")


def download_dataset(path, dataset_type, dataset_size):
    """This is the script to download all the related datasets."""
    _dataset_type = dataset_type[:3]

    # For large datasets, use the two shell scripts to download.
    if dataset_size in ["large", "full"]:
        command = f"./download_{dataset_size}_igbh.sh"
        subprocess.run(["bash", command], check=True, text=True)
        shutil.move(src=f"igb-{_dataset_type}-{dataset_size}", dst=f"{path}")
        return path + "/" + "igb-" + _dataset_type + "-" + dataset_size
    # For the three smaller version, use the url to download.
    else:
        output_directory = path
        if not os.path.exists(
            output_directory
            + "igb_"
            + _dataset_type
            + "_"
            + dataset_size
            + ".tar.gz"
        ):
            url = dataset_urls[dataset_type][dataset_size]
            if decide_download(url):
                data = ur.urlopen(url)
                size = int(data.info()["Content-Length"])
                chunk_size = 1024 * 1024
                num_iter = int(size / chunk_size) + 2
                downloaded_size = 0
                filename = (
                    path
                    + "/igb_"
                    + _dataset_type
                    + "_"
                    + dataset_size
                    + ".tar.gz"
                )
                with open(filename, "wb") as f:
                    pbar = tqdm(range(num_iter))
                    for i in pbar:
                        chunk = data.read(chunk_size)
                        downloaded_size += len(chunk)
                        pbar.set_description(
                            "Downloaded {:.2f} GB".format(
                                float(downloaded_size) / GBFACTOR
                            )
                        )
                        f.write(chunk)
            print(
                "Downloaded" + " igb_" + _dataset_type + "_" + dataset_size,
                end=" ->",
            )
            check_md5sum(dataset_type, dataset_size, filename)
        else:  # No need to download the tar file again if it is already downloaded.
            print(
                "The file igb_"
                + _dataset_type
                + "_"
                + dataset_size
                + ".tar.gz already exists, directly extracting..."
            )
            filename = (
                path + "/igb_" + _dataset_type + "_" + dataset_size + ".tar.gz"
            )
        # Extract the tar file
        file = tarfile.open(filename)
        file.extractall(output_directory)
        file.close()
        size = 0
        for path, dirs, files in os.walk(output_directory + "/" + dataset_size):
            for f in files:
                fp = os.path.join(path, f)
                size += os.path.getsize(fp)
        print("Final dataset size {:.2f} GB.".format(size / GBFACTOR))
        # os.remove(filename)
        os.rename(
            output_directory + "/" + dataset_size,
            output_directory
            + "/"
            + "igb-"
            + _dataset_type
            + "-"
            + dataset_size,
        )
        return (
            output_directory + "/" + "igb-" + _dataset_type + "-" + dataset_size
        )


num_nodes = {
    "full": {
        "paper": 269346174,
    },
    "large": {
        "paper": 100000000,
    },
    "medium": {
        "paper": 10000000,
    },
    "small": {
        "paper": 1000000,
    },
    "tiny": {
        "paper": 100000,
    },
}

num_edges = {
    "full": {
        "paper__cites__paper": 3996442004,
    },
    "large": {
        "paper__cites__paper": 1223571364,
    },
    "medium": {
        "paper__cites__paper": 120077694,
    },
    "small": {
        "paper__cites__paper": 12070502,
    },
    "tiny": {
        "paper__cites__paper": 447416,
    },
}


def split_data(label_path, set_dir, dataset_size, class_num):
    """This is for splitting the labels into three sets: train, validation, and test sets."""
    # labels = np.memmap(label_path, dtype='int32', mode='r',  shape=(num_nodes[dataset_size]["paper"], 1))
    labels = np.load(label_path)

    total_samples = len(labels)
    train_end = int(0.6 * total_samples)
    validation_end = int(0.8 * total_samples)

    indices = np.arange(total_samples)
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]
    print(indices)
    print(train_indices)
    print(validation_indices)
    print(test_indices)

    train_labels = labels[:train_end].astype(np.int64)
    validation_labels = labels[train_end:validation_end].astype(np.int64)
    test_labels = labels[validation_end:].astype(np.int64)
    print(train_labels, len(train_labels))
    print(validation_labels, len(validation_labels))
    print(test_labels, len(test_labels))

    gb.numpy_save_aligned(
        f"{set_dir}/train_indices_{class_num}.npy", train_indices
    )
    gb.numpy_save_aligned(
        f"{set_dir}/validation_indices_{class_num}.npy", validation_indices
    )
    gb.numpy_save_aligned(
        f"{set_dir}/test_indices_{class_num}.npy", test_indices
    )
    gb.numpy_save_aligned(
        f"{set_dir}/train_labels_{class_num}.npy", train_labels
    )
    gb.numpy_save_aligned(
        f"{set_dir}/validation_labels_{class_num}.npy", validation_labels
    )
    gb.numpy_save_aligned(f"{set_dir}/test_labels_{class_num}.npy", test_labels)


def add_edges(edges, source, dest, dataset_size):
    """This is for processing the edges in the graph and convert them to correct shape."""
    for edge in edges:
        print(f"\t Processing {edge} edge...")

        old_edge_path = source + "/" + edge + "/" + "edge_index.npy"
        new_edge_path = dest + "/" + edge + ".npy"
        rev_edge_path = dest + "/" + edge + "_rev.npy"
        os.rename(src=old_edge_path, dst=new_edge_path)

        # edge_array = np.memmap(new_edge_path, dtype='int32', mode='r',  shape=(num_edges[dataset_size][edge], 2))
        edge_array = np.load(new_edge_path)
        new_edge_array = edge_array.transpose()
        rev_edge_array = new_edge_array[:, ::-1]

        assert new_edge_array.shape == (2, num_edges[dataset_size][edge])
        assert rev_edge_array.shape == (2, num_edges[dataset_size][edge])
        assert np.array_equal(new_edge_array, edge_array.transpose())
        assert np.array_equal(rev_edge_array, new_edge_array[:, ::-1])

        gb.numpy_save_aligned(new_edge_path, new_edge_array)
        gb.numpy_save_aligned(rev_edge_path, rev_edge_array)


def process_feat(file_path, node_name, dataset_size):
    """This is for processing the node features."""
    # array = np.memmap(file_path, dtype='float32', mode='r',  shape=(num_nodes[dataset_size][node_name], 1024))
    array = np.load(file_path)
    assert array.shape == (num_nodes[dataset_size][node_name], 1024)
    gb.numpy_save_aligned(file_path, array)

    # Assert the shape and elements of the array are correct
    # new_array = np.memmap(file_path, dtype='float32', mode='r',  shape=(num_nodes[dataset_size][node_name], 1024))
    new_array = np.load(file_path)
    assert new_array.shape == (num_nodes[dataset_size][node_name], 1024)
    assert np.array_equal(array, new_array)


def process_label(file_path, num_class, dataset_size):
    """This is for processing the node labels."""
    if (
        num_class == 2983 and dataset_size == "full"
    ):  # only this case label number changes
        # array = np.memmap(file_path, dtype='int32', mode='r',  shape=(227130858, 1))
        array = np.load(file_path)
        assert array.shape[0] == 227130858
    else:
        # array = np.memmap(file_path, dtype='int32', mode='r',  shape=(num_nodes[dataset_size]["paper"], 1))
        array = np.load(file_path)
        assert array.shape[0] == num_nodes[dataset_size]["paper"]

    gb.numpy_save_aligned(file_path, array)

    # Assert the shape and elements of the array are correct
    if num_class == 2983 and dataset_size == "full":
        # new_array = np.memmap(file_path, dtype='int32', mode='r',  shape=(227130858, 1))
        new_array = np.load(file_path)
        assert new_array.shape[0] == 227130858
        assert np.array_equal(array, new_array)
    else:
        # new_array = np.memmap(file_path, dtype='int32', mode='r',  shape=(num_nodes[dataset_size]["paper"], 1))
        new_array = np.load(file_path)
        assert new_array.shape[0] == num_nodes[dataset_size]["paper"]
        assert np.array_equal(array, new_array)


def add_nodes(nodes, source, dest, dataset_size):
    """This is for processing the nodes in the graph and store them in correct format."""
    for node in nodes:
        print(f"\t Processing {node} node feature...")
        old_node_path = source + "/" + node + "/" + "node_feat.npy"
        new_node_path = dest + "/" + node + "_feat.npy"
        os.rename(src=old_node_path, dst=new_node_path)
        process_feat(
            file_path=new_node_path, node_name=node, dataset_size=dataset_size
        )
        # If the node is a paper type, process the labels
        if node == "paper":
            print(f"\t Processing {node} labels...")
            old_label_path_19 = source + "/" + node + "/" + "node_label_19.npy"
            new_label_path_19 = dest + "/" + "paper_label_19.npy"
            os.rename(src=old_label_path_19, dst=new_label_path_19)
            process_label(
                file_path=new_label_path_19,
                num_class=19,
                dataset_size=dataset_size,
            )

            old_label_path_2K = source + "/" + node + "/" + "node_label_2K.npy"
            new_label_path_2K = dest + "/" + "paper_label_2K.npy"
            os.rename(src=old_label_path_2K, dst=new_label_path_2K)
            process_label(
                file_path=new_label_path_2K,
                num_class=2983,
                dataset_size=dataset_size,
            )

    return new_label_path_19, new_label_path_2K


def process_dataset(path, dataset_size):
    print(f"Starting to process the {dataset_size} dataset...")

    # Step 0: Make the directory for processed dataset
    processed_dir = path + "-seeds"
    os.makedirs(name=processed_dir, exist_ok=True)
    original_path = path + "/" + "processed"

    # Step 1: Move Nodes files
    print("Processing Node files...")
    node_dir = processed_dir + "/" + "data"
    os.makedirs(name=node_dir, exist_ok=True)
    # These are the one node in this homogeneous citation network
    nodes = ["paper"]
    label_file_19, label_file_2K = add_nodes(
        nodes=nodes,
        source=original_path,
        dest=node_dir,
        dataset_size=dataset_size,
    )

    # Step 2: Create labels
    print("Processing train/valid/test files...")
    set_dir = processed_dir + "/" + "set"
    os.makedirs(name=set_dir, exist_ok=True)
    split_data(
        label_path=label_file_19,
        set_dir=set_dir,
        dataset_size=dataset_size,
        class_num=19,
    )
    split_data(
        label_path=label_file_2K,
        set_dir=set_dir,
        dataset_size=dataset_size,
        class_num=2983,
    )

    # Step 3: Move edge files
    print("Processing Edge files...")
    edge_dir = processed_dir + "/" + "edges"
    os.makedirs(name=edge_dir, exist_ok=True)
    # These are the one edge in this homogeneous citation network
    edges = [
        "paper__cites__paper",
    ]
    add_edges(
        edges=edges,
        source=original_path,
        dest=edge_dir,
        dataset_size=dataset_size,
    )

    # Step 4: Build the yaml file
    print("Building yaml file...")
    build_yaml(
        original_path=path,
        current_path=processed_dir,
        dataset_size=dataset_size,
    )

    # shutil.rmtree(path)
    print(f"Finished processing the {dataset_size} dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="datasets/",
        help="path to store the datasets",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="homogeneous",
        choices=["homogeneous", "heterogeneous"],
        help="dataset type",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium"],
        help="size of the datasets",
    )
    args = parser.parse_args()
    path = download_dataset(
        path=args.path, dataset_type=args.type, dataset_size=args.size
    )
    process_dataset(path=path, dataset_size=args.size)
