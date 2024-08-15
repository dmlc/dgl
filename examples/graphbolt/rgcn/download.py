import argparse, hashlib, os, shutil, tarfile, yaml
import subprocess
import urllib.request as ur

import dgl.graphbolt as gb
import numpy as np
from tqdm import tqdm

GBFACTOR = float(1 << 30)


def _get_size(file_path, node_name):
    if "full" in file_path:
        return num_nodes["full"][node_name]
    if "large" in file_path:
        return num_nodes["large"][node_name]
    path = f"{file_path}/processed/{node_name}/{node_name}_id_index_mapping.npy"
    array = np.load(path, allow_pickle=True)
    return len(array.item())


def build_yaml_helper(path, in_memory=True):
    data = {
        "graph": {
            "nodes": [
                {"num": _get_size(path, "paper"), "type": "paper"},
                {"num": _get_size(path, "author"), "type": "author"},
                {"num": _get_size(path, "institute"), "type": "institution"},
                {"num": _get_size(path, "fos"), "type": "field_of_study"},
            ],
            "edges": [
                {
                    "path": "edges/author__affiliated_to__institute.npy",
                    "type": "author:affiliated_to:institution",
                    "format": "numpy",
                },
                {
                    "path": "edges/paper__written_by__author.npy",
                    "type": "paper:written_by:author",
                    "format": "numpy",
                },
                {
                    "path": "edges/paper__cites__paper.npy",
                    "type": "paper:cites:paper",
                    "format": "numpy",
                },
                {
                    "path": "edges/paper__topic__fos.npy",
                    "type": "paper:has_topic:field_of_study",
                    "format": "numpy",
                },
            ],
        },
        "tasks": [
            {
                "num_classes": 19,
                "validation_set": [
                    {
                        "data": [
                            {
                                "in_memory": in_memory,
                                "path": "set/validation_indices.npy",
                                "name": "seeds",
                                "format": "numpy",
                            },
                            {
                                "in_memory": in_memory,
                                "path": "set/validation_labels.npy",
                                "name": "labels",
                                "format": "numpy",
                            },
                        ],
                        "type": "paper",
                    }
                ],
                "name": "node_classification",
                "train_set": [
                    {
                        "data": [
                            {
                                "in_memory": in_memory,
                                "path": "set/train_indices.npy",
                                "name": "seeds",
                                "format": "numpy",
                            },
                            {
                                "in_memory": in_memory,
                                "path": "set/train_labels.npy",
                                "name": "labels",
                                "format": "numpy",
                            },
                        ],
                        "type": "paper",
                    }
                ],
                "test_set": [
                    {
                        "data": [
                            {
                                "in_memory": in_memory,
                                "path": "set/test_indices.npy",
                                "name": "seeds",
                                "format": "numpy",
                            },
                            {
                                "in_memory": in_memory,
                                "path": "set/test_labels.npy",
                                "name": "labels",
                                "format": "numpy",
                            },
                        ],
                        "type": "paper",
                    }
                ],
            }
        ],
        "feature_data": [
            {
                "domain": "node",
                "name": "feat",
                "format": "numpy",
                "in_memory": in_memory,
                "path": "data/paper_feat.npy",
                "type": "paper",
            },
            {
                "domain": "node",
                "name": "label",
                "format": "numpy",
                "in_memory": in_memory,
                "path": "data/paper_label_19.npy",
                "type": "paper",
            },
            {
                "domain": "node",
                "name": "feat",
                "format": "numpy",
                "in_memory": in_memory,
                "path": "data/author_feat.npy",
                "type": "author",
            },
            {
                "domain": "node",
                "name": "feat",
                "format": "numpy",
                "in_memory": in_memory,
                "path": "data/institute_feat.npy",
                "type": "institute",
            },
            {
                "domain": "node",
                "name": "feat",
                "format": "numpy",
                "in_memory": in_memory,
                "path": "data/fos_feat.npy",
                "type": "fos",
            },
            {
                "domain": "node",
                "name": "feat",
                "format": "numpy",
                "in_memory": in_memory,
                "path": "data/author_feat.npy",
                "type": "author",
            },
        ],
        "dataset_name": os.path.basename(path),
    }

    return data


def build_yaml(original_path, current_path):
    if "large" in current_path or "full" in current_path:
        data = build_yaml_helper(original_path, in_memory=False)
    else:
        data = build_yaml_helper(original_path)
    with open(f"{current_path}/metadata.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def decide_download(url):
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


def check_md5sum(dataset_type, dataset_size, filename):
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
    if dataset_size in ["large", "full"]:
        command = f"./download_{dataset_size}_igbh.sh"
        subprocess.run(["bash", command], check=True, text=True)
        shutil.move(src=f"igb-{dataset_type}-{dataset_size}", dst=f"{path}")
        return path + "/" + "igb-" + dataset_type + "-" + dataset_size
    else:
        output_directory = path
        if not os.path.exists(
            output_directory
            + "igb_"
            + dataset_type
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
                    + dataset_type
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
                "Downloaded" + " igb_" + dataset_type + "_" + dataset_size,
                end=" ->",
            )
            check_md5sum(dataset_type, dataset_size, filename)
        else:
            print(
                "The file igb_"
                + dataset_type
                + "_"
                + dataset_size
                + ".tar.gz already exists, directly extracting..."
            )
            filename = (
                path + "/igb_" + dataset_type + "_" + dataset_size + ".tar.gz"
            )
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
            output_directory + "/" + "igb-" + dataset_type + "-" + dataset_size,
        )
        return (
            output_directory + "/" + "igb-" + dataset_type + "-" + dataset_size
        )


num_nodes = {
    "full": {
        "paper": 269346174,
        "author": 277220883,
        "institute": 26918,
        "fos": 712960,
    },
    "large": {
        "paper": 100000000,
        "author": 116959896,
        "institute": 26524,
        "fos": 649707,
    },
    "medium": {
        "paper": 10000000,
        "author": 15544654,
        "institute": 23256,
        "fos": 415054,
    },
    "small": {
        "paper": 1000000,
        "author": 1926066,
        "institute": 14751,
        "fos": 190449,
    },
    "tiny": {
        "paper": 100000,
        "author": 357041,
        "institute": 8738,
        "fos": 84220,
    },
}

num_edges = {
    "full": {
        "paper__cites__paper": 3996442004,
        "paper__written_by__author": 716761549,
        "paper__topic__fos": 1050280600,
        "author__affiliated_to__institute": 48521486,
    },
    "large": {
        "paper__cites__paper": 1223571364,
        "paper__written_by__author": 289502107,
        "paper__topic__fos": 457577294,
        "author__affiliated_to__institute": 34099660,
    },
    "medium": {
        "paper__cites__paper": 120077694,
        "paper__written_by__author": 39854592,
        "paper__topic__fos": 68510495,
        "author__affiliated_to__institute": 11049412,
    },
    "small": {
        "paper__cites__paper": 12070502,
        "paper__written_by__author": 4553516,
        "paper__topic__fos": 7234122,
        "author__affiliated_to__institute": 1630476,
    },
    "tiny": {
        "paper__cites__paper": 447416,
        "paper__written_by__author": 471443,
        "paper__topic__fos": 718445,
        "author__affiliated_to__institute": 325410,
    },
}


def split_data(label_path, set_dir, dataset_size):
    # labels = np.memmap(label_path, dtype='int32', mode='r',  shape=(num_nodes[dataset_size]["paper"], 1))
    labels = np.load(label_path)

    total_samples = len(labels)
    train_end = int(0.8 * total_samples)
    validation_end = int(0.9 * total_samples)

    indices = np.arange(total_samples)
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]
    print(indices)
    print(train_indices)
    print(validation_indices)
    print(test_indices)

    train_labels = labels[:train_end]
    validation_labels = labels[train_end:validation_end]
    test_labels = labels[validation_end:]
    print(train_labels, len(train_labels))
    print(validation_labels, len(validation_labels))
    print(test_labels, len(test_labels))

    gb.numpy_save_aligned(f"{set_dir}/train_indices.npy", train_indices)
    gb.numpy_save_aligned(
        f"{set_dir}/validation_indices.npy", validation_indices
    )
    gb.numpy_save_aligned(f"{set_dir}/test_indices.npy", test_indices)
    gb.numpy_save_aligned(f"{set_dir}/train_labels.npy", train_labels)
    gb.numpy_save_aligned(f"{set_dir}/validation_labels.npy", validation_labels)
    gb.numpy_save_aligned(f"{set_dir}/test_labels.npy", test_labels)


def add_edges(edges, source, dest, dataset_size):
    for edge in edges:
        print(f"\t Processing {edge} edge...")

        old_edge_path = source + "/" + edge + "/" + "edge_index.npy"
        new_edge_path = dest + "/" + edge + ".npy"
        os.rename(src=old_edge_path, dst=new_edge_path)

        # edge_array = np.memmap(new_edge_path, dtype='int32', mode='r',  shape=(num_edges[dataset_size][edge], 2))
        edge_array = np.load(new_edge_path)
        new_edge_array = edge_array.transpose()

        assert new_edge_array.shape == (2, num_edges[dataset_size][edge])

        np.save(new_edge_path, new_edge_array)


def process_feat(file_path, node_name, dataset_size):
    # array = np.memmap(file_path, dtype='float32', mode='r',  shape=(num_nodes[dataset_size][node_name], 1024))
    array = np.load(file_path)
    assert array.shape == (num_nodes[dataset_size][node_name], 1024)
    gb.numpy_save_aligned(file_path, array)

    # Assert the shape and elements of the array are correct
    # new_array = np.memmap(file_path, dtype='float32', mode='r',  shape=(num_nodes[dataset_size][node_name], 1024))
    new_array = np.load(file_path)
    assert array.shape == (num_nodes[dataset_size][node_name], 1024)
    assert np.array_equal(array, new_array)


def process_label(file_path, num_class, dataset_size):
    if (
        num_class == 2983 and dataset_size == "full"
    ):  # only this case label number changes
        # array = np.memmap(file_path, dtype='int32', mode='r',  shape=(227130858, 1))
        array = np.load(file_path)
        assert array.shape == (227130858, 1) or array.shape == (227130858,)
    else:
        # array = np.memmap(file_path, dtype='int32', mode='r',  shape=(num_nodes[dataset_size]["paper"], 1))
        array = np.load(file_path)
        assert array.shape == (
            num_nodes[dataset_size]["paper"],
            1,
        ) or array.shape == (num_nodes[dataset_size]["paper"],)

    gb.numpy_save_aligned(file_path, array)

    # Assert the shape and elements of the array are correct
    if num_class == 2983 and dataset_size == "full":
        # new_array = np.memmap(file_path, dtype='int32', mode='r',  shape=(227130858, 1))
        new_array = np.load(file_path)
        assert new_array.shape == (227130858, 1) or new_array.shape == (
            227130858,
        )
        assert np.array_equal(array, new_array)
    else:
        # new_array = np.memmap(file_path, dtype='int32', mode='r',  shape=(num_nodes[dataset_size]["paper"], 1))
        new_array = np.load(file_path)
        assert new_array.shape == (
            num_nodes[dataset_size]["paper"],
            1,
        ) or new_array.shape == (num_nodes[dataset_size]["paper"],)
        assert np.array_equal(array, new_array)


def add_nodes(nodes, source, dest, dataset_size):
    for node in nodes:
        print(f"\t Processing {node} node feature...")
        old_node_path = source + "/" + node + "/" + "node_feat.npy"
        new_node_path = dest + "/" + node + "_feat.npy"
        os.rename(src=old_node_path, dst=new_node_path)
        process_feat(
            file_path=new_node_path, node_name=node, dataset_size=dataset_size
        )
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
                file_path=new_label_path_19,
                num_class=2983,
                dataset_size=dataset_size,
            )

    return new_label_path_19, new_label_path_2K


def process_dataset(path, dataset_size):
    print(f"Starting to process the {dataset_size} dataset...")

    # Make the directory for processed dataset
    processed_dir = path + "-seeds"
    os.makedirs(name=processed_dir, exist_ok=True)
    original_path = path + "/" + "processed"

    # Step 1: Move Nodes files
    print("Processing Node files...")
    node_dir = processed_dir + "/" + "data"
    os.makedirs(name=node_dir, exist_ok=True)
    # These are the four nodes in this citation network
    nodes = ["paper", "author", "fos", "institute"]
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
        label_path=label_file_19, set_dir=set_dir, dataset_size=dataset_size
    )

    # Step 3: Move edge files
    print("Processing Edge files...")
    edge_dir = processed_dir + "/" + "edges"
    os.makedirs(name=edge_dir, exist_ok=True)
    # These are the four edges in this citation network
    edges = [
        "paper__cites__paper",
        "paper__written_by__author",
        "paper__topic__fos",
        "author__affiliated_to__institute",
    ]
    add_edges(
        edges=edges,
        source=original_path,
        dest=edge_dir,
        dataset_size=dataset_size,
    )

    # Step 4: Build the yaml file
    print("Building yaml file...")
    build_yaml(original_path=path, current_path=processed_dir)

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
        default="heterogeneous",
        choices=["homogeneous", "heterogeneous"],
        help="dataset type",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large", "full"],
        help="size of the datasets",
    )
    args = parser.parse_args()
    path = download_dataset(
        path=args.path, dataset_type=args.type, dataset_size=args.size
    )
    process_dataset(path=path, dataset_size=args.size)
