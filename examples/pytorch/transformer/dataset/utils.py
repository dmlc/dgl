import os

import numpy as np
import torch as th

from dgl.data.utils import *

_urls = {
    "wmt": "https://data.dgl.ai/dataset/wmt14bpe_de_en.zip",
    "scripts": "https://data.dgl.ai/dataset/transformer_scripts.zip",
}


def prepare_dataset(dataset_name):
    "download and generate datasets"
    script_dir = os.path.join("scripts")
    if not os.path.exists(script_dir):
        download(_urls["scripts"], path="scripts.zip")
        extract_archive("scripts.zip", "scripts")

    directory = os.path.join("data", dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        return
    if dataset_name == "multi30k":
        os.system("bash scripts/prepare-multi30k.sh")
    elif dataset_name == "wmt14":
        download(_urls["wmt"], path="wmt14.zip")
        os.system("bash scripts/prepare-wmt14.sh")
    elif dataset_name == "copy" or dataset_name == "tiny_copy":
        train_size = 9000
        valid_size = 1000
        test_size = 1000
        char_list = [chr(i) for i in range(ord("a"), ord("z") + 1)]
        with open(os.path.join(directory, "train.in"), "w") as f_in, open(
            os.path.join(directory, "train.out"), "w"
        ) as f_out:
            for i, l in zip(
                range(train_size),
                np.random.normal(15, 3, train_size).astype(int),
            ):
                l = max(l, 1)
                line = " ".join(np.random.choice(char_list, l)) + "\n"
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, "valid.in"), "w") as f_in, open(
            os.path.join(directory, "valid.out"), "w"
        ) as f_out:
            for i, l in zip(
                range(valid_size),
                np.random.normal(15, 3, valid_size).astype(int),
            ):
                l = max(l, 1)
                line = " ".join(np.random.choice(char_list, l)) + "\n"
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, "test.in"), "w") as f_in, open(
            os.path.join(directory, "test.out"), "w"
        ) as f_out:
            for i, l in zip(
                range(test_size), np.random.normal(15, 3, test_size).astype(int)
            ):
                l = max(l, 1)
                line = " ".join(np.random.choice(char_list, l)) + "\n"
                f_in.write(line)
                f_out.write(line)

        with open(os.path.join(directory, "vocab.txt"), "w") as f:
            for c in char_list:
                f.write(c + "\n")

    elif dataset_name == "sort" or dataset_name == "tiny_sort":
        train_size = 9000
        valid_size = 1000
        test_size = 1000
        char_list = [chr(i) for i in range(ord("a"), ord("z") + 1)]
        with open(os.path.join(directory, "train.in"), "w") as f_in, open(
            os.path.join(directory, "train.out"), "w"
        ) as f_out:
            for i, l in zip(
                range(train_size),
                np.random.normal(15, 3, train_size).astype(int),
            ):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(" ".join(seq) + "\n")
                f_out.write(" ".join(np.sort(seq)) + "\n")

        with open(os.path.join(directory, "valid.in"), "w") as f_in, open(
            os.path.join(directory, "valid.out"), "w"
        ) as f_out:
            for i, l in zip(
                range(valid_size),
                np.random.normal(15, 3, valid_size).astype(int),
            ):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(" ".join(seq) + "\n")
                f_out.write(" ".join(np.sort(seq)) + "\n")

        with open(os.path.join(directory, "test.in"), "w") as f_in, open(
            os.path.join(directory, "test.out"), "w"
        ) as f_out:
            for i, l in zip(
                range(test_size), np.random.normal(15, 3, test_size).astype(int)
            ):
                l = max(l, 1)
                seq = np.random.choice(char_list, l)
                f_in.write(" ".join(seq) + "\n")
                f_out.write(" ".join(np.sort(seq)) + "\n")

        with open(os.path.join(directory, "vocab.txt"), "w") as f:
            for c in char_list:
                f.write(c + "\n")
