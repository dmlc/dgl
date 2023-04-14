import numpy as np
import torch


def one_hotify(labels, pad=-1):
    """
    cast label to one hot vector
    """
    num_instances = len(labels)
    if pad <= 0:
        dim_embedding = np.max(labels) + 1  # zero-indexed assumed
    else:
        assert pad > 0, "result_dim for padding one hot embedding not set!"
        dim_embedding = pad + 1
    embeddings = np.zeros((num_instances, dim_embedding))
    embeddings[np.arange(num_instances), labels] = 1

    return embeddings


def pre_process(dataset, prog_args):
    """
    diffpool specific data partition, pre-process and shuffling
    """
    if prog_args.data_mode != "default":
        print("overwrite node attributes with DiffPool's preprocess setting")
        if prog_args.data_mode == "id":
            for g, _ in dataset:
                id_list = np.arange(g.num_nodes())
                g.ndata["feat"] = one_hotify(id_list, pad=dataset.max_num_node)

        elif prog_args.data_mode == "deg-num":
            for g, _ in dataset:
                g.ndata["feat"] = np.expand_dims(g.in_degrees(), axis=1)

        elif prog_args.data_mode == "deg":
            for g in dataset:
                degs = list(g.in_degrees())
                degs_one_hot = one_hotify(degs, pad=dataset.max_degrees)
                g.ndata["feat"] = degs_one_hot
