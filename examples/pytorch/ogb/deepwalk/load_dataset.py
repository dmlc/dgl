""" load dataset from ogb """

import argparse
import time

from ogb.linkproppred import DglLinkPropPredDataset


def load_from_ogbl_with_name(name):
    choices = ["ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation"]
    assert name in choices, "name must be selected from " + str(choices)
    dataset = DglLinkPropPredDataset(name)
    return dataset[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        choices=["ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation"],
        default="ogbl-collab",
        help="name of datasets by ogb",
    )
    args = parser.parse_args()

    print("loading graph... it might take some time")
    name = args.name
    g = load_from_ogbl_with_name(name=name)

    try:
        w = g.edata["edge_weight"]
        weighted = True
    except:
        weighted = False

    edge_num = g.edges()[0].shape[0]
    src = list(g.edges()[0])
    tgt = list(g.edges()[1])
    if weighted:
        weight = list(g.edata["edge_weight"])

    print("writing...")
    start_time = time.time()
    with open(name + "-net.txt", "w") as f:
        for i in range(edge_num):
            if weighted:
                f.write(
                    str(src[i].item())
                    + " "
                    + str(tgt[i].item())
                    + " "
                    + str(weight[i].item())
                    + "\n"
                )
            else:
                f.write(
                    str(src[i].item()) + " " + str(tgt[i].item()) + " " + "1\n"
                )
    print("writing used time: %d s" % int(time.time() - start_time))
