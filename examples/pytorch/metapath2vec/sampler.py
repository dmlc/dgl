import os
import random
import sys
import time

import dgl
import numpy as np
import tqdm

num_walks_per_node = 1000
walk_length = 100
path = sys.argv[1]


def construct_graph():
    paper_ids = []
    paper_names = []
    author_ids = []
    author_names = []
    conf_ids = []
    conf_names = []
    f_3 = open(os.path.join(path, "id_author.txt"), encoding="ISO-8859-1")
    f_4 = open(os.path.join(path, "id_conf.txt"), encoding="ISO-8859-1")
    f_5 = open(os.path.join(path, "paper.txt"), encoding="ISO-8859-1")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        author_ids.append(identity)
        author_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break
        w = w.strip().split()
        identity = int(w[0])
        conf_ids.append(identity)
        conf_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break
        v = v.strip().split()
        identity = int(v[0])
        paper_name = "p" + "".join(v[1:])
        paper_ids.append(identity)
        paper_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    author_ids_invmap = {x: i for i, x in enumerate(author_ids)}
    conf_ids_invmap = {x: i for i, x in enumerate(conf_ids)}
    paper_ids_invmap = {x: i for i, x in enumerate(paper_ids)}

    paper_author_src = []
    paper_author_dst = []
    paper_conf_src = []
    paper_conf_dst = []
    f_1 = open(os.path.join(path, "paper_author.txt"), "r")
    f_2 = open(os.path.join(path, "paper_conf.txt"), "r")
    for x in f_1:
        x = x.split("\t")
        x[0] = int(x[0])
        x[1] = int(x[1].strip("\n"))
        paper_author_src.append(paper_ids_invmap[x[0]])
        paper_author_dst.append(author_ids_invmap[x[1]])
    for y in f_2:
        y = y.split("\t")
        y[0] = int(y[0])
        y[1] = int(y[1].strip("\n"))
        paper_conf_src.append(paper_ids_invmap[y[0]])
        paper_conf_dst.append(conf_ids_invmap[y[1]])
    f_1.close()
    f_2.close()

    hg = dgl.heterograph(
        {
            ("paper", "pa", "author"): (paper_author_src, paper_author_dst),
            ("author", "ap", "paper"): (paper_author_dst, paper_author_src),
            ("paper", "pc", "conf"): (paper_conf_src, paper_conf_dst),
            ("conf", "cp", "paper"): (paper_conf_dst, paper_conf_src),
        }
    )
    return hg, author_names, conf_names, paper_names


# "conference - paper - Author - paper - conference" metapath sampling
def generate_metapath():
    output_path = open(os.path.join(path, "output_path.txt"), "w")
    count = 0

    hg, author_names, conf_names, paper_names = construct_graph()

    for conf_idx in tqdm.trange(hg.num_nodes("conf")):
        traces, _ = dgl.sampling.random_walk(
            hg,
            [conf_idx] * num_walks_per_node,
            metapath=["cp", "pa", "ap", "pc"] * walk_length,
        )
        for tr in traces:
            outline = " ".join(
                (conf_names if i % 4 == 0 else author_names)[tr[i]]
                for i in range(0, len(tr), 2)
            )  # skip paper
            print(outline, file=output_path)
    output_path.close()


if __name__ == "__main__":
    generate_metapath()
