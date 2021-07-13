import argparse
import os
from collections import defaultdict
import glob

path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="name of graph to create", default="order")
parser.add_argument("-p", "--path", help="path where data is stored", default=path)
parser.add_argument("-f", "--file", nargs="+", default=None)
parser.add_argument("-r", "--removed", help="file where we have edges that were dropped", default="remove.csv")
args = parser.parse_args()

graph_name = args.name

if args.file is None:
    edges_file = sorted(glob.glob(os.path.join(args.path, "*-{}_edges.txt".format(graph_name))))



for edge_file in edges_file:
    edges_dict = defaultdict(dict)
    with open(edge_file) as fp:
        for line in fp:
            if line:
                formatted_line = line.strip()
                resp_array = formatted_line.split(" ")
                edges_dict[resp_array[2]][resp_array[3]] = "{} {}".format(resp_array[0], resp_array[1])

    with open(args.removed) as fp, open(edge_file,'a') as wr:
        for line in fp:
            if line:
                delete_array = line.split(" ")
                if edges_dict.get(delete_array[0], {}).__contains__(delete_array[1]):
                    wr.write("{} {}".format(edges_dict[delete_array[0]][delete_array[1]], line))

