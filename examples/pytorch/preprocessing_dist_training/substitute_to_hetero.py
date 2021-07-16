import argparse
import os
import glob
import pandas as pd

path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="name of graph to create", default="order")
parser.add_argument("-p", "--path", help="path where data is stored", default=path)
parser.add_argument("-f", "--file", nargs="+", default=None)
parser.add_argument("-r", "--removed", help="file where we have edges that were dropped", default="remove.csv")
args = parser.parse_args()

if args.file is None:
    edges_file = sorted(glob.glob(os.path.join(args.path, "*-{}_edges.txt".format(args.name))))

remove_column_index = [0, 1, 2, 3]
remove_column_name = ["distributed_src_id", "distributed_dest_id", "src_id", "dest_id"]
removed_df = pd.read_csv(args.removed, sep=" ", usecols=[0, 1, 2, 3], names=["src_id", "dest_id", "id", "type"])

for edge_file in edges_file:
    part_df = pd.read_csv(edge_file, sep=" ", usecols=remove_column_index, names=remove_column_name).drop_duplicates(["src_id", "dest_id"])
    merge_df = pd.merge(part_df, removed_df, how='inner', on=["src_id", "dest_id"])
    merge_df.to_csv(edge_file, mode='a', header=False, index=False, sep=" ")
