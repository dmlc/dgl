import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd

path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n", "--name", help="name of graph to create", default="order"
)
parser.add_argument(
    "-nc",
    "--node_column",
    nargs="+",
    default=["order_id", "entity_index", "order_datetime", "cid"],
)
parser.add_argument("-nk", "--node_key", default="entity_index")
parser.add_argument(
    "-ec",
    "--edge_column",
    nargs="+",
    default=[
        "predicate_type",
        "predicate_index",
        "entity_index",
        "entity_index_y",
    ],
)
parser.add_argument("-es", "--edge_start", default="entity_index")
parser.add_argument("-en", "--edge_end", default="entity_index_y")
args = parser.parse_args()


# Store all types of node in nodes folder
nodes_list = sorted(glob.glob(os.path.join(path, "nodes/*")))

if os.path.exists("{}_nodes.txt".format(args.name)):
    os.remove("{}_nodes.txt".format(args.name))

schema_dict = defaultdict(dict)

node_type_id = 0
all_nodes_count = 0
for node_type_name in nodes_list:
    nodes_count = 0
    csv_files = sorted(glob.glob(os.path.join(node_type_name, "*.csv")))
    for file_name in csv_files:
        df = pd.read_csv(
            file_name,
            error_bad_lines=False,
            escapechar="\\",
            names=args.node_column,
            usecols=[*range(len(args.node_column))],
        )
        df_entity = pd.DataFrame(df[args.node_key], columns=[args.node_key])
        df_entity["type"] = node_type_id
        column_list = ["type"]
        for weight_index in range(len(nodes_list)):
            weight_num = "weight{}".format(weight_index)
            column_list.append(weight_num)
            if weight_index == node_type_id:
                df_entity[weight_num] = 1
            else:
                df_entity[weight_num] = 0
        nodes_count += len(df_entity.index)
        column_list.append(args.node_key)
        # This loop is trying to create file which servers as an input for Metis Algorithm.
        # More details about metis input can been found here : https://docs.dgl.ai/en/0.6.x/guide/distributed-preprocessing.html#input-format-for-parmetis
        df_entity.to_csv(
            "{}_nodes.txt".format(args.name),
            columns=column_list,
            sep=" ",
            index=False,
            header=False,
            mode="a",
        )
    schema_dict["nid"][os.path.basename(node_type_name)] = [
        all_nodes_count,
        nodes_count + all_nodes_count,
    ]
    all_nodes_count += nodes_count
    node_type_id += 1


if os.path.exists("{}_edges.txt".format(args.name)):
    os.remove("{}_edges.txt".format(args.name))

# Store all types of edge in edges folder
edges_list = sorted(glob.glob(os.path.join(path, "edges/*")))


all_edges_count = 0
edge_type_id = 0
for edge_type_name in edges_list:
    edge_count = 0
    csv_files = sorted(glob.glob(os.path.join(edge_type_name, "*.csv")))
    for file_name in csv_files:
        df = pd.read_csv(
            file_name,
            error_bad_lines=False,
            escapechar="\\",
            names=args.edge_column,
            usecols=[*range(len(args.edge_column))],
        )
        df_entity = pd.DataFrame(
            df[[args.edge_start, args.edge_end]],
            columns=[args.edge_start, args.edge_end],
        )
        df_entity["type"] = edge_type_id
        df_entity = df_entity.reset_index()
        df_entity["number"] = df_entity.index + edge_count
        edge_count += len(df_entity.index)
        # This loop is trying to create file which servers as an input for Metis Algorithm.
        # More details about metis input can been found here : https://docs.dgl.ai/en/0.6.x/guide/distributed-preprocessing.html#input-format-for-parmetis
        df_entity.to_csv(
            "{}_edges.txt".format(args.name),
            columns=[args.edge_start, args.edge_end, "number", "type"],
            sep=" ",
            index=False,
            header=False,
            mode="a",
        )
    schema_dict["eid"][os.path.basename(edge_type_name)] = [
        all_edges_count,
        all_edges_count + edge_count,
    ]
    edge_type_id += 1
    all_edges_count += edge_count

if os.path.exists("{}_stats.txt".format(args.name)):
    os.remove("{}_stats.txt".format(args.name))


df = pd.DataFrame(
    [[all_nodes_count, all_edges_count, len(nodes_list)]],
    columns=["nodes_count", "edges_count", "weight_count"],
)
df.to_csv(
    "{}_stats.txt".format(args.name),
    columns=["nodes_count", "edges_count", "weight_count"],
    sep=" ",
    index=False,
    header=False,
)

if os.path.exists("{}.json".format(args.name)):
    os.remove("{}.json".format(args.name))

with open("{}.json".format(args.name), "w", encoding="utf8") as json_file:
    json.dump(schema_dict, json_file, ensure_ascii=False)
