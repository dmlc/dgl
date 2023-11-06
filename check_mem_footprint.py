import dgl
from dgl.distributed import load_partition
import psutil
import os
import argparse
import gc

parser = argparse.ArgumentParser(description="check memory footprint")
parser.add_argument(
    "--part_config",
    type=str,
    help="partition config file",
)
parser.add_argument(
    "--graphbolt",
    action="store_true",
    help="use graphbolt",
)
parser.add_argument(
    "--part_id",
    type=int,
    help="partition id",
)

args = parser.parse_args()

use_graphbolt = args.graphbolt
part_id = args.part_id

prev_rss = psutil.Process(os.getpid()).memory_info().rss
(
    client_g,
    _,
    _,
    gpb,
    graph_name,
    ntypes,
    etypes,
) = load_partition(
    args.part_config,
    part_id,
    load_feats=False,
    use_graphbolt=use_graphbolt,
)
if not use_graphbolt:
    graph_format=("csc")
    client_g = client_g.formats(graph_format)
    client_g.create_formats_()
new_rss = psutil.Process(os.getpid()).memory_info().rss
print(f"[PartID_{part_id}] Loaded {graph_name} with use_graphbolt[{use_graphbolt}] in size[{(new_rss - prev_rss)/1024/1024 : .0f} MB]")
client_g = None
gc.collect()
