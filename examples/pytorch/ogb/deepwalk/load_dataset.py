""" load dataset from ogb """

from ogb.linkproppred import PygLinkPropPredDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, choices=['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa'], default='ogbl-collab',
        help="name of datasets by ogb") 
args = parser.parse_args()

name = args.name

dataset = PygLinkPropPredDataset(name=name)
data = dataset[0]

try:
    weighted = data.edge_weight
    weighted = True
except:
    weighted = False


with open(name + "-net.txt", "w") as f:
    for i in range(data.edge_index.shape[1]):
        if weighted:
            f.write(str(data.edge_index[0][i].item()) + " "\
                +str(data.edge_index[1][i].item()) + " "\
                +str(data.edge_weight[i].item()) + "\n")
        else:
            f.write(str(data.edge_index[0][i].item()) + " "\
                +str(data.edge_index[1][i].item()) + " "\
                +"1\n")