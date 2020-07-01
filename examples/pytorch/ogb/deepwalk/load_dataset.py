""" load dataset from ogb """

import argparse

<<<<<<< HEAD
def load_from_ogbl_with_name(name):
    from ogb.linkproppred import DglLinkPropPredDataset
    
    choices = ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation']
    assert name in choices, "name must be selected from " + str(choices)
    dataset = DglLinkPropPredDataset(name)
    return dataset[0]

if __name__ == "__main__":
    from ogb.linkproppred import PygLinkPropPredDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
        choices=['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation'],
        default='ogbl-collab',
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
=======
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
    choices=['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation'], 
    default='ogbl-collab',
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
>>>>>>> 2c8e73b677837932ea40a8184046c49f6c5c682b
