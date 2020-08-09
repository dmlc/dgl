""" load dataset from ogb """

import argparse
from ogb.linkproppred import DglLinkPropPredDataset

def load_from_ogbl_with_name(name):    
    choices = ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation']
    assert name in choices, "name must be selected from " + str(choices)
    dataset = DglLinkPropPredDataset(name)
    return dataset[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
        choices=['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation'],
        default='ogbl-collab',
        help="name of datasets by ogb")
    args = parser.parse_args()

    name = args.name
    g = load_from_ogbl_with_name(name=name)

    try:
        w = g.edata['edge_weight']
        weighted = True
    except:
        weighted = False

    with open(name + "-net.txt", "w") as f:
        for i in range(g.edges()[0].shape[0]):
            if weighted:
                f.write(str(g.edges()[0][i].item()) + " "\
                    +str(g.edges()[1][i].item()) + " "\
                    +str(g.edata['edge_weight'][i].item()) + "\n")
            else:
                f.write(str(g.edges()[0][i].item()) + " "\
                    +str(g.edges()[1][i].item()) + " "\
                    +"1\n")