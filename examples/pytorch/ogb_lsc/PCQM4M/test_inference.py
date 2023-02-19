import argparse
import os
import random

import dgl

import numpy as np
import torch
from gnn import GNN
from ogb.lsc import PCQM4MDataset, PCQM4MEvaluator
from ogb.utils import smiles2graph
from torch.utils.data import DataLoader
from tqdm import tqdm


def collate_dgl(graphs):
    batched_graph = dgl.batch(graphs)

    return batched_graph


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, bg in enumerate(tqdm(loader, desc="Iteration")):
        bg = bg.to(device)
        x = bg.ndata.pop("feat")
        edge_attr = bg.edata.pop("feat")

        with torch.no_grad():
            pred = model(bg, x, edge_attr).view(
                -1,
            )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


class OnTheFlyPCQMDataset(object):
    def __init__(self, smiles_list, smiles2graph=smiles2graph):
        super(OnTheFlyPCQMDataset, self).__init__()
        self.smiles_list = smiles_list
        self.smiles2graph = smiles2graph

    def __getitem__(self, idx):
        """Get datapoint with index"""
        smiles, _ = self.smiles_list[idx]
        graph = self.smiles2graph(smiles)

        dgl_graph = dgl.graph(
            (graph["edge_index"][0], graph["edge_index"][1]),
            num_nodes=graph["num_nodes"],
        )
        dgl_graph.edata["feat"] = torch.from_numpy(graph["edge_feat"]).to(
            torch.int64
        )
        dgl_graph.ndata["feat"] = torch.from_numpy(graph["node_feat"]).to(
            torch.int64
        )

        return dgl_graph

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.smiles_list)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="GNN baselines on pcqm4m with DGL"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed to use (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin-virtual",
        help="GNN to use, which can be from "
        "[gin, gin-virtual, gcn, gcn-virtual] (default: gin-virtual)",
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="sum",
        help="graph pooling strategy mean or sum (default: sum)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0, help="dropout ratio (default: 0)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=600,
        help="dimensionality of hidden units in GNNs (default: 600)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="directory to save checkpoint",
    )
    parser.add_argument(
        "--save_test_dir",
        type=str,
        default="",
        help="directory to save test submission file",
    )
    args = parser.parse_args()

    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    ### automatic data loading and splitting
    ### Read in the raw SMILES strings
    smiles_dataset = PCQM4MDataset(root="dataset/", only_smiles=True)
    split_idx = smiles_dataset.get_idx_split()

    test_smiles_dataset = [smiles_dataset[i] for i in split_idx["test"]]
    onthefly_dataset = OnTheFlyPCQMDataset(test_smiles_dataset)
    test_loader = DataLoader(
        onthefly_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_dgl,
    )

    ### automatic evaluator.
    evaluator = PCQM4MEvaluator()

    shared_params = {
        "num_layers": args.num_layers,
        "emb_dim": args.emb_dim,
        "drop_ratio": args.drop_ratio,
        "graph_pooling": args.graph_pooling,
    }

    if args.gnn == "gin":
        model = GNN(gnn_type="gin", virtual_node=False, **shared_params).to(
            device
        )
    elif args.gnn == "gin-virtual":
        model = GNN(gnn_type="gin", virtual_node=True, **shared_params).to(
            device
        )
    elif args.gnn == "gcn":
        model = GNN(gnn_type="gcn", virtual_node=False, **shared_params).to(
            device
        )
    elif args.gnn == "gcn-virtual":
        model = GNN(gnn_type="gcn", virtual_node=True, **shared_params).to(
            device
        )
    else:
        raise ValueError("Invalid GNN type")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint file not found at {checkpoint_path}")

    ## reading in checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Predicting on test data...")
    y_pred = test(model, device, test_loader)
    print("Saving test submission file...")
    evaluator.save_test_submission({"y_pred": y_pred}, args.save_test_dir)


if __name__ == "__main__":
    main()
