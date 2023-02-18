import argparse

import dgl
import dgl.nn as dglnn

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, 2 * in_feats),
            nn.BatchNorm1d(2 * in_feats),
            nn.ReLU(),
            nn.Linear(2 * in_feats, in_feats),
            nn.BatchNorm1d(in_feats),
        )

    def forward(self, h):
        return self.mlp(h)


class GIN(nn.Module):
    def __init__(self, n_hidden, n_output, n_layers=5):
        super().__init__()
        self.node_encoder = AtomEncoder(n_hidden)
        self.edge_encoders = nn.ModuleList(
            [BondEncoder(n_hidden) for _ in range(n_layers)]
        )

        self.pool = dglnn.AvgPooling()
        self.dropout = nn.Dropout(0.5)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(dglnn.GINEConv(MLP(n_hidden), learn_eps=True))
        self.predictor = nn.Linear(n_hidden, n_output)

        # add virtual node
        self.virtual_emb = nn.Embedding(1, n_hidden)
        nn.init.constant_(self.virtual_emb.weight.data, 0)
        self.virtual_layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.virtual_layers.append(MLP(n_hidden))
        self.virtual_pool = dglnn.SumPooling()

    def forward(self, g, x, x_e):
        v_emb = self.virtual_emb.weight.expand(g.batch_size, -1)
        hn = self.node_encoder(x)
        for i in range(len(self.layers)):
            v_hn = dgl.broadcast_nodes(g, v_emb)
            hn = hn + v_hn
            he = self.edge_encoders[i](x_e)
            hn = self.layers[i](g, hn, he)
            hn = F.relu(hn)
            hn = self.dropout(hn)
            if i != len(self.layers) - 1:
                v_emb_tmp = self.virtual_pool(g, hn) + v_emb
                v_emb = self.virtual_layers[i](v_emb_tmp)
                v_emb = self.dropout(F.relu(v_emb))
        hn = self.pool(g, hn)
        return self.predictor(hn)


@torch.no_grad()
def evaluate(dataloader, device, model, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for batched_graph, labels in tqdm(dataloader):
        batched_graph, labels = batched_graph.to(device), labels.to(device)
        node_feat, edge_feat = (
            batched_graph.ndata["feat"],
            batched_graph.edata["feat"],
        )
        y_hat = model(batched_graph, node_feat, edge_feat)
        y_true.append(labels.view(y_hat.shape).detach().cpu())
        y_pred.append(y_hat.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def train(rank, world_size, dataset_name, root):
    dist.init_process_group(
        "nccl", "tcp://127.0.0.1:12347", world_size=world_size, rank=rank
    )
    torch.cuda.set_device(rank)

    dataset = AsGraphPredDataset(DglGraphPropPredDataset(dataset_name, root))
    evaluator = Evaluator(dataset_name)

    model = GIN(300, dataset.num_tasks).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx], batch_size=256, use_ddp=True, shuffle=True
    )
    valid_dataloader = GraphDataLoader(dataset[dataset.val_idx], batch_size=256)
    test_dataloader = GraphDataLoader(dataset[dataset.test_idx], batch_size=256)

    for epoch in range(50):
        model.train()
        train_dataloader.set_epoch(epoch)
        for batched_graph, labels in train_dataloader:
            batched_graph, labels = batched_graph.to(rank), labels.to(rank)
            node_feat, edge_feat = (
                batched_graph.ndata["feat"],
                batched_graph.edata["feat"],
            )
            logits = model(batched_graph, node_feat, edge_feat)
            optimizer.zero_grad()
            is_labeled = labels == labels
            loss = F.binary_cross_entropy_with_logits(
                logits.float()[is_labeled], labels.float()[is_labeled]
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

        if rank == 0:
            val_metric = evaluate(
                valid_dataloader, rank, model.module, evaluator
            )[evaluator.eval_metric]
            test_metric = evaluate(
                test_dataloader, rank, model.module, evaluator
            )[evaluator.eval_metric]

            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                f"Val: {val_metric:.4f}, Test: {test_metric:.4f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        choices=["ogbg-molhiv", "ogbg-molpcba"],
        help="name of dataset (default: ogbg-molhiv)",
    )
    dataset_name = parser.parse_args().dataset
    root = "./data/OGB"
    DglGraphPropPredDataset(dataset_name, root)

    world_size = torch.cuda.device_count()
    print("Let's use", world_size, "GPUs!")
    args = (world_size, dataset_name, root)
    import torch.multiprocessing as mp

    mp.spawn(train, args=args, nprocs=world_size, join=True)
