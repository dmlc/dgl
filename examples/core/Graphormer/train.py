import argparse
import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from dgl import shortest_dist
from dgl.data import download
from dgl.dataloading import GraphDataLoader
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)


class Graphormer(nn.Module):
    def __init__(
        self,
        num_classes=1,
        edge_dim=3,
        num_atoms=4608,
        max_degree=512,
        num_spatial=511,
        multi_hop_max_dist=5,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        pre_layernorm: bool = True,
        activation_fn=th.nn.GELU(),
    ):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads

        # 1 for graph token
        self.atom_encoder = nn.Embedding(
            num_atoms + 1, embedding_dim, padding_idx=0
        )
        self.graph_token = nn.Embedding(1, embedding_dim)

        # degree encoder
        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree, embedding_dim=embedding_dim
        )

        # path encoder
        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_attention_heads,
        )

        # spatial encoder
        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_attention_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # map graph_rep to num_classes
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, num_classes, bias=False)
        self.lm_output_learned_bias = nn.Parameter(th.zeros(num_classes))

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(th.zeros(1))
        self.embed_out.reset_parameters()

    def forward(
        self,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
        attn_mask=None,
    ):
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(th.stack((in_degree, out_degree)))

        # node feauture + graph token
        node_feat = self.atom_encoder(node_feat.int()).sum(dim=-2)
        node_feat = node_feat + deg_emb
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
            num_graphs, 1, 1
        )
        x = th.cat([graph_token_feat, node_feat], dim=1)

        attn_bias = th.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        # cls spatial encoder
        t = self.graph_token_virtual_distance.weight.reshape(
            1, 1, self.num_heads
        )
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        x = self.emb_layer_norm(x)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 0, :]
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return graph_rep


def train_epoch(model, optimizer, data_loader, lr_scheduler):
    model.train()
    epoch_loss = 0
    list_scores = []
    list_labels = []
    for iter, (
        batch_labels,
        attn_mask,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
    ) in enumerate(data_loader):
        optimizer.zero_grad()
        device = accelerator.device

        batch_scores = model(
            node_feat.to(device),
            in_degree.to(device),
            out_degree.to(device),
            path_data.to(device),
            dist.to(device),
            attn_mask=attn_mask,
        )

        loss = nn.BCEWithLogitsLoss()(batch_scores, batch_labels.float())

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
        list_scores.append(batch_scores)
        list_labels.append(batch_labels)

        # release GPU memory
        del (
            batch_labels,
            batch_scores,
            loss,
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
        )
        th.cuda.empty_cache()

    epoch_loss /= iter + 1

    evaluator = Evaluator(name="ogbg-molhiv")
    epoch_train_metric = evaluator.eval(
        {"y_pred": th.cat(list_scores), "y_true": th.cat(list_labels)}
    )["rocauc"]

    return epoch_loss, epoch_train_metric


def evaluate_network(model, data_loader):
    model.eval()
    epoch_test_loss = 0
    with th.no_grad():
        list_scores = []
        list_labels = []
        for iter, (
            batch_labels,
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
        ) in enumerate(data_loader):
            device = accelerator.device

            batch_scores = model(
                node_feat.to(device),
                in_degree.to(device),
                out_degree.to(device),
                path_data.to(device),
                dist.to(device),
                attn_mask=attn_mask,
            )

            # Gather all predictions and targets
            all_predictions, all_targets = accelerator.gather_for_metrics(
                (batch_scores, batch_labels)
            )
            loss = nn.BCEWithLogitsLoss()(all_predictions, all_targets.float())

            epoch_test_loss += loss.item()
            list_scores.append(all_predictions)
            list_labels.append(all_targets)

        epoch_test_loss /= iter + 1

        evaluator = Evaluator(name="ogbg-molhiv")
        epoch_test_metric = evaluator.eval(
            {"y_pred": th.cat(list_scores), "y_true": th.cat(list_labels)}
        )["rocauc"]

    return epoch_test_loss, epoch_test_metric


def train_val_pipeline(params):

    dataset = MolHIVDataset()

    train_loader = GraphDataLoader(
        dataset.train,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        pin_memory=True,
        num_workers=16,
    )
    val_loader = GraphDataLoader(
        dataset.val,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        pin_memory=True,
        num_workers=16,
    )
    test_loader = GraphDataLoader(
        dataset.test,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        pin_memory=True,
        num_workers=16,
    )

    # load pretrain model
    download(url="https://data.dgl.ai/pre_trained/graphormer_pcqm.pth")
    filename = "new_graphormer_pcqm.pkl"
    model = Graphormer()
    state_dict = th.load(filename)
    model.load_state_dict(state_dict)
    # reset output layer parameters
    model.reset_output_layer_parameters()
    num_epochs = 16
    tot_updates = 33000 * num_epochs / params.batch_size
    warmup_updates = tot_updates * 0.16

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=tot_updates,
        lr_end=1e-9,
        power=1.0,
    )

    epoch_train_AUCs, epoch_val_AUCs, epoch_test_AUCs = [], [], []

    # multi-GPUs
    (
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
    )

    for epoch in range(num_epochs):

        epoch_train_loss, epoch_train_auc = train_epoch(
            model, optimizer, train_loader, lr_scheduler
        )
        epoch_val_loss, epoch_val_auc = evaluate_network(model, val_loader)
        epoch_test_loss, epoch_test_auc = evaluate_network(model, test_loader)

        epoch_train_AUCs.append(epoch_train_auc)
        epoch_val_AUCs.append(epoch_val_auc)
        epoch_test_AUCs.append(epoch_test_auc)

        accelerator.print(
            "Epoch={}, lr={}, train_ROCAUC={:.3f}, val_ROCAUC={:.3f}, test_ROCAUC={:.3f}".format(
                epoch + 1,
                optimizer.param_groups[0]["lr"],
                epoch_train_auc,
                epoch_val_auc,
                epoch_test_auc,
            )
        )

    # Return test and train metrics at best val metric
    index = epoch_val_AUCs.index(max(epoch_val_AUCs))
    val_auc = epoch_val_AUCs[index]
    train_auc = epoch_train_AUCs[index]
    test_auc = epoch_test_AUCs[index]

    accelerator.print("Test ROCAUC: {:.4f}".format(test_auc))
    accelerator.print("Val ROCAUC: {:.4f}".format(val_auc))
    accelerator.print("Train ROCAUC: {:.4f}".format(train_auc))
    accelerator.print("Best epoch index: {:.4f}".format(index))


class MolHIVDataset(th.utils.data.Dataset):
    def __init__(self):

        dataset = DglGraphPropPredDataset(name="ogbg-molhiv")
        split_idx = dataset.get_idx_split()

        # shortest path distance
        for g, label in dataset:
            spd, path = shortest_dist(g, root=None, return_paths=True)
            g.ndata["spd"] = spd
            g.ndata["path"] = path

        self.train, self.val, self.test = (
            dataset[split_idx["train"]],
            dataset[split_idx["valid"]],
            dataset[split_idx["test"]],
        )

        accelerator.print(
            "train, test, val sizes :",
            len(self.train),
            len(self.test),
            len(self.val),
        )
        accelerator.print("[I] Finished loading.")

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = th.stack(labels)

        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        dist = -th.ones(
            (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
        )

        for i in range(num_graphs):
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            node_feat.append(graphs[i].ndata["feat"] + 1)

            in_degree.append(
                th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
            )
            out_degree.append(
                th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
            )

            # path & spatial
            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)

            # shape: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, 0:max_len]
            else:
                p1d = (0, max_len - path_len)
                shortest_path = F.pad(path, p1d, "constant", -1)
            p3d = (
                0,
                0,
                0,
                max_num_nodes - num_nodes[i],
                0,
                max_num_nodes - num_nodes[i],
            )
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)
            edata = graphs[i].edata["feat"] + 1
            edata = th.cat(
                (edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )

            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        # node feat padding
        node_feat = pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)

        return (
            labels.reshape(num_graphs, -1),
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            th.stack(path_data),
            dist,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Please give a value for seed",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please give a value for batch_size",
    )
    args = parser.parse_args()

    # multi-GPUs
    accelerator = Accelerator()

    # setting seeds
    random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)

    train_val_pipeline(args)
