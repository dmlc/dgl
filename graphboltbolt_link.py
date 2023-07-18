import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.linkproppred import Evaluator


############## Model ##################
hidden_channels = 256
out_channels = 1
num_layers = 3


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h


# Miss `num_features` in dataset.
model = SAGE(3, hidden_channels)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)


############## DataLoader ##################
dataset = gb.OnDiskDataset("output_config.yaml")
features = dataset.feature()[("node", None, "feat")]
graph = dataset.graph()
train_set = dataset.train_sets()[0]
test_set = dataset.test_sets()[0]
valid_set = dataset.validation_sets()[0]
fanout = [2]
hop = 2
batch_size = 2048


def sampler_func(data):
    adjs = []
    seeds = data
    node_pairs, labels = data[:2], data[2]
    seeds, compacted_pairs = gb.unique_and_compact_node_pairs(node_pairs)
    for _ in range(num_layers):
        sg = graph.sample_neighbors(seeds, torch.LongTensor(fanout))
        sg = dgl.graph(sg.node_pairs[("_N", "_E", "_N")])
        block = dgl.to_block(sg, seeds)
        seeds = block.srcdata[dgl.NID]
        adjs.insert(0, block)

    input_nodes = seeds
    return input_nodes, compacted_pairs, labels, adjs


def fetch_func(data):
    input_nodes, compacted_pairs, labels, adjs = data
    input_features = features.read(input_nodes)
    return input_features, compacted_pairs, labels, adjs


minibatch_sampler = gb.MinibatchSampler(train_set, batch_size=batch_size, shuffle=True)
subgraph_sampler = gb.SubgraphSampler(
    minibatch_sampler,
    sampler_func,
)
feature_fetcher = gb.FeatureFetcher(subgraph_sampler, fetch_func)
device_transfer = gb.CopyTo(feature_fetcher, torch.device("cpu"))
dataloader = gb.SingleProcessDataLoader(device_transfer)


############## Train ##################
# total_loss = 0
# for input_features, compacted_pairs, labels, blocks in tqdm.tqdm(
#     dataloader, desc="Train"
# ):
#     z = model(blocks, input_features)
#     logits = model.predictor(z[compacted_pairs[0]] * z[compacted_pairs[1]]).squeeze()
#     loss = F.binary_cross_entropy_with_logits(logits, labels.float())
#     print(loss)
#     optimizer.zero_grad()
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     optimizer.step()
#     total_loss += loss.item()


############## Eval #################
minibatch_sampler = gb.MinibatchSampler(test_set, batch_size=batch_size)
subgraph_sampler.datapipe = minibatch_sampler

evaluator = Evaluator(name="ogbl-vessel")
pos_test_pred = []
neg_test_pred = []
for input_features, compacted_pairs, labels, blocks in tqdm.tqdm(
    dataloader, desc="Test"
):
    optimizer.zero_grad()

    x = model(blocks, input_features)
    logits = (
        (model.predictor(x[compacted_pairs[0]] * x[compacted_pairs[1]]))
        .squeeze()
        .detach()
    )
    mask = labels == 1
    pos_test_pred.append(logits[mask])
    neg_test_pred.append(logits[~mask])
pos_test_pred = torch.cat(pos_test_pred, dim=0)
neg_test_pred = torch.cat(neg_test_pred, dim=0)
test_rocauc = evaluator.eval(
    {
        "y_pred_pos": pos_test_pred,
        "y_pred_neg": neg_test_pred,
    }
)[f"rocauc"]
print(f"test_rocauc: {test_rocauc}")

minibatch_sampler = gb.MinibatchSampler(valid_set, batch_size=batch_size)
subgraph_sampler.datapipe = minibatch_sampler
pos_valid_pred = []
neg_valid_pred = []
for input_features, compacted_pairs, labels, blocks in tqdm.tqdm(
    dataloader, desc="Valid"
):
    optimizer.zero_grad()

    x = model(blocks, input_features)
    logits = (
        (model.predictor(x[compacted_pairs[0]] * x[compacted_pairs[1]]))
        .squeeze()
        .detach()
    )
    mask = labels == 1
    pos_valid_pred.append(logits[mask])
    neg_valid_pred.append(logits[~mask])
pos_valid_pred = torch.cat(pos_valid_pred, dim=0)
neg_valid_pred = torch.cat(neg_valid_pred, dim=0)
valid_rocauc = evaluator.eval(
    {
        "y_pred_pos": pos_valid_pred,
        "y_pred_neg": neg_valid_pred,
    }
)[f"rocauc"]
print(f"valid_rocauc: {valid_rocauc}")
