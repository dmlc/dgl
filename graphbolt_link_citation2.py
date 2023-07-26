import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.linkproppred import Evaluator


############## Subgraph sampler function ##################
class LinkSampler:
    def __init__(self, graph, num_layer, fanout):
        assert num_layer == len(fanout)
        self.graph = graph
        self.num_layer = num_layer
        self.fanout = fanout

    def __call__(self, data):
        adjs = []
        # Data format is (u, v, neg_v...)
        u = data[0]
        u = torch.cat([u.repeat_interleave(neg_ratio), u])
        v = [t.view(1, -1) for t in data[2:]]
        v = torch.cat(v, dim=1).view(-1)
        # Cat pos and neg v.
        v = torch.cat([data[1], v])
        node_pairs = (u, v)

        seeds, compacted_pairs = gb.unique_and_compact_node_pairs(node_pairs)
        for hop in range(num_layers):
            sg = self._generate(seeds, torch.LongTensor(self.fanout[hop]))
            sg = dgl.graph(sg.node_pairs[("_N", "_E", "_N")])
            block = dgl.to_block(sg, seeds)
            seeds = block.srcdata[dgl.NID]
            adjs.insert(0, block)

        input_nodes = seeds
        return input_nodes, compacted_pairs, adjs

    def _generate(self, seeds, fanout):
        raise NotImplementedError


class LinkNeighborSampler(LinkSampler):
    def _generate(self, seeds, fanout):
        return self.graph.sample_neighbors(seeds, fanout)


############## Negative sampler function ##################
class NegativeSampler:
    def __init__(self, graph, negative_ratio):
        self.graph = graph
        self.negative_ratio = negative_ratio

    def __call__(self, data):
        node_pairs = data
        neg_pairs = self._generate(node_pairs)
        num_sample = neg_pairs[0].shape[0] // self.negative_ratio
        neg_dst = list(neg_pairs[1].split(num_sample))
        data = list(data) + neg_dst
        return data

    def _generate(self, node_pairs, edge_type=None):
        raise NotImplementedError


class PerSourceUniformSampler(NegativeSampler):
    def _generate(self, node_pairs, edge_type=None):
        return self.graph.sample_negative_edges_uniform(
            edge_type=edge_type,
            node_pairs=node_pairs,
            negative_ratio=self.negative_ratio,
        )


############## Model ##################
hidden_channels = 256
out_channels = 1
num_layers = 3
lr = 0.0005
in_size = 128


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        return h


# Miss `num_features` in dataset.
model = SAGE(in_size, hidden_channels)
optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)


############## DataLoader ##################
dataset = gb.OnDiskDataset("output_config.yaml")
features = dataset.feature()[("node", None, "feat")]
graph = dataset.graph()
train_set = dataset.train_sets()[0]
test_set = dataset.test_sets()[0]
valid_set = dataset.validation_sets()[0]
fanout = [[15], [10], [5]]
hop = 2
batch_size = 512
neg_ratio = 1000


def fetch_func(data):
    input_nodes, compacted_pairs, adjs = data
    input_features = features.read(input_nodes)
    return input_features.float(), compacted_pairs, adjs


minibatch_sampler = gb.MinibatchSampler(
    train_set, batch_size=batch_size, shuffle=True
)
negative_sampler = gb.SubgraphSampler(
    minibatch_sampler, PerSourceUniformSampler(graph, neg_ratio)
)

subgraph_sampler = gb.SubgraphSampler(
    negative_sampler,
    LinkNeighborSampler(graph, num_layers, fanout),
)

feature_fetcher = gb.FeatureFetcher(subgraph_sampler, fetch_func)
device_transfer = gb.CopyTo(feature_fetcher, torch.device("cpu"))
dataloader = gb.SingleProcessDataLoader(device_transfer)


############## Train ##################
total_loss = 0
for epoch in range(1):
    model.train()
    for it, (input_features, compacted_pairs, blocks) in enumerate(dataloader):
        z = model(blocks, input_features)
        logits = model.predictor(
            z[compacted_pairs[0]] * z[compacted_pairs[1]]
        ).squeeze()
        labels = torch.zeros_like(logits)
        num_batch = compacted_pairs[0].shape[0] // (neg_ratio + 1)
        labels[:num_batch] = 1.0
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        print(f"it = {it}, loss = {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
torch.save(model.state_dict(), "model.pth")

############## Eval #################
model.load_state_dict(torch.load("model.pth"))
evaluator = Evaluator(name="ogbl-citation2")

num_layers = 1
fanout = [[-1]]
minibatch_sampler = gb.MinibatchSampler(test_set, batch_size=512)
subgraph_sampler.datapipe = minibatch_sampler
pos_test_pred = []
neg_test_pred = []
model.eval()
for it, (input_features, compacted_pairs, blocks) in enumerate(dataloader):
    x = model(blocks, input_features)
    score = (
        (model.predictor(x[compacted_pairs[0]] * x[compacted_pairs[1]]))
        .squeeze()
        .detach()
    )
    num_batch = compacted_pairs[0].shape[0] // (neg_ratio + 1)
    pos_test_pred.append(score[:num_batch])
    neg_test_pred.append(score[num_batch:])
    print(it)
    if it == 15:
        break
pos_test_pred = torch.cat(pos_test_pred, dim=0)
neg_test_pred = torch.cat(neg_test_pred, dim=0).view(-1, neg_ratio)

input_dict = {"y_pred_pos": pos_test_pred, "y_pred_neg": neg_test_pred}
test_mrr = evaluator.eval(input_dict)["mrr_list"]
print(f"test_mrr: {test_mrr}")
print(f"mean test_mrr: {test_mrr.mean()}")


minibatch_sampler = gb.MinibatchSampler(valid_set, batch_size=1000)
subgraph_sampler.datapipe = minibatch_sampler
pos_valid_pred = []
neg_valid_pred = []
for input_features, compacted_pairs, blocks in tqdm.tqdm(
    dataloader, desc="valid"
):
    optimizer.zero_grad()

    x = model(blocks, input_features)
    score = (
        (model.predictor(x[compacted_pairs[0]] * x[compacted_pairs[1]]))
        .squeeze()
        .detach()
    )
    num_batch = compacted_pairs[0].shape[0] // (neg_ratio + 1)
    pos_valid_pred.append(score[:num_batch])
    neg_valid_pred.append(score[num_batch:])
pos_valid_pred = torch.cat(pos_valid_pred, dim=0)
neg_valid_pred = torch.cat(neg_valid_pred, dim=0).view(-1, neg_ratio)

input_dict = {"y_pred_pos": pos_valid_pred, "y_pred_neg": neg_valid_pred}
valid_mrr = evaluator.eval(input_dict)["mrr_list"]
print(f"valid_mrr: {valid_mrr}")
print(f"mean valid_mrr: {valid_mrr.mean()}")
