import torch
from torch import nn
from ....base import NID, EID
from ....subgraph import khop_in_subgraph
import tqdm

__all__ = ['PGExplainer']


class PGExplainer(nn.Module):
    # model must either be a tuple of (gnne, gnnc) or have an embedding() function
    def __init__(self, model, graph, n_features, features, explain='node', epochs=30, lr=0.003,
                 temp=(5.0, 2.0), reg_coefs=(0.05, 1.0), sample_bias=0, log=True):
        super(PGExplainer, self).__init__()

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.explain = explain

        assert self.explain in ['node'], "Currently, we only support node explanations"

        self.model = model
        self.graph = graph
        self.n_inputs = (n_features * 3 if self.explain == 'node' else n_features * 2)
        self.features = features

        self.log = log

        self._prepared = False

        self.device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        self.graph = self.graph.to(self.device)
        self.features = self.features.to(self.device)

        self.explainer_model = nn.Sequential(
            nn.Linear(self.n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

    def prepare(self):
        if self._prepared: return

        self.train(range(self.graph.num_nodes()))
        self._prepared = True

    def _create_explainer_input(self, graph, g_mapping, embeds, n):
        src, dst = graph.edges()

        rows = graph.ndata[NID][src]
        cols = graph.ndata[NID][dst]

        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        # print(embeds.size())

        if self.explain == 'node':
            real_n = graph.ndata[dgl.NID][n]
            node_data = embeds[real_n]
            node_data = node_data.repeat(row_embeds.size(0), 1)
            items = [row_embeds, col_embeds, node_data]
        else:
            items = [row_embeds, col_embeds]

        # print([t.size() for t in items], embeds.size())

        inputs = torch.cat(items, 1)
        # print(inputs.size())
        return inputs

    def _sample_graph(self, sampling_weights, temp=1.0, bias=0.0, training=True):
        # Reparameterization Trick
        if training:
            bias = bias + 0.0001
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size(), device=self.device) + (
                        1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temp
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        return graph

    def _loss(self, masked_pred, orig_pred, mask, reg_coefs):
        size_reg, entropy_reg = reg_coefs

        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        cce_loss = torch.nn.functional.cross_entropy(masked_pred, orig_pred)

        return cce_loss + size_loss + mask_ent_loss

    def generate_embeddings(self, graph, features):
        if isinstance(self.model, tuple):
            return self.model[0].forward(graph, features)
        else:
            return self.model.embedding(graph, features)

    def train(self, indices):
        self.explainer_model.train()

        optimizer = torch.optim.Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * (
                    (self.temp[1] / self.temp[0]) ** (e / self.epochs))

        progress = tqdm(range(self.epochs)) if self.log else range(self.epochs)

        embeds = self.generate_embeddings(self.graph, self.features).detach()

        for epoch in progress:
            optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float64, device=self.device).detach()
            t = temp_schedule(epoch)

            for n in indices:
                if self.explain == 'node':  # explaining a node
                    g, g_mapping = khop_in_subgraph(self.graph, n, 3, store_ids=True)
                    feats = self.features[g.ndata[NID]].detach()
                else:
                    raise NotImplementedError("Graph-typed explainers not yet implemented")

                new_n = g_mapping[0]
                input_expl = self._create_explainer_input(g, g_mapping, embeds, new_n)
                # print(input_expl.size())
                # input_expl = input_expl.transpose(0, 1)
                # print(input_expl.size())
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
                # print(sampling_weights.size(), mask.size(), g.number_of_edges())
                masked_pred = self.model.forward(g, feats, eweight=mask)
                orig_pred = self.model.forward(g, feats)

                if self.explain == 'node':
                    masked_pred = masked_pred[new_n].unsqueeze(dim=0)
                    orig_pred = orig_pred[new_n]

                id_loss = self._loss(masked_pred, torch.argmax(orig_pred).unsqueeze(0), mask,
                                     self.reg_coefs)
                loss += id_loss

            loss.backward()
            optimizer.step()

            if self.log:
                progress.set_postfix({"loss": float(loss)})

    def explain_graph(self, index):
        self.prepare()

        if True:
            g, g_mapping = khop_in_subgraph(self.graph, index, 3)
            embeds = self.generate_embeddings(self.graph, self.features).detach()
        else:
            raise NotImplementedError("Graph explanations not yet implemented")

        new_index = g_mapping[index]
        input_expl = self._create_explainer_input(g, g_mapping, embeds, new_index).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(g.number_of_edges())
        src, dst = g.edges()
        for i, (src, dst) in enumerate(zip(src, dst)):
            expl_graph_weights[i] = mask[i]

        return g, expl_graph_weights