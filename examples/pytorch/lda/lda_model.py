# Copyright 2021 Yifei Ma
# with references from "sklearn.decomposition.LatentDirichletAllocation"
# with the following original authors:
# * Chyi-Kwei Yau (the said scikit-learn implementation)
# * Matthew D. Hoffman (original onlineldavb implementation)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os, functools, warnings
import numpy as np, scipy as sp
import torch
import dgl
from dgl import function as fn


def _lbeta(alpha, axis):
    return torch.lgamma(alpha).sum(axis) - torch.lgamma(alpha.sum(axis))

# Taken from scikit-learn.  Worked better than uniform.
# Perhaps this is due to concentration around one.
_init = torch.distributions.gamma.Gamma(100, 100).sample


def _edge_update(edges, step_size=1):
    """ the Gibbs posterior distribution of z propto theta*beta.
    As step_size -> infty, the result becomes MAP estimate on the dst nodes.
    """
    q = edges.src['weight'] * edges.dst['weight']
    marg = q.sum(axis=1, keepdims=True) + np.finfo(float).eps
    p = q / marg

    return {
        'z': p * step_size,
        'edge_elbo': marg.squeeze(1).log() * step_size,
    }


def _weight_fn(z, ntype, prior):
    """Node weight is approximately normalized for VB along the ntype
    direction.
    """
    prior = prior + z * 0 # convert numpy to torch
    gamma = prior + z

    axis = 1 if ntype == 'doc' else 0 # word
    Elog = torch.digamma(gamma) - torch.digamma(gamma.sum(axis, keepdims=True))

    return Elog.exp()


def _message_passing(G, dst_type, step_size, doc_weight, word_weight, var_name):
    """Follows Eq (5) of Hoffman et al., 2010.
    """
    G = G.reverse() if dst_type == 'doc' else G # word

    G.nodes['doc'].data['weight'] = doc_weight.to(G.device)
    G.nodes['word'].data['weight'] = word_weight.to(G.device)
    msg_fn = lambda edges: _edge_update(edges, step_size)

    G.update_all(msg_fn, fn.sum(var_name, var_name))
    return G.nodes[dst_type].data[var_name]


class LatentDirichletAllocation:
    """LDA model that works with a HeteroGraph with doc->word meta paths.
    The model alters the attributes of G arbitrarily, starting with word->weight.
    This is inspired by [1] and its corresponding scikit-learn implementation.

    References
    ---
    [1] Matthew Hoffman, Francis Bach, David Blei. Online Learning for Latent
    Dirichlet Allocation. Advances in Neural Information Processing Systems 23
    (NIPS 2010).
    [2] Reactive LDA Library blogpost by Yingjie Miao for a similar Gibbs model
    """
    def __init__(
        self, n_words, n_components,
        prior=None,
        step_size={'doc': 1, 'word': 1}, # use larger value to get MAP
        word_rho=1, # use smaller value for online update
        verbose=True,
        device='cpu', # use gpus for faster _word_weight update
        ):
        self.n_components = n_components

        if prior is None:
            prior = {'doc': 1./n_components, 'word': 1./n_components}
        self.prior = prior

        self.step_size = step_size

        self.word_rho = word_rho
        self.word_z = _init((n_words, self.n_components)).to(device)
        self._word_weight = _weight_fn(self.word_z, 'word', self.prior['word'])
        self.verbose = verbose

    def _e_step(self, G, mean_change_tol=1e-3, max_iters=100, word_ids=slice(None)):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        word_weight = self._word_weight[word_ids].to(G.device)
        doc_z = _init((G.num_nodes('doc'), self.n_components)).to(G.device)

        for i in range(max_iters):
            last_doc_z, doc_z = doc_z, _message_passing(
                G, 'doc', self.step_size['doc'],
                _weight_fn(doc_z, 'doc', self.prior['doc']),
                word_weight, 'z'
            )

            mean_change = (doc_z - last_doc_z).abs().mean()
            if mean_change < mean_change_tol:
                break
        if self.verbose:
            print(f'e-step num_iters={i+1} with mean_change={mean_change:.4f}')
        return {'z': doc_z, 'weight': _weight_fn(doc_z, 'doc', self.prior['doc'])}

    transform = _e_step

    def _m_step(self, G, doc_data, word_ids=slice(None)):
        """_m_step implements word data sampling and stores word_z stats
        """
        word_z = _message_passing(
            G, 'word', self.step_size['word'],
            doc_data['weight'], self._word_weight[word_ids].to(G.device), 'z'
        )
        self.word_z[word_ids] = (
            (1 - self.word_rho) * self.word_z[word_ids]
            + self.word_rho * word_z.to(self.word_z.device)
        )
        # alone full word direction
        self._word_weight = _weight_fn(self.word_z, 'word', self.prior['word'])

    def partial_fit(self, G, word_ids=slice(None)):
        last_word_z = self.word_z[word_ids].clone()

        doc_data = self._e_step(G, word_ids=word_ids)
        self._m_step(G, doc_data, word_ids)

        self._last_mean_change = (self.word_z[word_ids] - last_word_z).abs().mean()

        if self.verbose:
            print(f"m-step mean_change: {self._last_mean_change:.4f}, "
                  f"perplexity: {self.perplexity(G, doc_data, word_ids=word_ids):.1f}")
        return self

    def fit(self, G, mean_change_tol=1e-3, max_epochs=10, word_ids=slice(None)):
        for i in range(max_epochs):
            if self.verbose:
                print(f'epoch {i+1}, ', end='')
            self.partial_fit(G, word_ids=word_ids)
            if self._last_mean_change < mean_change_tol:
                break
        return self

    def perplexity(self, G, doc_data=None, word_ids=slice(None)):
        """ppl = exp{-sum[log(p(w1,...,wn|d))] / n}
        Follows Eq (15) in Hoffman et al., 2010.
        """
        if doc_data is None:
            doc_data = self._e_step(G, word_ids=word_ids)

        word_data = {'z': self.word_z, 'weight': self._word_weight}

        edge_elbo = _message_passing(
            G, 'doc', self.step_size['doc'],
            doc_data['weight'], word_data['weight'][word_ids].to(G.device),
            'edge_elbo'
        )

        # compute E[log p(docs | theta, beta)]
        edge_elbo = (edge_elbo.sum() / doc_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'neg_elbo phi: {-edge_elbo:.3f}', end=' ')

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        doc_elbo = (
            (-doc_data['z'] * doc_data['weight'].log()).sum(axis=1)
            -_lbeta(self.prior['doc'] + doc_data['z'] * 0, axis=1)
            +_lbeta(self.prior['doc'] + doc_data['z'], axis=1)
        )
        doc_elbo = (doc_elbo.sum() / doc_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'theta: {-doc_elbo:.3f}', end=' ')

        # compute E[log p(beta | eta) - log q (beta | lambda)]
        word_elbo = (
            (-word_data['z'] * word_data['weight'].log()).sum(axis=0)
            -_lbeta(self.prior['word'] + word_data['z'] * 0, axis=0)
            +_lbeta(self.prior['word'] + word_data['z'], axis=0)
        )
        word_elbo = (word_elbo.sum() / word_data['z'].sum()).cpu().numpy()
        if self.verbose:
            print(f'beta: {-word_elbo:.3f}')

        return np.exp(-edge_elbo - doc_elbo - word_elbo)


if __name__ == '__main__':
    print('Testing LatentDirichletAllocation ...')
    G = dgl.heterograph({('doc', '', 'word'): [(0, 0), (0, 3)]})
    model = LatentDirichletAllocation(G.num_nodes('word'), 5, verbose=False)
    model.fit(G)
    model.transform(G)
    model.perplexity(G)

    dataloader = dgl.dataloading.NodeDataLoader(
        G.reverse(),
        {'doc': np.arange(G.num_nodes('doc'))},
        dgl.dataloading.MultiLayerFullNeighborSampler(1))

    for input_nodes, output_nodes, blocks in dataloader:
        word2doc = blocks[0].adjacency_matrix()._indices().T.numpy().tolist()
        G_batch = dgl.heterograph({('word', '', 'doc'): word2doc}).reverse()
        model.partial_fit(G_batch, word_ids=input_nodes['word'])
    print('Testing LatentDirichletAllocation passed!')
