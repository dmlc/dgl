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


import os, functools, warnings, torch, typing, gc, collections
import numpy as np, scipy as sp
import dgl
from dgl import function as fn
try:
    from functools import cached_property
except ImportError:
    try:
        from backports.cached_property import cached_property
    except ImportError:
        warnings.warn("cached_property not found - using property instead")
        cached_property = property


class EdgeData:
    def __init__(self, src_data, dst_data):
        self.src_data = src_data
        self.dst_data = dst_data

    @cached_property
    def loglike(self):
        return (self.src_data['Elog'] + self.dst_data['Elog']).logsumexp(1)

    @cached_property
    def phi(self):
        return (
            self.src_data['Elog'] + self.dst_data['Elog'] - self.loglike.unsqueeze(1)
        ).exp()


class _Dirichlet:
    def __init__(self, prior, nphi):
        self.prior = prior
        self.nphi = nphi
        self.device = nphi.device

    @cached_property
    def posterior(self):
        return self.prior + self.nphi

    @cached_property
    def Elog(self):
        return torch.digamma(self.posterior) - \
               torch.digamma(self.posterior.sum(1, keepdims=True))

    @cached_property
    def loglike(self):
        neg_evid = -(self.nphi * self.Elog).sum(1)

        prior = torch.as_tensor(self.prior).to(self.nphi)
        K = self.nphi.shape[1]
        log_B_prior = torch.lgamma(prior) * K - torch.lgamma(prior * K)

        log_B_posterior = torch.lgamma(self.posterior).sum(1) - \
                          torch.lgamma(self.posterior.sum(1))

        return neg_evid - log_B_prior + log_B_posterior

    @cached_property
    def n(self):
        return self.nphi.sum()

    @cached_property
    def weight(self):
        return self.posterior / self.posterior.sum(1, keepdims=True)

    @cached_property
    def cdf(self):
        cdf = self.posterior.cumsum(1)
        return cdf / cdf[:, -1:]


class DocData(_Dirichlet):
    """ nphi (n_docs by n_topics) """
    def prepare_graph(self, G):
        G.nodes['doc'].data['Elog'] = self.Elog.to(G.device)

    def extract_graph(self, G, mult):
        out = G.nodes['doc'].data['nphi'] * mult
        return self.__class__(self.prior, out.to(self.device))


class _Distributed(collections.UserList):
    """ split on dim=0 and store on multiple devices  """
    def __init__(self, prior, nphi):
        self.prior = prior
        self.nphi = nphi
        self.clear_cache()

    def clear_cache(self, gc_collect=False):
        super().__init__([_Dirichlet(self.prior, nphi) for nphi in self.nphi])
        if gc_collect:
            gc.collect()
            for part in self:
                if part.device.type == 'cuda':
                    with torch.cuda.device(part.device):
                        torch.cuda.empty_cache()

    def split_device(self, other, dim=0):
        split_sections = [x.shape[0] for x in self.nphi]
        out = torch.split(other, split_sections, dim)
        return [y.to(x.device) for x,y in zip(self.nphi, out)]


class WordData(_Distributed):
    """ distributed nphi (n_topics by n_words), transpose to/from graph nodes data """
    def prepare_graph(self, G):
        if '_ID' in G.nodes['word'].data:
            _ID = G.nodes['word'].data['_ID']
            out = [part.Elog[:, _ID.to(part.device)].to(G.device) for part in self]
        else:
            out = [part.Elog.to(G.device) for part in self]

        G.nodes['word'].data['Elog'] = torch.cat(out).T

    def extract_graph(self, G, mult):
        nphi = G.nodes['word'].data['nphi'].T * mult

        if '_ID' in G.nodes['word'].data:
            _ID = G.nodes['word'].data['_ID']

            out = [torch.zeros_like(x) for x in self.nphi]
            for x, y in zip(out, self.split_device(nphi)):
                x[:, _ID.to(x.device)] = y
        else:
            out = self.split_device(nphi)

        return self.__class__(self.prior, out)


class GammaParams(collections.namedtuple('GammaParams', "concentration, rate")):
    """ articulate the difference between torch gamma and numpy gamma """
    @property
    def shape(self):
        return self.concentration

    @property
    def scale(self):
        return 1 / self.rate

    def to(self, device):
        return self.__class__(*[torch.as_tensor(v).to(device) for v in self])


class LatentDirichletAllocation:
    """LDA model that works with a HeteroGraph with doc->word meta paths.
    The model alters the attributes of G arbitrarily.
    This is inspired by [1] and its corresponding scikit-learn implementation.

    Inputs
    ---
    * G: a template graph or an integer showing n_words
    * n_components: latent feature dimension; automatically set priors if missing.
    * prior: parameters in the Dirichlet prior; default to 1/n_components and 1/n_words
    * rho: new_nphi = (1-rho)*old_nphi + rho*nphi; default to 1 for full gradients.
    * mult: multiplier for nphi-update; a large value effectively disables prior.
    * init: sklearn initializers (100.0, 100.0); the sample points concentrate around 1.0
    * device_list: accelerate word_data updates.

    Notes
    ---
    Some differences between this and sklearn.decomposition.LatentDirichletAllocation:
    * default word perplexity is normalized by training set instead of testing set.

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
        rho=1,
        mult={'doc': 1, 'word': 1},
        init={'doc': (100., 100.), 'word': (100., 100.)},
        device_list=['cpu'],
        verbose=True,
        ):
        self.n_words = n_words
        self.n_components = n_components

        if prior is None:
            prior = {'doc': 1./n_components, 'word': 1./n_components}
        self.prior = prior

        self.rho = rho
        self.mult = mult
        self.init = init

        assert not isinstance(device_list, str), "plz wrap devices in a list"
        self.device_list = device_list
        self.verbose = verbose

        self._init_word_data()


    def _init_word_data(self):
        split_sections = np.diff(
            np.linspace(0, self.n_components, len(self.device_list)+1).astype(int)
        )
        word_nphi = [
            torch.distributions.gamma.Gamma(
                *GammaParams(*self.init['word']).to(device)
                ).sample((s, self.n_words))
            for s, device in zip(split_sections, self.device_list)
        ]
        self.word_data = WordData(self.prior['word'], word_nphi)


    def _init_doc_data(self, n_docs, device):
        doc_nphi = torch.distributions.gamma.Gamma(
            *GammaParams(*self.init['doc']).to(device)
            ).sample((n_docs, self.n_components))
        return DocData(self.prior['doc'], doc_nphi)


    def _prepare_graph(self, G, doc_data):
        doc_data.prepare_graph(G)
        self.word_data.prepare_graph(G)


    def _e_step(self, G, doc_data=None, mean_change_tol=1e-3, max_iters=100):
        """_e_step implements doc data sampling until convergence or max_iters
        """
        if doc_data is None:
            doc_data = self._init_doc_data(G.num_nodes('doc'), G.device)

        G_rev = G.reverse() # word -> doc

        for i in range(max_iters):
            self._prepare_graph(G_rev, doc_data)
            G_rev.update_all(
                lambda edges: {'phi': EdgeData(edges.src, edges.dst).phi},
                fn.sum('phi', 'nphi')
            )
            old_nphi = doc_data.nphi
            doc_data = doc_data.extract_graph(G_rev, self.mult['doc'])

            mean_change = (old_nphi - doc_data.nphi).abs().mean()
            if mean_change < mean_change_tol:
                break

        if self.verbose:
            print(f"e-step num_iters={i+1} with mean_change={mean_change:.4f}, "
                  f"perplexity={self.perplexity(G, doc_data):.4f}")

        return doc_data


    transform = _e_step


    def sample(self, doc_data, num_samples):
        def fn(cdf):
            u = torch.rand(cdf.shape[0], num_samples, device=cdf.device)
            return torch.searchsorted(cdf, u).to(doc_data.device)

        topic_ids = fn(doc_data.cdf)
        word_ids = torch.cat([fn(part.cdf) for part in self.word_data])
        return torch.gather(word_ids, 0, topic_ids) # pick components by topic_ids


    def _m_step(self, G, doc_data):
        """_m_step implements word data sampling and stores word_z stats.
        mean_change is in the sense of full graph with rho=1.
        """
        G = G.clone()
        self._prepare_graph(G, doc_data)
        G.update_all(
            lambda edges: {'phi': EdgeData(edges.src, edges.dst).phi},
            fn.sum('phi', 'nphi')
        )
        self.word_data.clear_cache(gc_collect=True)
        word_data = self.word_data.extract_graph(G, self.mult['word'])

        self._last_mean_change = np.mean([
            (old_nphi - nphi).abs().mean().tolist()
            for (old_nphi, nphi) in zip(self.word_data.nphi, word_data.nphi)
            ])
        if self.verbose:
            print(f"m-step mean_change={self._last_mean_change:.4f}, ", end="")

        new_nphi = [
            old_nphi * (1 - self.rho) + nphi * self.rho
            for (old_nphi, nphi) in zip(self.word_data.nphi, word_data.nphi)
        ]
        del word_data
        self.word_data = WordData(self.word_data.prior, new_nphi)

        if self.verbose:
            Bayesian_gap = np.mean([
                part.Elog.exp().sum(1).mean().tolist()
                for part in self.word_data
            ])
            print(f"Bayesian_gap={1 - Bayesian_gap:.4f}")


    def partial_fit(self, G):
        doc_data = self._e_step(G)
        self._m_step(G, doc_data)
        return self


    def fit(self, G, mean_change_tol=1e-3, max_epochs=10):
        for i in range(max_epochs):
            if self.verbose:
                print(f"epoch {i+1}, ", end="")
            self.partial_fit(G)

            if self._last_mean_change < mean_change_tol:
                break
        return self


    def perplexity(self, G, doc_data=None):
        """ppl = exp{-sum[log(p(w1,...,wn|d))] / n}
        Follows Eq (15) in Hoffman et al., 2010.
        """
        if doc_data is None:
            doc_data = self._e_step(G)

        # compute E[log p(docs | theta, beta)]
        G = G.clone()
        self._prepare_graph(G, doc_data)
        G.apply_edges(lambda edges: {'loglike': EdgeData(edges.src, edges.dst).loglike})
        edge_elbo = (G.edata['loglike'].sum() / G.num_edges()).tolist()
        if self.verbose:
            print(f'neg_elbo phi: {-edge_elbo:.3f}', end=' ')

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        doc_elbo = (doc_data.loglike.sum() / doc_data.n).tolist()
        if self.verbose:
            print(f'theta: {-doc_elbo:.3f}', end=' ')

        # compute E[log p(beta | eta) - log q(beta | lambda)]
        # The denominator n for extrapolation perplexity is undefined.
        # We use the train set, whereas sklearn uses the test set.
        word_elbo = (
            sum([part.loglike.sum().tolist() for part in self.word_data])
            / sum([part.n.tolist() for part in self.word_data])
            )
        if self.verbose:
            print(f'beta: {-word_elbo:.3f}')

        ppl = np.exp(-edge_elbo - doc_elbo - word_elbo)
        if G.num_edges()>0 and np.isnan(ppl):
            warnings.warn("numerical issue in perplexity")
        return ppl


def doc_subgraph(G, doc_ids):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    block, *_ = sampler.sample_blocks(G.reverse(), {'doc': torch.as_tensor(doc_ids)})
    B = dgl.DGLHeteroGraph(
        block._graph, ['_', 'word', 'doc', '_'], block.etypes
    ).reverse()
    B.nodes['word'].data['_ID'] = block.nodes['word'].data['_ID']
    return B


if __name__ == '__main__':
    print('Testing LatentDirichletAllocation ...')
    G = dgl.heterograph({('doc', '', 'word'): [(0, 0), (1, 3)]}, {'doc': 2, 'word': 5})
    model = LatentDirichletAllocation(n_words=5, n_components=10, verbose=False)
    model.fit(G)
    model.transform(G)
    if hasattr(torch, "searchsorted"):
        model.sample(model.transform(G), 2)
    model.perplexity(G)

    for doc_id in range(2):
        B = doc_subgraph(G, [doc_id])
        model.partial_fit(B)
    print('Testing LatentDirichletAllocation passed!')
