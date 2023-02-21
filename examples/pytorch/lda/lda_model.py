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


import collections
import functools
import io
import os
import warnings

import dgl

import numpy as np
import scipy as sp
import torch

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

    @property
    def loglike(self):
        return (self.src_data["Elog"] + self.dst_data["Elog"]).logsumexp(1)

    @property
    def phi(self):
        return (
            self.src_data["Elog"]
            + self.dst_data["Elog"]
            - self.loglike.unsqueeze(1)
        ).exp()

    @property
    def expectation(self):
        return (
            self.src_data["expectation"] * self.dst_data["expectation"]
        ).sum(1)


class _Dirichlet:
    def __init__(self, prior, nphi, _chunksize=int(1e6)):
        self.prior = prior
        self.nphi = nphi
        self.device = nphi.device
        self._sum_by_parts = lambda map_fn: functools.reduce(
            torch.add,
            [
                map_fn(slice(i, min(i + _chunksize, nphi.shape[1]))).sum(1)
                for i in list(range(0, nphi.shape[1], _chunksize))
            ],
        )

    def _posterior(self, _ID=slice(None)):
        return self.prior + self.nphi[:, _ID]

    @cached_property
    def posterior_sum(self):
        return self.nphi.sum(1) + self.prior * self.nphi.shape[1]

    def _Elog(self, _ID=slice(None)):
        return torch.digamma(self._posterior(_ID)) - torch.digamma(
            self.posterior_sum.unsqueeze(1)
        )

    @cached_property
    def loglike(self):
        neg_evid = -self._sum_by_parts(
            lambda s: (self.nphi[:, s] * self._Elog(s))
        )

        prior = torch.as_tensor(self.prior).to(self.nphi)
        K = self.nphi.shape[1]
        log_B_prior = torch.lgamma(prior) * K - torch.lgamma(prior * K)

        log_B_posterior = self._sum_by_parts(
            lambda s: torch.lgamma(self._posterior(s))
        ) - torch.lgamma(self.posterior_sum)

        return neg_evid - log_B_prior + log_B_posterior

    @cached_property
    def n(self):
        return self.nphi.sum(1)

    @cached_property
    def cdf(self):
        cdf = self._posterior()
        torch.cumsum(cdf, 1, out=cdf)
        cdf /= cdf[:, -1:].clone()
        return cdf

    def _expectation(self, _ID=slice(None)):
        expectation = self._posterior(_ID)
        expectation /= self.posterior_sum.unsqueeze(1)
        return expectation

    @cached_property
    def Bayesian_gap(self):
        return 1.0 - self._sum_by_parts(lambda s: self._Elog(s).exp())

    _cached_properties = [
        "posterior_sum",
        "loglike",
        "n",
        "cdf",
        "Bayesian_gap",
    ]

    def clear_cache(self):
        for name in self._cached_properties:
            try:
                delattr(self, name)
            except AttributeError:
                pass

    def update(self, new, _ID=slice(None), rho=1):
        """inplace: old * (1-rho) + new * rho"""
        self.clear_cache()
        mean_change = (self.nphi[:, _ID] - new).abs().mean().tolist()

        self.nphi *= 1 - rho
        self.nphi[:, _ID] += new * rho
        return mean_change


class DocData(_Dirichlet):
    """nphi (n_docs by n_topics)"""

    def prepare_graph(self, G, key="Elog"):
        G.nodes["doc"].data[key] = getattr(self, "_" + key)().to(G.device)

    def update_from(self, G, mult):
        new = G.nodes["doc"].data["nphi"] * mult
        return self.update(new.to(self.device))


class _Distributed(collections.UserList):
    """split on dim=0 and store on multiple devices"""

    def __init__(self, prior, nphi):
        self.prior = prior
        self.nphi = nphi
        super().__init__([_Dirichlet(self.prior, nphi) for nphi in self.nphi])

    def split_device(self, other, dim=0):
        split_sections = [x.shape[0] for x in self.nphi]
        out = torch.split(other, split_sections, dim)
        return [y.to(x.device) for x, y in zip(self.nphi, out)]


class WordData(_Distributed):
    """distributed nphi (n_topics by n_words), transpose to/from graph nodes data"""

    def prepare_graph(self, G, key="Elog"):
        if "_ID" in G.nodes["word"].data:
            _ID = G.nodes["word"].data["_ID"]
        else:
            _ID = slice(None)

        out = [getattr(part, "_" + key)(_ID).to(G.device) for part in self]
        G.nodes["word"].data[key] = torch.cat(out).T

    def update_from(self, G, mult, rho):
        nphi = G.nodes["word"].data["nphi"].T * mult

        if "_ID" in G.nodes["word"].data:
            _ID = G.nodes["word"].data["_ID"]
        else:
            _ID = slice(None)

        mean_change = [
            x.update(y, _ID, rho) for x, y in zip(self, self.split_device(nphi))
        ]
        return np.mean(mean_change)


class Gamma(collections.namedtuple("Gamma", "concentration, rate")):
    """articulate the difference between torch gamma and numpy gamma"""

    @property
    def shape(self):
        return self.concentration

    @property
    def scale(self):
        return 1 / self.rate

    def sample(self, shape, device):
        return torch.distributions.gamma.Gamma(
            torch.as_tensor(self.concentration, device=device),
            torch.as_tensor(self.rate, device=device),
        ).sample(shape)


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
        self,
        n_words,
        n_components,
        prior=None,
        rho=1,
        mult={"doc": 1, "word": 1},
        init={"doc": (100.0, 100.0), "word": (100.0, 100.0)},
        device_list=["cpu"],
        verbose=True,
    ):
        self.n_words = n_words
        self.n_components = n_components

        if prior is None:
            prior = {"doc": 1.0 / n_components, "word": 1.0 / n_components}
        self.prior = prior

        self.rho = rho
        self.mult = mult
        self.init = init

        assert not isinstance(device_list, str), "plz wrap devices in a list"
        self.device_list = device_list[:n_components]  # avoid edge cases
        self.verbose = verbose

        self._init_word_data()

    def _init_word_data(self):
        split_sections = np.diff(
            np.linspace(0, self.n_components, len(self.device_list) + 1).astype(
                int
            )
        )
        word_nphi = [
            Gamma(*self.init["word"]).sample((s, self.n_words), device)
            for s, device in zip(split_sections, self.device_list)
        ]
        self.word_data = WordData(self.prior["word"], word_nphi)

    def _init_doc_data(self, n_docs, device):
        doc_nphi = Gamma(*self.init["doc"]).sample(
            (n_docs, self.n_components), device
        )
        return DocData(self.prior["doc"], doc_nphi)

    def save(self, f):
        for w in self.word_data:
            w.clear_cache()
        torch.save(
            {
                "prior": self.prior,
                "rho": self.rho,
                "mult": self.mult,
                "init": self.init,
                "word_data": [part.nphi for part in self.word_data],
            },
            f,
        )

    def _prepare_graph(self, G, doc_data, key="Elog"):
        doc_data.prepare_graph(G, key)
        self.word_data.prepare_graph(G, key)

    def _e_step(self, G, doc_data=None, mean_change_tol=1e-3, max_iters=100):
        """_e_step implements doc data sampling until convergence or max_iters"""
        if doc_data is None:
            doc_data = self._init_doc_data(G.num_nodes("doc"), G.device)

        G_rev = G.reverse()  # word -> doc
        self.word_data.prepare_graph(G_rev)

        for i in range(max_iters):
            doc_data.prepare_graph(G_rev)
            G_rev.update_all(
                lambda edges: {"phi": EdgeData(edges.src, edges.dst).phi},
                dgl.function.sum("phi", "nphi"),
            )
            mean_change = doc_data.update_from(G_rev, self.mult["doc"])
            if mean_change < mean_change_tol:
                break

        if self.verbose:
            print(
                f"e-step num_iters={i+1} with mean_change={mean_change:.4f}, "
                f"perplexity={self.perplexity(G, doc_data):.4f}"
            )

        return doc_data

    transform = _e_step

    def predict(self, doc_data):
        pred_scores = [
            # d_exp @ w._expectation()
            (lambda x: x @ w.nphi + x.sum(1, keepdims=True) * w.prior)(
                d_exp / w.posterior_sum.unsqueeze(0)
            )
            for (d_exp, w) in zip(
                self.word_data.split_device(doc_data._expectation(), dim=1),
                self.word_data,
            )
        ]
        x = torch.zeros_like(pred_scores[0], device=doc_data.device)
        for p in pred_scores:
            x += p.to(x.device)
        return x

    def sample(self, doc_data, num_samples):
        """draw independent words and return the marginal probabilities,
        i.e., the expectations in Dirichlet distributions.
        """

        def fn(cdf):
            u = torch.rand(cdf.shape[0], num_samples, device=cdf.device)
            return torch.searchsorted(cdf, u).to(doc_data.device)

        topic_ids = fn(doc_data.cdf)
        word_ids = torch.cat([fn(part.cdf) for part in self.word_data])
        ids = torch.gather(
            word_ids, 0, topic_ids
        )  # pick components by topic_ids

        # compute expectation scores on sampled ids
        src_ids = (
            torch.arange(ids.shape[0], dtype=ids.dtype, device=ids.device)
            .reshape((-1, 1))
            .expand(ids.shape)
        )
        unique_ids, inverse_ids = torch.unique(
            ids, sorted=False, return_inverse=True
        )

        G = dgl.heterograph(
            {("doc", "", "word"): (src_ids.ravel(), inverse_ids.ravel())}
        )
        G.nodes["word"].data["_ID"] = unique_ids
        self._prepare_graph(G, doc_data, "expectation")
        G.apply_edges(
            lambda e: {"expectation": EdgeData(e.src, e.dst).expectation}
        )
        expectation = G.edata.pop("expectation").reshape(ids.shape)

        return ids, expectation

    def _m_step(self, G, doc_data):
        """_m_step implements word data sampling and stores word_z stats.
        mean_change is in the sense of full graph with rho=1.
        """
        G = G.clone()
        self._prepare_graph(G, doc_data)
        G.update_all(
            lambda edges: {"phi": EdgeData(edges.src, edges.dst).phi},
            dgl.function.sum("phi", "nphi"),
        )
        self._last_mean_change = self.word_data.update_from(
            G, self.mult["word"], self.rho
        )

        if self.verbose:
            print(f"m-step mean_change={self._last_mean_change:.4f}, ", end="")
            Bayesian_gap = np.mean(
                [part.Bayesian_gap.mean().tolist() for part in self.word_data]
            )
            print(f"Bayesian_gap={Bayesian_gap:.4f}")

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
        G.apply_edges(
            lambda edges: {"loglike": EdgeData(edges.src, edges.dst).loglike}
        )
        edge_elbo = (G.edata["loglike"].sum() / G.num_edges()).tolist()
        if self.verbose:
            print(f"neg_elbo phi: {-edge_elbo:.3f}", end=" ")

        # compute E[log p(theta | alpha) - log q(theta | gamma)]
        doc_elbo = (doc_data.loglike.sum() / doc_data.n.sum()).tolist()
        if self.verbose:
            print(f"theta: {-doc_elbo:.3f}", end=" ")

        # compute E[log p(beta | eta) - log q(beta | lambda)]
        # The denominator n for extrapolation perplexity is undefined.
        # We use the train set, whereas sklearn uses the test set.
        word_elbo = sum(
            [part.loglike.sum().tolist() for part in self.word_data]
        ) / sum([part.n.sum().tolist() for part in self.word_data])
        if self.verbose:
            print(f"beta: {-word_elbo:.3f}")

        ppl = np.exp(-edge_elbo - doc_elbo - word_elbo)
        if G.num_edges() > 0 and np.isnan(ppl):
            warnings.warn("numerical issue in perplexity")
        return ppl


def doc_subgraph(G, doc_ids):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    _, _, (block,) = sampler.sample(
        G.reverse(), {"doc": torch.as_tensor(doc_ids)}
    )
    B = dgl.DGLGraph(
        block._graph, ["_", "word", "doc", "_"], block.etypes
    ).reverse()
    B.nodes["word"].data["_ID"] = block.nodes["word"].data["_ID"]
    return B


if __name__ == "__main__":
    print("Testing LatentDirichletAllocation ...")
    G = dgl.heterograph(
        {("doc", "", "word"): [(0, 0), (1, 3)]}, {"doc": 2, "word": 5}
    )
    model = LatentDirichletAllocation(n_words=5, n_components=10, verbose=False)
    model.fit(G)
    model.transform(G)
    model.predict(model.transform(G))
    if hasattr(torch, "searchsorted"):
        model.sample(model.transform(G), 3)
    model.perplexity(G)

    for doc_id in range(2):
        B = doc_subgraph(G, [doc_id])
        model.partial_fit(B)

    with io.BytesIO() as f:
        model.save(f)
        f.seek(0)
        print(torch.load(f))

    print("Testing LatentDirichletAllocation passed!")
