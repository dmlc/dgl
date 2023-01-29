#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""

import math
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm
from utils import Timer

from .faiss_search import faiss_search_knn

__all__ = [
    "knn_faiss",
    "knn_faiss_gpu",
    "fast_knns2spmat",
    "build_knns",
    "knns2ordered_nbrs",
]


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


def fast_knns2spmat(knns, k, th_sim=0, use_sim=True, fill_value=None):
    # convert knns to symmetric sparse matrix
    from scipy.sparse import csr_matrix

    eps = 1e-5
    n = len(knns)
    if isinstance(knns, list):
        knns = np.array(knns)
    if len(knns.shape) == 2:
        # knns saved by hnsw has different shape
        n = len(knns)
        ndarr = np.ones([n, 2, k])
        ndarr[:, 0, :] = -1  # assign unknown dist to 1 and nbr to -1
        for i, (nbr, dist) in enumerate(knns):
            size = len(nbr)
            assert size == len(dist)
            ndarr[i, 0, :size] = nbr[:size]
            ndarr[i, 1, :size] = dist[:size]
        knns = ndarr
    nbrs = knns[:, 0, :]
    dists = knns[:, 1, :]
    assert (
        -eps <= dists.min() <= dists.max() <= 1 + eps
    ), "min: {}, max: {}".format(dists.min(), dists.max())
    if use_sim:
        sims = 1.0 - dists
    else:
        sims = dists
    if fill_value is not None:
        print("[fast_knns2spmat] edge fill value:", fill_value)
        sims.fill(fill_value)
    row, col = np.where(sims >= th_sim)
    # remove the self-loop
    idxs = np.where(row != nbrs[row, col])
    row = row[idxs]
    col = col[idxs]
    data = sims[row, col]
    col = nbrs[row, col]  # convert to absolute column
    assert len(row) == len(col) == len(data)
    spmat = csr_matrix((data, (row, col)), shape=(n, n))
    return spmat


def build_knns(feats, k, knn_method, dump=True):
    with Timer("build index"):
        if knn_method == "faiss":
            index = knn_faiss(feats, k, omp_num_threads=None)
        elif knn_method == "faiss_gpu":
            index = knn_faiss_gpu(feats, k)
        else:
            raise KeyError(
                "Only support faiss and faiss_gpu currently ({}).".format(
                    knn_method
                )
            )
        knns = index.get_knns()
    return knns


class knn:
    def __init__(self, feats, k, index_path="", verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return (th_nbrs, th_dists)

    def get_knns(self, th=None):
        if th is None or th <= 0.0:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer(
            "filter edges by th {} (CPU={})".format(th, nproc), self.verbose
        ):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot)
                )
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    def __init__(
        self,
        feats,
        k,
        nprobe=128,
        omp_num_threads=None,
        rebuild_index=True,
        verbose=True,
        **kwargs
    ):
        import faiss

        if omp_num_threads is not None:
            faiss.omp_set_num_threads(omp_num_threads)
        self.verbose = verbose
        with Timer("[faiss] build index", verbose):
            feats = feats.astype("float32")
            size, dim = feats.shape
            index = faiss.IndexFlatIP(dim)
            index.add(feats)
        with Timer("[faiss] query topk {}".format(k), verbose):
            sims, nbrs = index.search(feats, k=k)
            self.knns = [
                (
                    np.array(nbr, dtype=np.int32),
                    1 - np.array(sim, dtype=np.float32),
                )
                for nbr, sim in zip(nbrs, sims)
            ]


class knn_faiss_gpu(knn):
    def __init__(
        self,
        feats,
        k,
        nprobe=128,
        num_process=4,
        is_precise=True,
        sort=True,
        verbose=True,
        **kwargs
    ):
        with Timer("[faiss_gpu] query topk {}".format(k), verbose):
            dists, nbrs = faiss_search_knn(
                feats,
                k=k,
                nprobe=nprobe,
                num_process=num_process,
                is_precise=is_precise,
                sort=sort,
                verbose=verbose,
            )

            self.knns = [
                (
                    np.array(nbr, dtype=np.int32),
                    np.array(dist, dtype=np.float32),
                )
                for nbr, dist in zip(nbrs, dists)
            ]
