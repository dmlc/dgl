"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""
import gc

from tqdm import tqdm

from .faiss_gpu import faiss_search_approx_knn

__all__ = ["faiss_search_knn"]


def precise_dist(feat, nbrs, num_process=4, sort=True, verbose=False):
    import torch

    feat_share = torch.from_numpy(feat).share_memory_()
    nbrs_share = torch.from_numpy(nbrs).share_memory_()
    dist_share = torch.zeros_like(nbrs_share).float().share_memory_()

    precise_dist_share_mem(
        feat_share,
        nbrs_share,
        dist_share,
        num_process=num_process,
        sort=sort,
        verbose=verbose,
    )

    del feat_share
    gc.collect()
    return dist_share.numpy(), nbrs_share.numpy()


def precise_dist_share_mem(
    feat,
    nbrs,
    dist,
    num_process=16,
    sort=True,
    process_unit=4000,
    verbose=False,
):
    from torch import multiprocessing as mp

    num, _ = feat.shape
    num_per_proc = int(num / num_process) + 1

    for pi in range(num_process):
        sid = pi * num_per_proc
        eid = min(sid + num_per_proc, num)

        kwargs = {
            "feat": feat,
            "nbrs": nbrs,
            "dist": dist,
            "sid": sid,
            "eid": eid,
            "sort": sort,
            "process_unit": process_unit,
            "verbose": verbose,
        }
        bmm(**kwargs)


def bmm(
    feat, nbrs, dist, sid, eid, sort=True, process_unit=4000, verbose=False
):
    import torch

    _, cols = dist.shape
    batch_sim = torch.zeros((eid - sid, cols), dtype=torch.float32)
    for s in tqdm(
        range(sid, eid, process_unit), desc="bmm", disable=not verbose
    ):
        e = min(eid, s + process_unit)
        query = feat[s:e].unsqueeze(1)
        gallery = feat[nbrs[s:e]].permute(0, 2, 1)
        batch_sim[s - sid : e - sid] = torch.clamp(
            torch.bmm(query, gallery).view(-1, cols), 0.0, 1.0
        )

    if sort:
        sort_unit = int(1e6)
        batch_nbr = nbrs[sid:eid]
        for s in range(0, batch_sim.shape[0], sort_unit):
            e = min(s + sort_unit, eid)
            batch_sim[s:e], indices = torch.sort(
                batch_sim[s:e], descending=True
            )
            batch_nbr[s:e] = torch.gather(batch_nbr[s:e], 1, indices)
        nbrs[sid:eid] = batch_nbr
    dist[sid:eid] = 1.0 - batch_sim


def faiss_search_knn(
    feat,
    k,
    nprobe=128,
    num_process=4,
    is_precise=True,
    sort=True,
    verbose=False,
):
    dists, nbrs = faiss_search_approx_knn(
        query=feat, target=feat, k=k, nprobe=nprobe, verbose=verbose
    )

    if is_precise:
        print("compute precise dist among k={} nearest neighbors".format(k))
        dists, nbrs = precise_dist(
            feat, nbrs, num_process=num_process, sort=sort, verbose=verbose
        )

    return dists, nbrs
