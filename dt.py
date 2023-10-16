import time

import dgl.graphbolt as gb
import torch


def benchmark_itemsampler(use_indexing, shuffle, num_ids, batch_size):
    node_ids = torch.arange(0, num_ids)
    labels = torch.arange(0, num_ids)
    item_set = gb.ItemSet((node_ids, labels), names=("seed_nodes", "labels"))
    item_sampler = gb.ItemSampler(
        item_set,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        use_indexing=use_indexing,
    )
    t_start = time.perf_counter()
    for data in item_sampler:
        pass
    t_end = time.perf_counter()
    print(
        f"`use_indexing={use_indexing} shuffle={shuffle} num_ids={num_ids:,} "
        f"batch_size={batch_size}`: {t_end - t_start:.4f}s"
    )

batch_size = 1000
for num_ids in [10 * 1000, 100 * 1000, 1000 * 1000, 10 * 1000 * 1000]:
    for shuffle in [False, True]:
        for use_indexing in [False, True]:
            benchmark_itemsampler(
                use_indexing=use_indexing,
                shuffle=shuffle,
                num_ids=num_ids,
                batch_size=batch_size,
            )
        print("----------------------------")
