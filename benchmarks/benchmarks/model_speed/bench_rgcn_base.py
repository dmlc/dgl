import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import rgcn, utils


@utils.benchmark("time", 1200)
@utils.parametrize("data", ["aifb", "am"])
def track_time(data):
    # args
    if data == "aifb":
        num_bases = -1
        l2norm = 0.0
    elif data == "am":
        num_bases = 40
        l2norm = 5e-4
    else:
        raise ValueError()

    (
        g,
        num_rels,
        num_classes,
        labels,
        train_idx,
        test_idx,
        target_idx,
    ) = rgcn.load_data(data, get_norm=True)
    num_hidden = 16

    model = rgcn.RGCN(
        g.num_nodes(), num_hidden, num_classes, num_rels, num_bases=num_bases
    )
    device = utils.get_bench_device()
    labels = labels.to(device)
    model = model.to(device)
    g = g.int().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=l2norm
    )

    model.train()
    num_epochs = 30
    t0 = time.time()
    for epoch in range(num_epochs):
        logits = model(g)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t1 = time.time()

    return (t1 - t0) / num_epochs
