import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from .. import rgcn, utils


@utils.benchmark("acc", timeout=1200)
@utils.parametrize("dataset", ["aifb", "mutag"])
@utils.parametrize("ns_mode", [False])
def track_acc(dataset, ns_mode):
    (
        g,
        num_rels,
        num_classes,
        labels,
        train_idx,
        test_idx,
        target_idx,
    ) = rgcn.load_data(dataset, get_norm=True)
    num_hidden = 16
    if dataset == "aifb":
        num_bases = -1
        l2norm = 0.0
    elif dataset == "mutag":
        num_bases = 30
        l2norm = 5e-4
    elif dataset == "am":
        num_bases = 40
        l2norm = 5e-4
    else:
        raise ValueError()
    model = rgcn.RGCN(
        g.num_nodes(),
        num_hidden,
        num_classes,
        num_rels,
        num_bases=num_bases,
        ns_mode=ns_mode,
    )
    device = utils.get_bench_device()
    labels = labels.to(device)
    model = model.to(device)
    g = g.int().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=l2norm
    )

    model.train()
    for epoch in range(30):
        logits = model(g)
        logits = logits[target_idx]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(g)
    logits = logits[target_idx]
    test_acc = accuracy(
        logits[test_idx].argmax(dim=1),
        labels[test_idx],
        task="multiclass",
        num_classes=num_classes,
    ).item()

    return test_acc
