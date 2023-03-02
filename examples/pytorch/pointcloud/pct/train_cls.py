import argparse
import os
import time
from functools import partial

import provider
import torch
import torch.nn as nn
import tqdm

from dgl.data.utils import download, get_download_dir
from ModelNetDataLoader import ModelNetDataLoader
from pct import PointTransformerCLS
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="")
parser.add_argument("--load-model-path", type=str, default="")
parser.add_argument("--save-model-path", type=str, default="")
parser.add_argument("--num-epochs", type=int, default=250)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size

data_filename = "modelnet40_normal_resampled.zip"
download_path = os.path.join(get_download_dir(), data_filename)
local_path = args.dataset_path or os.path.join(
    get_download_dir(), "modelnet40_normal_resampled"
)

if not os.path.exists(local_path):
    download(
        "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip",
        download_path,
        verify_ssl=False,
    )
    from zipfile import ZipFile

    with ZipFile(download_path) as z:
        z.extractall(path=get_download_dir())

CustomDataLoader = partial(
    DataLoader,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


def train(net, opt, scheduler, train_loader, dev):
    net.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    loss_f = nn.CrossEntropyLoss()
    start_time = time.time()
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            data = data.data.numpy()
            data = provider.random_point_dropout(data)
            data[:, :, 0:3] = provider.random_scale_point_cloud(data[:, :, 0:3])
            data[:, :, 0:3] = provider.jitter_point_cloud(data[:, :, 0:3])
            data[:, :, 0:3] = provider.shift_point_cloud(data[:, :, 0:3])
            data = torch.tensor(data)
            label = label[:, 0]

            num_examples = label.shape[0]
            data, label = data.to(dev), label.to(dev).squeeze().long()
            opt.zero_grad()
            logits = net(data)
            loss = loss_f(logits, label)
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            num_batches += 1
            count += num_examples
            loss = loss.item()
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix(
                {
                    "AvgLoss": "%.5f" % (total_loss / num_batches),
                    "AvgAcc": "%.5f" % (total_correct / count),
                }
            )
    print(
        "[Train] AvgLoss: {:.5}, AvgAcc: {:.5}, Time: {:.5}s".format(
            total_loss / num_batches,
            total_correct / count,
            time.time() - start_time,
        )
    )
    scheduler.step()


def evaluate(net, test_loader, dev):
    net.eval()

    total_correct = 0
    count = 0
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq:
                label = label[:, 0]
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev).squeeze().long()
                logits = net(data)
                _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix({"AvgAcc": "%.5f" % (total_correct / count)})
    print(
        "[Test]  AvgAcc: {:.5}, Time: {:.5}s".format(
            total_correct / count, time.time() - start_time
        )
    )
    return total_correct / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PointTransformerCLS()

net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))


opt = torch.optim.SGD(
    net.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.num_epochs
)

train_dataset = ModelNetDataLoader(local_path, 1024, split="train")
test_dataset = ModelNetDataLoader(local_path, 1024, split="test")
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=True,
)

best_test_acc = 0

for epoch in range(args.num_epochs):
    print("Epoch #{}: ".format(epoch))
    train(net, opt, scheduler, train_loader, dev)
    if (epoch + 1) % 1 == 0:
        test_acc = evaluate(net, test_loader, dev)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_model_path:
                torch.save(net.state_dict(), args.save_model_path)
        print("Current test acc: %.5f (best: %.5f)" % (test_acc, best_test_acc))
    print()
