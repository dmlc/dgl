import argparse
import os
import urllib
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from dgl.data.utils import download, get_download_dir
from model import compute_loss, Model
from modelnet import ModelNet
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default="")
parser.add_argument("--load-model-path", type=str, default="")
parser.add_argument("--save-model-path", type=str, default="")
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
data_filename = "modelnet40-sampled-2048.h5"
local_path = args.dataset_path or os.path.join(
    get_download_dir(), data_filename
)

if not os.path.exists(local_path):
    download(
        "https://data.dgl.ai/dataset/modelnet40-sampled-2048.h5", local_path
    )

CustomDataLoader = partial(
    DataLoader,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


def train(model, opt, scheduler, train_loader, dev):
    scheduler.step()

    model.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label in tq:
            num_examples = label.shape[0]
            data, label = data.to(dev), label.to(dev).squeeze().long()
            opt.zero_grad()
            logits = model(data)
            loss = compute_loss(logits, label)
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
                    "Loss": "%.5f" % loss,
                    "AvgLoss": "%.5f" % (total_loss / num_batches),
                    "Acc": "%.5f" % (correct / num_examples),
                    "AvgAcc": "%.5f" % (total_correct / count),
                }
            )


def evaluate(model, test_loader, dev):
    model.eval()

    total_correct = 0
    count = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label in tq:
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev).squeeze().long()
                logits = model(data)
                _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix(
                    {
                        "Acc": "%.5f" % (correct / num_examples),
                        "AvgAcc": "%.5f" % (total_correct / count),
                    }
                )

    return total_correct / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(20, [64, 64, 128, 256], [512, 512, 256], 40)
model = model.to(dev)
if args.load_model_path:
    model.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    opt, args.num_epochs, eta_min=0.001
)

modelnet = ModelNet(local_path, 1024)

train_loader = CustomDataLoader(modelnet.train())
valid_loader = CustomDataLoader(modelnet.valid())
test_loader = CustomDataLoader(modelnet.test())

best_valid_acc = 0
best_test_acc = 0

for epoch in range(args.num_epochs):
    print("Epoch #%d Validating" % epoch)
    valid_acc = evaluate(model, valid_loader, dev)
    test_acc = evaluate(model, test_loader, dev)
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_test_acc = test_acc
        if args.save_model_path:
            torch.save(model.state_dict(), args.save_model_path)
    print(
        "Current validation acc: %.5f (best: %.5f), test acc: %.5f (best: %.5f)"
        % (valid_acc, best_valid_acc, test_acc, best_test_acc)
    )

    train(model, opt, scheduler, train_loader, dev)
