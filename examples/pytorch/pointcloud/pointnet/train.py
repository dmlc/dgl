import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ModelNet
# from model import Model, compute_loss
import dgl
from pointnet import PointNet, compute_loss
from pointnet2 import FarthestPointSampler, EpsBallPoints
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=250)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size
data_filename = 'modelnet40-sampled-2048.h5'
local_path = args.dataset_path or os.path.join(get_download_dir(), data_filename)

if not os.path.exists(local_path):
    download('https://data.dgl.ai/dataset/modelnet40-sampled-2048.h5', local_path)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate)

def train(net, opt, scheduler, train_loader, dev):

    net.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        # for data, label in tq:
        for g, label in tq:
            num_examples = label.shape[0]
            label = label.to(dev).squeeze().long()
            g.ndata['x'] = g.ndata['x'].to(dev)
            tmp = FarthestPointSampler(512)
            res = tmp(g)
            tmp2 = EpsBallPoints(0.2, 64)
            res = tmp2(g)
            opt.zero_grad()
            logits = net(g)
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

            tq.set_postfix({
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AvgAcc': '%.5f' % (total_correct / count)})
    scheduler.step()

def evaluate(net, test_loader, dev):
    net.eval()

    total_correct = 0
    count = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for g, label in tq:
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                g.ndata['x'] = g.ndata['x'].to(dev)
                logits = net(g)
                _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix({
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

    return total_correct / count


# dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev = 'cpu'

# net = EdgeConvModel(1, [64, 64, 128, 256], [512, 512, 256], 40)
net = PointNet(40)
net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))

# opt = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

# scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.num_epochs, eta_min=0.001)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)

modelnet = ModelNet(local_path, 1024)

train_loader = CustomDataLoader(modelnet.train())
valid_loader = CustomDataLoader(modelnet.valid())
test_loader = CustomDataLoader(modelnet.test())

best_valid_acc = 0
best_test_acc = 0

for epoch in range(args.num_epochs):
    train(net, opt, scheduler, train_loader, dev)
    if (epoch + 1) % 5 == 0:
        print('Epoch #%d Validating' % epoch)
        valid_acc = evaluate(net, valid_loader, dev)
        test_acc = evaluate(net, test_loader, dev)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            if args.save_model_path:
                torch.save(net.state_dict(), args.save_model_path)
        print('Current validation acc: %.5f (best: %.5f), test acc: %.5f (best: %.5f)' % (
            valid_acc, best_valid_acc, test_acc, best_test_acc))
