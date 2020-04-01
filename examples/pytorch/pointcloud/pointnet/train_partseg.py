import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ShapeNet
import dgl
from pointnet_partseg import PointNetPartSeg, PartSegLoss
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

def collate(samples):
    return dgl.batch(samples)

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
        for g in tq:
            num_examples = g.batch_size
            g.ndata['x'] = g.ndata['x'].to(dev, dtype=torch.float)
            g.ndata['y'] = g.ndata['y'].to(dev, dtype=torch.long)
            opt.zero_grad()
            logits = net(g)
            loss = L(logits, g.ndata['y'])
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            num_batches += 1
            count += num_examples * 2048
            loss = loss.item()
            correct = (preds.view(-1) == g.ndata['y']).sum().item()
            import pdb; pdb.set_trace()
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
            for g in tq:
                num_examples = g.batch_size
                g.ndata['x'] = g.ndata['x'].to(dev, dtype=torch.float)
                g.ndata['y'] = g.ndata['y'].to(dev, dtype=torch.long)
                logits = net(g)
                _, preds = logits.max(1)

                correct = (preds.view(-1) == g.ndata['y']).sum().item()
                total_correct += correct
                count += num_examples * 2048

                tq.set_postfix({
                    'AvgAcc': '%.5f' % (total_correct / count)})

    return total_correct / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = "cpu"

net = PointNetPartSeg(50, 6, 2048)
net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)
L = PartSegLoss()

modelnet = ShapeNet(2048)

train_loader = CustomDataLoader(modelnet.trainval())
test_loader = CustomDataLoader(modelnet.test())

best_test_acc = 0

for epoch in range(args.num_epochs):
    train(net, opt, scheduler, train_loader, dev)
    if (epoch + 1) % 5 == 0:
        print('Epoch #%d Testing' % epoch)
        test_acc = evaluate(net, test_loader, dev)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_model_path:
                torch.save(net.state_dict(), args.save_model_path)
        print('Current test acc: %.5f (best: %.5f)' % (
               test_acc, best_test_acc))
