import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import dgl
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse

from dataset import ShapeNet
from pointnet_partseg import PointNetPartSeg, PartSegLoss
from pointnet2 import FarthestPointSampler, EpsBallPoints

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
    graphs, cat = map(list, zip(*samples))
    return dgl.batch(graphs), cat

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
        for g, _ in tq:
            num_examples = g.batch_size
            g.ndata['x'] = g.ndata['x'].to(dev, dtype=torch.float)
            g.ndata['y'] = g.ndata['y'].to(dev, dtype=torch.long)
            opt.zero_grad()
            logits = net(g)
            loss = L(logits, g.ndata['y'])
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            count += num_examples * 2048
            loss = loss.item()
            total_loss += loss
            num_batches += 1
            correct = (preds.view(-1) == g.ndata['y']).sum().item()
            total_correct += correct

            tq.set_postfix({
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AvgAcc': '%.5f' % (total_correct / count)})
    scheduler.step()

def mIoU(preds, label, cat, cat_miou, seg_classes):
    n_cat = len(seg_classes)
    for i in range(preds.shape[0]):
        shape_iou = 0
        n = len(seg_classes[cat[i]])
        for cls in seg_classes[cat[i]]:
            gt_ind = label[i,:] == cls
            pred_set = set(np.where(preds[i,:] == cls)[0])
            label_set = set(np.where(label[i,:] == cls)[0])
            union = len(pred_set.union(label_set))
            inter = len(pred_set.intersection(label_set))
            if union == 0:
                shape_iou += 1
            else:
                shape_iou += inter / union
        shape_iou /= n
        cat_miou[cat[i]][0] += shape_iou
        cat_miou[cat[i]][1] += 1

    return cat_miou

def evaluate(net, test_loader, dev):
    net.eval()

    cat_miou = {}
    for k in shapenet.seg_classes.keys():
        cat_miou[k] = [0, 0]
    miou = 0
    count = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for g, category in tq:
                num_examples = g.batch_size
                g.ndata['x'] = g.ndata['x'].to(dev, dtype=torch.float)
                g.ndata['y'] = g.ndata['y'].to(dev, dtype=torch.long)
                logits = net(g)
                _, preds = logits.max(1)

                cat_miou = mIoU(preds.cpu().numpy(),
                                g.ndata['y'].view(num_examples, -1).cpu().numpy(),
                                category, cat_miou, shapenet.seg_classes)
                for _, v in cat_miou.items():
                    if v[1] > 0:
                        miou += v[0] / v[1]
                        count += 1
                tq.set_postfix({
                    'mIoU': '%.5f' % (miou / count)})
    return miou / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = "cpu"

net = PointNetPartSeg(50, 6, 2048)
net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)
L = PartSegLoss()

shapenet = ShapeNet(2048)

train_loader = CustomDataLoader(shapenet.trainval())
test_loader = CustomDataLoader(shapenet.test())

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
