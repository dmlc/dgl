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

from ShapeNet import ShapeNet
from pointnet_partseg import PointNetPartSeg, PartSegLoss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=250)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=16)
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
        drop_last=True)

def train(net, opt, scheduler, train_loader, dev):
    category_list = sorted(list(shapenet.seg_classes.keys()))
    eye_mat = np.eye(16)
    net.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        for data, label, cat in tq:
            num_examples = data.shape[0]
            data = data.to(dev, dtype=torch.float)
            label = label.to(dev, dtype=torch.long).view(-1)
            opt.zero_grad()
            cat_ind = [category_list.index(c) for c in cat]
            # An one-hot encoding for the object category
            cat_tensor = torch.tensor(eye_mat[cat_ind]).to(dev, dtype=torch.float).repeat(1, 2048)
            cat_tensor = cat_tensor.view(num_examples, -1, 16).permute(0,2,1)
            logits = net(data, cat_tensor)
            loss = L(logits, label)
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            count += num_examples * 2048
            loss = loss.item()
            total_loss += loss
            num_batches += 1
            correct = (preds.view(-1) == label).sum().item()
            total_correct += correct

            tq.set_postfix({
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'AvgAcc': '%.5f' % (total_correct / count)})
    scheduler.step()

def mIoU(preds, label, cat, cat_miou, seg_classes):
    for i in range(preds.shape[0]):
        shape_iou = 0
        n = len(seg_classes[cat[i]])
        for cls in seg_classes[cat[i]]:
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

def evaluate(net, test_loader, dev, per_cat_verbose=False):
    category_list = sorted(list(shapenet.seg_classes.keys()))
    eye_mat = np.eye(16)
    net.eval()

    cat_miou = {}
    for k in shapenet.seg_classes.keys():
        cat_miou[k] = [0, 0]
    miou = 0
    count = 0
    per_cat_miou = 0
    per_cat_count = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, ascii=True) as tq:
            for data, label, cat in tq:
                num_examples = data.shape[0]
                data = data.to(dev, dtype=torch.float)
                label = label.to(dev, dtype=torch.long)
                cat_ind = [category_list.index(c) for c in cat]
                cat_tensor = torch.tensor(eye_mat[cat_ind]).to(dev, dtype=torch.float).repeat(1, 2048)
                cat_tensor = cat_tensor.view(num_examples, -1, 16).permute(0,2,1)
                logits = net(data, cat_tensor)
                _, preds = logits.max(1)

                cat_miou = mIoU(preds.cpu().numpy(),
                                label.view(num_examples, -1).cpu().numpy(),
                                cat, cat_miou, shapenet.seg_classes)
                for _, v in cat_miou.items():
                    if v[1] > 0:
                        miou += v[0]
                        count += v[1]
                        per_cat_miou += v[0] / v[1]
                        per_cat_count += 1
                tq.set_postfix({
                    'mIoU': '%.5f' % (miou / count),
                    'per Category mIoU': '%.5f' % (miou / count)})
    if per_cat_verbose:
        print("Per-Category mIoU:")
        for k, v in cat_miou.items():
            if v[1] > 0:
                print("%s mIoU=%.5f" % (k, v[0] / v[1]))
            else:
                print("%s mIoU=%.5f" % (k, 1))
    return miou / count, per_cat_miou / per_cat_count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = "cpu"

net = PointNetPartSeg(50, 3, 2048)
net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
L = PartSegLoss()

shapenet = ShapeNet(2048, normal_channel=False)

train_loader = CustomDataLoader(shapenet.trainval())
test_loader = CustomDataLoader(shapenet.test())

best_test_miou = 0
best_test_per_cat_miou = 0

for epoch in range(args.num_epochs):
    train(net, opt, scheduler, train_loader, dev)
    if (epoch + 1) % 5 == 0:
        print('Epoch #%d Testing' % epoch)
        test_miou, test_per_cat_miou = evaluate(net, test_loader, dev, (epoch + 1) % 5 ==0)
        if test_miou > best_test_miou:
            best_test_miou = test_miou
            best_test_per_cat_miou = test_per_cat_miou
            if args.save_model_path:
                torch.save(net.state_dict(), args.save_model_path)
        print('Current test mIoU: %.5f (best: %.5f), per-Category mIoU: %.5f (best: %.5f)' % (
               test_miou, best_test_miou, test_per_cat_miou, best_test_per_cat_miou))
