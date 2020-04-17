import torch
# torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
from dgl.data.utils import download, get_download_dir

from functools import partial
import tqdm
import urllib
import os
import argparse

# from dataset import ModelNet
import provider
from ModelNetDataLoader import ModelNetDataLoader
from pointnet_cls import PointNetCls, compute_loss
from pointnet2 import PointNet2SSGCls, PointNet2MSGCls

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default='')
parser.add_argument('--load-model-path', type=str, default='')
parser.add_argument('--save-model-path', type=str, default='')
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=16)
args = parser.parse_args()

num_workers = args.num_workers
batch_size = args.batch_size

data_filename = 'modelnet40_normal_resampled.zip'
download_path = os.path.join(get_download_dir(), data_filename)
local_path = args.dataset_path or os.path.join(get_download_dir(), modelnet40_normal_resampled)

if not os.path.exists(local_path):
    download('https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip', download_path)
    from zipfile import ZipFile
    with ZipFile(fpath) as z:
        z.extractall(path=local_path)
'''
data_filename = 'modelnet40-sampled-2048.h5'
local_path = args.dataset_path or os.path.join(get_download_dir(), data_filename)

if not os.path.exists(local_path):
    download('https://data.dgl.ai/dataset/modelnet40-sampled-2048.h5', local_path)
'''

CustomDataLoader = partial(
        DataLoader,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

def train(net, opt, scheduler, train_loader, dev):

    net.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader, ascii=True) as tq:
        # for data, label in tq:
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
            for data, label in tq:
                label = label[:,0]
                num_examples = label.shape[0]
                data, label = data.to(dev), label.to(dev).squeeze().long()
                logits = net(data)
                _, preds = logits.max(1)

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix({
                    'AvgAcc': '%.5f' % (total_correct / count)})

    return total_correct / count


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dev = "cpu"

# net = PointNetCls(40, input_dims=6)
net = PointNet2SSGCls(40, batch_size, input_dims=6)
# net = PointNet2MSGCls(40, batch_size)
net = net.to(dev)
if args.load_model_path:
    net.load_state_dict(torch.load(args.load_model_path, map_location=dev))

opt = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)

'''
modelnet = ModelNet(local_path, 1024)

train_loader = CustomDataLoader(modelnet.train())
test_loader = CustomDataLoader(modelnet.test())
'''

train_dataset = ModelNetDataLoader(local_path, 1024, split='train')
test_dataset = ModelNetDataLoader(local_path, 1024, split='test')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

best_test_acc = 0

for epoch in range(args.num_epochs):
    train(net, opt, scheduler, train_loader, dev)
    if (epoch + 1) % 1 == 0:
        print('Epoch #%d Testing' % epoch)
        test_acc = evaluate(net, test_loader, dev)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_model_path:
                torch.save(net.state_dict(), args.save_model_path)
        print('Current test acc: %.5f (best: %.5f)' % (
               test_acc, best_test_acc))
