# /*!
#  *   Copyright (c) 2022, NVIDIA Corporation
#  *   Copyright (c) 2022, GT-TDAlab (Muhammed Fatih Balin & Umit V. Catalyurek)  
#  *   All rights reserved.
#  *
#  *   Licensed under the Apache License, Version 2.0 (the "License");
#  *   you may not use this file except in compliance with the License.
#  *   You may obtain a copy of the License at
#  *
#  *       http://www.apache.org/licenses/LICENSE-2.0
#  *
#  *   Unless required by applicable law or agreed to in writing, software
#  *   distributed under the License is distributed on an "AS IS" BASIS,
#  *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  *   See the License for the specific language governing permissions and
#  *   limitations under the License.
#  *
#  * \file train_lightning_labor.py
#  * \brief labor sampling example
#  */

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
import glob
import os
import sys
import time
import math
import itertools

from load_graph import load_dataset

from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from model import SAGE

class SAGELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr,
                 multilabel):
        super().__init__()
        self.save_hyperparameters()
        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
        self.lr = lr
        f1score_class = MulticlassF1Score if not multilabel else MultilabelF1Score
        self.train_acc = f1score_class(n_classes, average='micro')
        self.val_acc = f1score_class(n_classes, average='micro')
        self.test_acc = f1score_class(n_classes, average='micro')
        self.num_steps = 0
        self.cum_sampled_nodes = [0 for _ in range(n_layers + 1)]
        self.cum_sampled_edges = [0 for _ in range(n_layers)]
        self.w = 0.99
        self.loss_fn = nn.CrossEntropyLoss() if not multilabel else nn.BCEWithLogitsLoss()
        self.pt = 0
    
    def num_sampled_nodes(self, i):
        return self.cum_sampled_nodes[i] / self.num_steps if self.w >= 1 else self.cum_sampled_nodes[i] * (1 - self.w) / (1 - self.w ** self.num_steps)
    
    def num_sampled_edges(self, i):
        return self.cum_sampled_edges[i] / self.num_steps if self.w >= 1 else self.cum_sampled_edges[i] * (1 - self.w) / (1 - self.w ** self.num_steps)
    
    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        self.num_steps += 1
        for i, mfg in enumerate(mfgs):
            self.cum_sampled_nodes[i] = self.cum_sampled_nodes[i] * self.w + mfg.num_src_nodes()
            self.cum_sampled_edges[i] = self.cum_sampled_edges[i] * self.w + mfg.num_edges()
            self.log('num_nodes[{}]'.format(i), self.num_sampled_nodes(i), prog_bar=True, on_step=True, on_epoch=False)
            self.log('num_edges[{}]'.format(i), self.num_sampled_edges(i), prog_bar=True, on_step=True, on_epoch=False)
        # for batch size monitoring
        i = len(mfgs)
        self.cum_sampled_nodes[i] = self.cum_sampled_nodes[i] * self.w + mfgs[-1].num_dst_nodes()
        self.log('num_nodes[{}]'.format(i), self.num_sampled_nodes(i), prog_bar=True, on_step=True, on_epoch=False)
        
        batch_inputs = mfgs[0].srcdata['features']
        batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.module(mfgs, batch_inputs)
        loss = self.loss_fn(batch_pred, batch_labels)
        self.train_acc(batch_pred, batch_labels.int())
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_labels.shape[0])
        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch_labels.shape[0])
        t = time.time()
        self.log('iter_time', t - self.pt, prog_bar=True, on_step=True, on_epoch=False)
        self.pt = t
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['features']
        batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.module(mfgs, batch_inputs)
        loss = self.loss_fn(batch_pred, batch_labels)
        self.val_acc(batch_pred, batch_labels.int())
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_labels.shape[0])
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_labels.shape[0])
    
    def test_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['features']
        batch_labels = mfgs[-1].dstdata['labels']
        batch_pred = self.module(mfgs, batch_inputs)
        loss = self.loss_fn(batch_pred, batch_labels)
        self.test_acc(batch_pred, batch_labels.int())
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_labels.shape[0])
        self.log('test_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_labels.shape[0])

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, dataset_name, undirected, data_cpu=False, use_uva=False, fan_out=[10, 25],
                 device=th.device('cpu'), batch_size=1000, num_workers=4, sampler='labor', importance_sampling=0, layer_dependency=False):
        super().__init__()

        g, n_classes, multilabel = load_dataset(dataset_name)
        if undirected:
            src, dst = g.all_edges()
            g.add_edges(dst, src)
        cast_to_int = max(g.num_nodes(), g.num_edges()) <= 2e9
        if cast_to_int:
            g = g.int()

        train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

        fanouts = [int(_) for _ in fan_out]
        if sampler == 'neighbor':
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts, prefetch_node_feats=['features'], prefetch_labels=['labels'])
        else:
            sampler = dgl.dataloading.LaborSampler(fanouts, importance_sampling=importance_sampling, layer_dependency=layer_dependency, prefetch_node_feats=['features'], prefetch_labels=['labels'])

        dataloader_device = th.device('cpu')
        g = g.formats(['csc'])
        if use_uva or not data_cpu:
            train_nid = train_nid.to(device)
            val_nid = val_nid.to(device)
            test_nid = test_nid.to(device)
            if not data_cpu and not use_uva:
                g = g.to(device)
            dataloader_device = device

        self.g = g
        if cast_to_int:
            self.train_nid, self.val_nid, self.test_nid = train_nid.int(), val_nid.int(), test_nid.int()
        else:
            self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.sampler = sampler
        self.val_sampler = sampler
        self.test_sampler = sampler
        self.device = dataloader_device
        self.use_uva = use_uva
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_feats = g.ndata['features'].shape[1]
        self.n_classes = n_classes
        self.multilabel = multilabel

    def train_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.train_nid,
            self.sampler,
            device=self.device,
            use_uva=self.use_uva,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.val_nid,
            self.val_sampler,
            device=self.device,
            use_uva=self.use_uva,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)
    
    def test_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.test_nid,
            self.test_sampler,
            device=self.device,
            use_uva=self.use_uva,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers)

class BatchSizeCallback(Callback):
    def __init__(self, limit, factor=3):
        super().__init__()
        self.limit = limit
        self.factor = factor
        self.clear()

    def clear(self):
        self.n = 0
        self.m = 0
        self.s = 0

    def push(self, x):
        self.n += 1
        m = self.m
        self.m += (x - m) / self.n
        self.s += (x - m) * (x - self.m)

    @property
    def var(self):
        return self.s / (self.n - 1)

    @property
    def std(self):
        return math.sqrt(self.var)
    
    def on_train_batch_end(self, trainer, datamodule, outputs, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        self.push(mfgs[0].num_src_nodes())
    
    def on_train_epoch_end(self, trainer, datamodule):
        if self.limit > 0 and self.n >= 2 and abs(self.limit - self.m) * self.n >= self.std * self.factor:
            trainer.datamodule.batch_size = int(trainer.datamodule.batch_size * self.limit / self.m)
            trainer.reset_train_dataloader()
            trainer.reset_val_dataloader()
            self.clear()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=-1)
    argparser.add_argument('--num-steps', type=int, default=-1)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--sampler', type=str, default='labor')
    argparser.add_argument('--importance-sampling', type=int, default=0)
    argparser.add_argument('--layer-dependency', action='store_true')
    argparser.add_argument('--logdir', type=str, default='tb_logs')
    argparser.add_argument('--vertex-limit', type=int, default=-1)
    argparser.add_argument('--use-uva', action='store_true')
    argparser.add_argument('--undirected', action='store_true')
    argparser.add_argument('--val-acc-target', type=float, default=1)
    argparser.add_argument('--early-stopping-patience', type=int, default=10)
    argparser.add_argument('--disable-checkpoint', action='store_true')
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    datamodule = DataModule(
        args.dataset, args.undirected, args.data_cpu, args.use_uva,
        [int(_) for _ in args.fan_out.split(',')],
        device, args.batch_size, args.num_workers, args.sampler, args.importance_sampling, args.layer_dependency)
    model = SAGELightning(
        datamodule.in_feats, args.num_hidden, datamodule.n_classes, args.num_layers,
        F.relu, args.dropout, args.lr, datamodule.multilabel)

    # Train
    callbacks = []
    if not args.disable_checkpoint:
        callbacks.append(ModelCheckpoint(monitor='val_acc', save_top_k=1))
    callbacks.append(BatchSizeCallback(args.vertex_limit))
    callbacks.append(EarlyStopping(monitor='val_acc', stopping_threshold=args.val_acc_target, mode='max', patience=args.early_stopping_patience))
    subdir = '{}_{}_{}_{}'.format(args.dataset, args.sampler, args.importance_sampling, args.layer_dependency)
    logger = TensorBoardLogger(args.logdir, name=subdir)
    trainer = Trainer(gpus=[args.gpu] if args.gpu != -1 else None,
                      max_epochs=args.num_epochs,
                      max_steps=args.num_steps,
                      callbacks=callbacks,
                      logger=logger)
    trainer.fit(model, datamodule=datamodule)

    # Test
    if not args.disable_checkpoint:
        logdir = os.path.join(args.logdir, subdir)
        dirs = glob.glob('./{}/*'.format(logdir))
        version = max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])
        logdir = './{}/version_{}'.format(logdir, version)
        print('Evaluating model in', logdir)
        ckpt = glob.glob(os.path.join(logdir, 'checkpoints', '*'))[0]

        model = SAGELightning.load_from_checkpoint(
            checkpoint_path=ckpt, hparams_file=os.path.join(logdir, 'hparams.yaml')).to(device)
    test_acc = trainer.test(model, datamodule=datamodule)
    print('Test accuracy:', test_acc)
