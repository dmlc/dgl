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

from load_graph import load_dataset, inductive_split

from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
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
        # The usage of `train_acc` and `val_acc` is the recommended practice from now on as per
        # https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.num_steps = 0
        self.cum_sampled_nodes = [0 for _ in range(n_layers + 1)]
        self.cum_sampled_edges = [0 for _ in range(n_layers)]
        self.w = 0.99
        self.loss_fn = nn.NLLLoss() if not multilabel else nn.BCELoss()
        self.final_activation = nn.LogSoftmax(dim=1) if not multilabel else nn.Sigmoid()
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
        loss = self.loss_fn(self.final_activation(batch_pred), batch_labels)
        self.train_acc(self.final_activation(batch_pred), batch_labels.int())
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
        loss = self.loss_fn(self.final_activation(batch_pred), batch_labels)
        self.val_acc(self.final_activation(batch_pred), batch_labels.int())
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_labels.shape[0])
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_labels.shape[0])

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_cpu=False, graph_cpu=False, use_uva=False, fan_out=[10, 25],
                 device=th.device('cpu'), batch_size=1000, num_workers=4, sampler='labor', importance_sampling=0, layer_dependency=False, batch_dependency=1, cache_size=0):
        super().__init__()

        g, n_classes, multilabel = load_dataset(dataset_name)
        cast_to_int = max(g.num_nodes(), g.num_edges()) <= 2e9
        if cast_to_int:
            g = g.int()

        train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = th.nonzero(~(g.ndata['train_mask'] | g.ndata['val_mask']), as_tuple=True)[0]

        fanouts = [int(_) for _ in fan_out]
        if sampler == 'neighbor':
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts) #, prefetch_node_feats='features', prefetch_labels='labels')
        else:
            sampler = dgl.dataloading.LaborSampler(fanouts, importance_sampling=importance_sampling, layer_dependency=layer_dependency, batch_dependency=batch_dependency) #, prefetch_node_feats='features', prefetch_edge_feats='edge_weights', prefetch_labels='labels')

        dataloader_device = th.device('cpu')
        if use_uva or (not data_cpu and not graph_cpu):
            train_nid = train_nid.to(device)
            val_nid = val_nid.to(device)
            test_nid = test_nid.to(device)
            g = g.formats(['csc', 'csr', 'coo'])
            if not data_cpu:
                g = g.to(device)
            elif not graph_cpu:
                g._graph = g._graph.copy_to(dgl.utils.to_dgl_context(device))
            if use_uva:
                if graph_cpu:
                    g._graph.pin_memory_()
                if data_cpu:
                    for frame in itertools.chain(g._node_frames, g._edge_frames):
                        for col in frame._columns.values():
                            col.pin_memory_()
            dataloader_device = device

        self.g = g
        if cast_to_int:
            self.train_nid, self.val_nid, self.test_nid = train_nid.int(), val_nid.int(), test_nid.int()
        else:
            self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.sampler = sampler
        self.val_sampler = sampler # dgl.dataloading.MultiLayerFullNeighborSampler(len(fanouts))
        self.device = dataloader_device
        self.use_uva = use_uva
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_feats = g.ndata['features'].shape[1]
        self.n_classes = n_classes
        self.multilabel = multilabel
        try:
            self.cache = dgl.contrib.GpuCache(cache_size, self.in_feats, th.int32) if cache_size > 0 else None
        except:
            self.cache = None

    def train_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.train_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.val_nid,
            self.val_sampler,
            device=self.device,
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
    
    def on_train_batch_start(self, trainer, datamodule, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        cache = trainer.datamodule.cache
        feats = trainer.datamodule.g.ndata['features']
        if cache is not None:
            values, missing_index, missing_keys = cache.query(input_nodes)
        else:
            missing_index = slice(0, input_nodes.shape[0])
            missing_keys = input_nodes
            values = th.empty([input_nodes.shape[0], feats.shape[1]], dtype=feats.dtype, device=trainer.datamodule.device)
        if feats.is_pinned():
            missing_values = dgl.utils.gather_pinned_tensor_rows(feats, missing_keys)
        else:
            missing_values = feats[missing_keys.long()].to(values)
        values[missing_index] = missing_values
        mfgs[0].srcdata['features'] = values
        if cache is not None:
            cache.replace(missing_keys, missing_values)
            trainer.strategy.model.log('cache_hit', 1 - missing_keys.shape[0] / input_nodes.shape[0], prog_bar=True, on_step=True, on_epoch=False)

    def on_train_batch_end(self, trainer, datamodule, outputs, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        self.push(mfgs[0].num_src_nodes())
    
    def on_train_epoch_end(self, trainer, datamodule):
        if self.limit > 0 and self.n >= 2 and abs(self.limit - self.m) * self.n >= self.std * self.factor:
            trainer.datamodule.batch_size = int(trainer.datamodule.batch_size * self.limit / self.m)
            trainer.reset_train_dataloader()
            trainer.reset_val_dataloader()
            self.clear()
    
def evaluate(model, g, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    nfeat = g.ndata['features']
    labels = g.ndata['labels']
    with th.no_grad():
        pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    test_acc = Accuracy()
    return test_acc(th.softmax(pred[val_nid.to(device=pred.device, dtype=th.int64)], -1), labels[val_nid.to(device=labels.device, dtype=th.int64)].to(pred.device))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=-1)
    argparser.add_argument('--num-steps', type=int, default=5000)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--graph-cpu', action='store_true',
                           help="By default the script puts the graph"
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--sampler', type=str, default='labor')
    argparser.add_argument('--importance-sampling', type=int, default=0)
    argparser.add_argument('--layer-dependency', action='store_true')
    argparser.add_argument('--batch-dependency', type=int, default=1)
    argparser.add_argument('--cache-size', type=int, default=0)
    argparser.add_argument('--logdir', type=str, default='tb_logs')
    argparser.add_argument('--vertex-limit', type=int, default=-1)
    argparser.add_argument('--use-uva', action='store_true')
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    datamodule = DataModule(
        args.dataset, args.data_cpu, args.graph_cpu, args.use_uva,
        [int(_) for _ in args.fan_out.split(',')],
        device, args.batch_size, args.num_workers, args.sampler, args.importance_sampling, args.layer_dependency, args.batch_dependency, args.cache_size)
    model = SAGELightning(
        datamodule.in_feats, args.num_hidden, datamodule.n_classes, args.num_layers,
        F.relu, args.dropout, args.lr, datamodule.multilabel)

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    batchsize_callback = BatchSizeCallback(args.vertex_limit)
    subdir = '{}_{}_{}_{}_{}'.format(args.dataset, args.sampler, args.importance_sampling, args.layer_dependency, args.batch_dependency)
    logger = TensorBoardLogger(args.logdir, name=subdir)
    trainer = Trainer(gpus=[args.gpu] if args.gpu != -1 else None,
                      max_epochs=args.num_epochs,
                      max_steps=args.num_steps,
                      callbacks=[checkpoint_callback, batchsize_callback],
                      logger=logger)
    trainer.fit(model, datamodule=datamodule)

    # Test
    logdir = os.path.join(args.logdir, subdir)
    dirs = glob.glob('./{}/*'.format(logdir))
    version = max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])
    logdir = './{}/version_{}'.format(logdir, version)
    print('Evaluating model in', logdir)
    ckpt = glob.glob(os.path.join(logdir, 'checkpoints', '*'))[0]

    model = SAGELightning.load_from_checkpoint(
        checkpoint_path=ckpt, hparams_file=os.path.join(logdir, 'hparams.yaml')).to(device)
    test_acc = evaluate(model, datamodule.g, datamodule.test_nid, device)
    print('Test accuracy:', test_acc)
