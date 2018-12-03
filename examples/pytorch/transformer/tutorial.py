"""
In current version we use multi30k as the training and validation set.
Multi-GPU support is required to train the model on WMT14.
"""
from tqdm import tqdm
import torch as th
import numpy as np

from loss import LabelSmoothing, SimpleLossCompute
from modules import make_model
from optims import NoamOpt
from dgl.contrib.transformer import get_dataset, GraphPool


def run_epoch(data_iter, models, loss_compute, is_train=True):
    for i, gs in tqdm(enumerate(data_iter)):
        with th.set_grad_enabled(is_train):
            outputs = models[0](gs[0])
            loss = loss_compute(outputs, [g.tgt_y for g in gs], [g.n_tokens for g in gs])
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))


np.random.seed(1111)
N = 6
batch_size = 128
devices = ['cuda' if th.cuda.is_available() else 'cpu']

dataset = get_dataset("multi30k")
V = dataset.vocab_size
criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
dim_model = 512

graph_pool = GraphPool()
model = make_model(V, V, N=N, dim_model=dim_model)

# Sharing weights between Encoder & Decoder
model.src_embed.lut.weight = model.tgt_embed.lut.weight
model.generator.proj.weight = model.tgt_embed.lut.weight

model, criterion = model.to(devices[0]), criterion.to(devices[0])
model_opt = NoamOpt(dim_model, 1, 400,
                    th.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
models, criterions = [model], [criterion]
loss_compute = SimpleLossCompute

for epoch in range(100):
    train_iter = dataset(graph_pool, mode='train', batch_size=batch_size, devices=devices)
    valid_iter = dataset(graph_pool, mode='valid', batch_size=batch_size, devices=devices)
    print('Epoch: {} Training...'.format(epoch))
    model.train(True)
    run_epoch(train_iter, models,
              loss_compute(criterions, model_opt), is_train=True)
    print('Epoch: {} Evaluating...'.format(epoch))
    model.att_weight_map = None
    model.eval()
    run_epoch(valid_iter, models,
              loss_compute(criterions, None), is_train=False)
