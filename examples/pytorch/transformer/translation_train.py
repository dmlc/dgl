"""
In current version we use multi30k as the training and validation set.
Multi-GPU support is required to train the model on WMT14.
"""
from modules import *
from parallel import *
from modules.utils import * 
from loss import * 
from optims import *
from dataset import *
from tqdm import tqdm
import numpy as np
import argparse

def run_epoch(data_iter, models, loss_compute, is_train=True):
    for i, gs in tqdm(enumerate(data_iter)):
        with T.set_grad_enabled(is_train):
            models = models[:len(gs)]
            outputs = parallel_apply(models, gs)
            loss = loss_compute(outputs, [g.tgt_y for g in gs], [g.n_tokens for g in gs])
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))

if __name__ == '__main__':
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    np.random.seed(1111)
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='enc/dec layers')
    argparser.add_argument('--sparse', action='store_true', help='use sparse connection')
    argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=128, type=int, help='batch size')
    args = argparser.parse_args()
    args_filter = ['batch', 'gpus']
    exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
    devices = ['cpu'] if args.gpus == '-1' else [int(gpu_id) for gpu_id in args.gpus.split(',')]

    dataset = get_dataset(args.dataset)
    V = dataset.vocab_size
    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    dim_model = 512

    graph_pool = GraphPool(sparse=args.sparse)
    model = make_model(V, V, N=args.N, dim_model=dim_model)

    # Sharing weights between Encoder & Decoder
    model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
    model.generator.proj.weight = model.tgt_embed[0].lut.weight

    model, criterion = model.to(devices[0]), criterion.to(devices[0])
    model_opt = NoamOpt(dim_model, 1, 400,
                        T.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
    if len(devices) > 1:
        models, criterions = map(nn.parallel.replicate, [model, criterion], [devices, devices])
    else:
        models, criterions = [model], [criterion]
    loss_compute = SimpleLossCompute if len(devices) == 1 else MultiGPULossCompute

    for epoch in range(100):
        train_iter = dataset(graph_pool, mode='train', batch_size=args.batch, devices=devices)
        valid_iter = dataset(graph_pool, mode='valid', batch_size=args.batch, devices=devices)
        print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        run_epoch(train_iter, models,
                      loss_compute(criterions, model_opt), is_train=True)
        print('Epoch: {} Evaluating...'.format(epoch))
        model.eval()
        run_epoch(valid_iter, models,
                      loss_compute(criterions, None), is_train=False)
        with open('checkpoints/{}-{}.pkl'.format(exp_setting, epoch), 'wb') as f:
            th.save(model.state_dict(), f)

