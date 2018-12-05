"""
In current version we use multi30k as the default training and validation set.
Multi-GPU support is required to train the model on WMT14.
"""
from modules import *
from parallel import *
from loss import * 
from optims import *
from dataset import *
from modules.config import *
from modules.viz import *
from tqdm import tqdm
import numpy as np
import argparse

def run_epoch(data_iter, model, loss_compute, is_train=True):
    universal = isinstance(model, UTransformer)
    for i, g in tqdm(enumerate(data_iter)):
        with T.set_grad_enabled(is_train):
            if isinstance(model, list):
                model = model[:len(gs)]
                output = parallel_apply(model, g)
                tgt_y = [g.tgt_y for g in gs]
                n_tokens = [g.n_tokens for g in gs]
            else:
                if universal:
                    output, loss_act = model(g)
                    if is_train: loss_act.backward(retain_graph=True)
                else:
                    output = model(g)
                tgt_y = g.tgt_y
                n_tokens = g.n_tokens
            loss = loss_compute(output, tgt_y, n_tokens)

    if universal:
        for step in range(1, model.MAX_DEPTH + 1):
            print("nodes entering step {}: {:.2f}%".format(step, (1.0 * model.stat[step] / model.stat[0])))
        model.reset_stat()
    print('average loss: {}'.format(loss_compute.avg_loss))
    print('accuracy: {}'.format(loss_compute.accuracy))

if __name__ == '__main__':
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    np.random.seed(1111)
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='enc/dec layers')
    argparser.add_argument('--dataset', default='multi30k', help='dataset')
    argparser.add_argument('--batch', default=128, type=int, help='batch size')
    argparser.add_argument('--viz', action='store_true', help='visualize attention')
    argparser.add_argument('--universal', action='store_true', help='use universal transformer')
    args = argparser.parse_args()
    args_filter = ['batch', 'gpus', 'viz']
    exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
    devices = ['cpu'] if args.gpus == '-1' else [int(gpu_id) for gpu_id in args.gpus.split(',')]

    dataset = get_dataset(args.dataset)

    V = dataset.vocab_size
    criterion = LabelSmoothing(V, padding_idx=dataset.pad_id, smoothing=0.1)
    dim_model = 512

    graph_pool = GraphPool()
    model = make_model(V, V, N=args.N, dim_model=dim_model, universal=args.universal)

    # Sharing weights between Encoder & Decoder
    model.src_embed.lut.weight = model.tgt_embed.lut.weight
    model.generator.proj.weight = model.tgt_embed.lut.weight

    model, criterion = model.to(devices[0]), criterion.to(devices[0])
    model_opt = NoamOpt(dim_model, 1, 400,
                        T.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
    if len(devices) > 1:
        model, criterion = map(nn.parallel.replicate, [model, criterion], [devices, devices])
    loss_compute = SimpleLossCompute if len(devices) == 1 else MultiGPULossCompute

    for epoch in range(100):
        train_iter = dataset(graph_pool, mode='train', batch_size=args.batch, devices=devices)
        valid_iter = dataset(graph_pool, mode='valid', batch_size=args.batch, devices=devices)
        print('Epoch: {} Training...'.format(epoch))
        model.train(True)
        run_epoch(train_iter, model,
                      loss_compute(criterion, model_opt), is_train=True)
        print('Epoch: {} Evaluating...'.format(epoch))
        model.att_weight_map = None
        model.eval()
        run_epoch(valid_iter, model,
                      loss_compute(criterion, None), is_train=False)
        # Visualize attention
        if args.viz:
            src_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='src')
            tgt_seq = dataset.get_seq_by_id(VIZ_IDX, mode='valid', field='tgt')[:-1]
            draw_atts(model.att_weight_map, src_seq, tgt_seq, exp_setting, 'epoch_{}'.format(epoch))

        print('----------------------------------')
        with open('checkpoints/{}-{}.pkl'.format(exp_setting, epoch), 'wb') as f:
            th.save(model.state_dict(), f)

